"""Course video page-by-page transcript - FastAPI backend."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import Body, FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pipeline import merge_videos, process, to_srt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("srt_gen.app")

BASE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SG_DATA_DIR", BASE / "data"))
UPLOADS = DATA / "uploads"
JOBS = DATA / "jobs"
for d in (UPLOADS, JOBS):
    d.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = Path(os.environ.get("SG_CONFIG", BASE / "config.json"))


def load_config() -> dict:
    cfg = {}
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            log.exception("Failed to parse config file: %s", CONFIG_PATH)
    else:
        log.warning("Config file not found: %s (see config.example.json)", CONFIG_PATH)
    return {
        "base_url": os.environ.get("OPENAI_BASE_URL") or cfg.get("base_url")
        or "http://127.0.0.1:8000/v1",
        "api_key": os.environ.get("OPENAI_API_KEY") or cfg.get("api_key") or "",
        "model": os.environ.get("LLM_MODEL") or cfg.get("llm_model") or "",
    }


_cfg = load_config()
OPENAI_BASE_URL = str(_cfg["base_url"]).rstrip("/")
OPENAI_API_KEY = _cfg["api_key"]
LLM_MODEL = _cfg["model"]
if not OPENAI_API_KEY:
    log.warning("No API key configured: LLM titles will fall back to plain text")
DEFAULT_WHISPER = os.environ.get("WHISPER_MODEL", "small")
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v", ".ts", ".mpg", ".mpeg", ".wmv"}

app = FastAPI(title="Course Video Page Transcript")
worker = ThreadPoolExecutor(max_workers=1)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

# ---------------------------------------------------------------- auth ------
USERS_FILE = DATA / "users.json"
SESSIONS_FILE = DATA / "sessions.json"
SESSION_COOKIE = "sg_session"
SESSION_TTL = 7 * 86400
PBKDF2_ITER = 200_000

users: dict[str, dict] = {}
sessions: dict[str, dict] = {}
auth_lock = threading.Lock()


def _load_map(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            log.exception("Failed to load %s", path)
    return {}


def save_users() -> None:
    USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=1),
                          encoding="utf-8")


def save_sessions() -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions, ensure_ascii=False),
                             encoding="utf-8")


with auth_lock:
    users = _load_map(USERS_FILE)
    sessions = {k: v for k, v in _load_map(SESSIONS_FILE).items()
                if v.get("exp", 0) > time.time()}
    save_sessions()


def hash_password(pw: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"),
                             bytes.fromhex(salt), PBKDF2_ITER)
    return f"{salt}${dk.hex()}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        salt, want = stored.split("$", 1)
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"),
                             bytes.fromhex(salt), PBKDF2_ITER)
    return hmac.compare_digest(dk.hex(), want)


def find_user(username: str) -> dict | None:
    for u in users.values():
        if u["username"] == username:
            return u
    return None


def create_user(username: str, pw_hash: str) -> dict:
    with auth_lock:
        first = not users
        role = "admin" if first else "user"
        status = "active" if first else "pending"
        uid = uuid.uuid4().hex[:12]
        u = {"id": uid, "username": username, "name": username,
             "pw": pw_hash, "role": role, "status": status,
             "created_at": datetime.now().isoformat(timespec="seconds"),
             "last_login": None}
        users[uid] = u
        save_users()
    log.info("Registered user %s (role=%s, status=%s)", username, role, status)
    return u


def login_session(resp, user: dict) -> None:
    token = secrets.token_urlsafe(32)
    with auth_lock:
        sessions[token] = {"uid": user["id"], "exp": time.time() + SESSION_TTL}
        user["last_login"] = datetime.now().isoformat(timespec="seconds")
        save_sessions()
        save_users()
    resp.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax",
                    max_age=SESSION_TTL)


def current_user(request: Request) -> dict | None:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    with auth_lock:
        s = sessions.get(token)
        if not s or s.get("exp", 0) < time.time():
            sessions.pop(token, None)
            save_sessions()
            return None
        return users.get(s["uid"])


def require_user(request: Request) -> dict:
    u = current_user(request)
    if not u:
        raise HTTPException(401, "Not signed in")
    return u


def public_user(u: dict) -> dict:
    return {"id": u["id"], "username": u["username"],
            "name": u.get("name") or u["username"], "role": u["role"],
            "status": u.get("status", "active")}


@app.post("/api/auth/register")
def auth_register(payload: dict = Body(...)) -> JSONResponse:
    username = str(payload.get("username", "")).strip().lower()
    password = str(payload.get("password", ""))
    if not re.fullmatch(r"[a-z0-9][a-z0-9._\-]{1,39}", username):
        raise HTTPException(400, "Username must be 2-40 chars: letters, digits, . _ -")
    if len(password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    if find_user(username):
        raise HTTPException(409, "Username already taken")
    u = create_user(username, hash_password(password))
    pending = u.get("status") != "active"
    resp = JSONResponse({"user": public_user(u), "pending": pending})
    if not pending:
        login_session(resp, u)
    return resp


@app.post("/api/auth/login")
def auth_login(payload: dict = Body(...)) -> JSONResponse:
    username = str(payload.get("username", "")).strip().lower()
    password = str(payload.get("password", ""))
    u = find_user(username)
    if not u or not verify_password(password, u["pw"]):
        raise HTTPException(401, "Invalid username or password")
    st = u.get("status", "active")
    if st == "pending":
        raise HTTPException(403, "Account is awaiting administrator approval")
    if st == "disabled":
        raise HTTPException(403, "Account has been disabled")
    resp = JSONResponse({"user": public_user(u)})
    login_session(resp, u)
    return resp


@app.post("/api/auth/logout")
def auth_logout(request: Request) -> JSONResponse:
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        with auth_lock:
            sessions.pop(token, None)
            save_sessions()
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@app.get("/api/auth/me")
def auth_me(request: Request) -> JSONResponse:
    u = current_user(request)
    if not u:
        raise HTTPException(401, "Not signed in")
    return JSONResponse(public_user(u))


@app.get("/api/admin/users")
def admin_users(request: Request) -> JSONResponse:
    u = require_user(request)
    if u["role"] != "admin":
        raise HTTPException(403, "Admin only")
    return JSONResponse([{
        "id": x["id"], "username": x["username"], "role": x["role"],
        "status": x.get("status", "active"),
        "created_at": x.get("created_at"), "last_login": x.get("last_login"),
    } for x in sorted(users.values(), key=lambda x: x.get("created_at") or "")])


@app.post("/api/admin/users/{user_id}/status")
def admin_set_status(request: Request, user_id: str,
                     payload: dict = Body(...)) -> JSONResponse:
    actor = require_user(request)
    if actor["role"] != "admin":
        raise HTTPException(403, "Admin only")
    status = str(payload.get("status", ""))
    if status not in ("active", "disabled", "pending"):
        raise HTTPException(400, "Status must be active, disabled or pending")
    if user_id == actor["id"] and status != "active":
        raise HTTPException(400, "You cannot deactivate your own account")
    target = users.get(user_id)
    if not target:
        raise HTTPException(404, "User not found")
    with auth_lock:
        target["status"] = status
        if status != "active":
            for tok, s in list(sessions.items()):
                if s["uid"] == user_id:
                    del sessions[tok]
        save_users()
        save_sessions()
    log.info("Admin %s set status of %s to %s",
             actor["username"], target["username"], status)
    return JSONResponse({"ok": True})


@app.post("/api/admin/users/{user_id}/role")
def admin_set_role(request: Request, user_id: str,
                   payload: dict = Body(...)) -> JSONResponse:
    actor = require_user(request)
    if actor["role"] != "admin":
        raise HTTPException(403, "Admin only")
    role = str(payload.get("role", ""))
    if role not in ("admin", "user"):
        raise HTTPException(400, "Role must be admin or user")
    if user_id == actor["id"]:
        raise HTTPException(400, "You cannot change your own role")
    target = users.get(user_id)
    if not target:
        raise HTTPException(404, "User not found")
    with auth_lock:
        target["role"] = role
        save_users()
    log.info("Admin %s changed role of %s to %s",
             actor["username"], target["username"], role)
    return JSONResponse({"ok": True})


def load_persisted() -> None:
    if not JOBS.exists():
        return
    for d in sorted(JOBS.iterdir()):
        jf = d / "job.json"
        if not jf.exists():
            continue
        try:
            job = json.loads(jf.read_text(encoding="utf-8"))
            if job.get("status") in ("queued", "running"):
                job["status"] = "error"
                job["error"] = "Job aborted by server restart"
            jobs[job["id"]] = job
        except Exception:  # noqa: BLE001
            log.exception("Failed to load job: %s", jf)


load_persisted()


def save_job(job: dict) -> None:
    (JOBS / job["id"] / "job.json").write_text(
        json.dumps(job, ensure_ascii=False), encoding="utf-8")


def update_job(job_id: str, **fields) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job.update(fields)
        save_job(job)


def set_progress(job_id: str, step: str, pct: float) -> None:
    update_job(job_id, step=step, progress=round(min(pct, 0.999), 3))


def llm_model_id() -> str:
    global LLM_MODEL
    if LLM_MODEL:
        return LLM_MODEL
    try:
        r = httpx.get(f"{OPENAI_BASE_URL}/models",
                      headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                      timeout=10)
        r.raise_for_status()
        ids = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
        if ids:
            LLM_MODEL = ids[0]
            return LLM_MODEL
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to fetch model list: %s", e)
    return ""


def chat(prompt: str, max_tokens: int = 128, temperature: float = 0.3) -> str | None:
    model = llm_model_id()
    if not model:
        return None
    base = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature}
    payloads = [dict(base, chat_template_kwargs={"enable_thinking": False}), base]
    for payload in payloads:
        try:
            r = httpx.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json=payload, timeout=90)
            if r.status_code >= 400:
                log.warning("LLM HTTP %s", r.status_code)
                continue
            content = r.json()["choices"][0]["message"].get("content") or ""
            content = re.sub(r"<[^>]+>", "", content).strip().strip('"').strip()
            if content:
                return content
        except Exception as e:  # noqa: BLE001
            log.warning("LLM call failed: %s", e)
    return None


def page_title(page: dict) -> str:
    text = " ".join(s["text"] for s in page["segments"])
    fallback = re.sub(r"\s+", " ", page["segments"][0]["text"])[:30]
    if not text:
        return "(no speech)"
    out = chat(
        "Below is a transcript excerpt from one slide of a lecture video. "
        "Write a topic title of at most 10 words, in the same language as the "
        "transcript. Output ONLY the title: no quotes, no line breaks, no "
        f"trailing punctuation.\n\nTranscript: {text[:1500]}", max_tokens=48)
    out = (out or "").splitlines()[0].strip() if out else ""
    return out[:60] if out else fallback


def correction_guide(segments: list[dict]) -> str | None:
    draft = " ".join(s["text"] for s in segments)[:8000]
    if not draft.strip():
        return None
    out = chat(
        "The text below is a raw ASR (speech-to-text) draft transcript of a "
        "lecture; it contains recognition errors. Write a proofreading guide "
        "(a prompt) that another LLM will follow to correct this transcript. "
        "Reply in the transcript's main language, under 400 words, with these "
        "sections:\n"
        "1. 2-3 sentence summary of the lecture topic and context\n"
        "2. Domain background knowledge needed to understand it\n"
        "3. Likely correct technical terms, names and glossary inferred from "
        "context (bullet list)\n"
        "4. Expected ASR error patterns here (wrong homophones, split/merged "
        "words) with examples if visible\n"
        "5. Hard rules: keep the language, one output line per input line, no "
        "translation/summarization/rephrasing, only fix recognition errors\n"
        "Output ONLY the guide.\n\n" + draft,
        max_tokens=800, temperature=0.3)
    return out.strip() if out else None


def correct_all(segments: list[dict], guide: str,
                progress=lambda done, total: None) -> int:
    batches: list[list[dict]] = []
    cur: list[dict] = []
    size = 0
    for s in segments:
        cur.append(s)
        size += len(s["text"]) + 8
        if size >= 1500:
            batches.append(cur)
            cur, size = [], 0
    if cur:
        batches.append(cur)
    changed = 0
    for bi, batch in enumerate(batches):
        prev_txt = (" ".join(s["text"] for s in batches[bi - 1][-2:]))[-200:] if bi else ""
        next_txt = (" ".join(s["text"] for s in batches[bi + 1][:2]))[:200] \
            if bi + 1 < len(batches) else ""
        numbered = "\n".join(f"{n}: {s['text']}" for n, s in enumerate(batch, 1))
        out = chat(
            "Below are numbered ASR transcript lines from a lecture. Using the "
            "guide, correct the speech-recognition errors (wrong homophones, "
            "split/merged words, missing or wrong punctuation). Keep the "
            "original language; do not translate, summarize or rephrase; do not "
            "change the meaning; keep exactly one output line per input line "
            "with the same number prefix. Output ONLY the numbered corrected "
            f"lines.\n\nGUIDE:\n{guide}\n\n"
            f"Context BEFORE (reference only, do not output): "
            f"{prev_txt or '(start of transcript)'}\n\n"
            f"Lines to correct:\n{numbered}\n\n"
            f"Context AFTER (reference only, do not output): "
            f"{next_txt or '(end of transcript)'}",
            max_tokens=max(256, min(4096, int(len(numbered) * 1.8))),
            temperature=0.1)
        progress(bi + 1, len(batches))
        if not out:
            continue
        fixed: dict[int, str] = {}
        for line in out.splitlines():
            line = line.strip().lstrip("*")
            m = re.match(r"^(\d{1,4})\s*[:：.、]\s*(.*)$", line)
            if m and 1 <= int(m.group(1)) <= len(batch):
                fixed.setdefault(int(m.group(1)), m.group(2).strip())
        if len(fixed) != len(batch):
            log.warning("Proofread batch %d/%d: got %d/%d lines, keeping "
                        "original", bi + 1, len(batches), len(fixed), len(batch))
            continue
        for n, s in enumerate(batch, 1):
            new = fixed[n]
            if new and new != s["text"]:
                s["text"] = new
                changed += 1
    return changed


def job_meta(text: str) -> dict | None:
    out = chat(
        "Below is material from a lecture video transcript (page titles and/or "
        "numbered subtitle lines). Reply with ONLY a JSON object having exactly "
        "these four string keys:\n"
        'title_zh: concise Chinese title of the lecture (<= 24 chars)\n'
        'title_en: concise English title of the lecture (<= 60 chars)\n'
        'summary_zh: 2-4 sentence Chinese summary of the lecture\n'
        'summary_en: the same summary in English (2-4 sentences)\n\n'
        + text,
        max_tokens=500, temperature=0.2)
    if not out:
        return None
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group())
    except Exception:  # noqa: BLE001
        return None
    keys = ("title_zh", "title_en", "summary_zh", "summary_en")
    if not all(isinstance(d.get(k), str) and d[k].strip() for k in keys):
        return None
    return {k: d[k].strip() for k in keys}


def video_title(pages: list[dict], filename: str) -> str:
    joined = "; ".join(p.get("title", "") for p in pages)[:2000]
    if not joined.strip():
        return Path(filename).stem
    out = chat(
        "Below is the list of per-slide topic titles of a lecture course. "
        "Write one overall course title of at most 12 words, in the same "
        "language as the titles. Output ONLY the title.\n\n" + joined,
        max_tokens=40)
    out = (out or "").splitlines()[0].strip() if out else ""
    return out[:80] if out else Path(filename).stem


def build_markdown(job: dict) -> str:
    lines = [f"# {job['result']['title']}", "",
             f"- Source file: {job['filename']}",
             f"- Duration: {fmt_ts(job['result']['duration'])}",
             f"- Detected language: {job['result']['language']}",
             f"- Pages: {len(job['result']['pages'])}", ""]
    for p in job["result"]["pages"]:
        lines += [f"## Page {p['index']} - {p['title']}",
                  f"*{fmt_ts(p['start'])} - {fmt_ts(p['end'])}*", ""]
        for s in p["segments"]:
            lines.append(f"`[{fmt_ts(s['start'])}]` {s['text']}")
        lines.append("")
    return "\n".join(lines)


def fmt_ts(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def run_job(job_id: str) -> None:
    job = jobs[job_id]
    outdir = JOBS / job_id
    try:
        update_job(job_id, status="running", step="Preparing", progress=0.0)
        parts = [Path(p) for p in job["params"].get("parts") or []]
        merged = len(parts) > 1 or bool(job.get("merged_from"))
        video = Path(job["video_path"])
        if len(parts) > 1:
            set_progress(job_id, "Merging videos", 0.01)
            out = UPLOADS / f"{job_id}{parts[0].suffix or '.mp4'}"
            merge_videos(parts, out)
            for pt in parts:
                pt.unlink(missing_ok=True)
            video = out
            update_job(job_id, video_path=str(video), merged_from=True)
            set_progress(job_id, "Merging videos", 0.04)

        def prog(step: str, pct: float) -> None:
            set_progress(job_id, step,
                         0.05 + 0.83 * pct if merged else 0.88 * pct)

        result = process(video, outdir,
                         model_size=job["params"]["model"],
                         language=job["params"]["language"],
                         scene_threshold=job["params"]["scene_threshold"],
                         reuse_segments=job["params"].get("reuse_segments", False),
                         progress=prog)
        result["merged"] = merged
        pages = result["pages"]

        if job["params"].get("correct", 1) and result["segments"]:
            set_progress(job_id, "Proofreading transcript", 0.85)
            guide = correction_guide(result["segments"])
            changed = 0
            if guide:
                def cb(done_ct: int, total: int) -> None:
                    set_progress(job_id, "Proofreading transcript",
                                 0.86 + 0.06 * done_ct / max(total, 1))
                changed = correct_all(result["segments"], guide, cb)
            else:
                log.warning("Job %s: correction guide unavailable, keeping "
                            "draft transcript", job_id)
            (outdir / "segments.json").write_text(
                json.dumps({"language": result["language"],
                            "segments": result["segments"]},
                           ensure_ascii=False, indent=1), encoding="utf-8")
            log.info("Job %s proofread (2-round): %d/%d lines changed", job_id,
                     changed, len(result["segments"]))

        set_progress(job_id, "Generating page titles", 0.955)
        with ThreadPoolExecutor(max_workers=4) as pool:
            titles = list(pool.map(page_title, pages))
        for page, title in zip(pages, titles):
            page["title"] = title
            page["text"] = "\n".join(s["text"] for s in page["segments"])

        set_progress(job_id, "Generating course title", 0.985)
        result["title"] = video_title(pages, job["filename"])
        result["pages"] = pages
        result["meta"] = job_meta("\n".join(
            f"{p.get('title') or ''}: "
            + " ".join(s["text"] for s in p["segments"])[:150]
            for p in pages)[:4500])

        (outdir / "full.srt").write_text(to_srt(result["segments"]), encoding="utf-8")
        with jobs_lock:
            jobs[job_id].update(status="done", step="Done", progress=1.0,
                                result=result)
        save_job(jobs[job_id])
        (outdir / "transcript.md").write_text(build_markdown(jobs[job_id]),
                                              encoding="utf-8")
        log.info("Job %s finished: %s pages", job_id, len(pages))
    except Exception as e:  # noqa: BLE001
        log.exception("Job %s failed", job_id)
        update_job(job_id, status="error", error=str(e))


@app.post("/api/jobs")
async def create_job(request: Request,
                     files: list[UploadFile] = File(default=[]),
                     model: str = Form(DEFAULT_WHISPER),
                     language: str = Form("auto"),
                     scene_threshold: float = Form(0.03),
                     reuse_from: str | None = Form(None),
                     correct: int = Form(1)) -> JSONResponse:
    user = require_user(request)
    if not 0.005 <= scene_threshold <= 0.5:
        raise HTTPException(400, "scene_threshold must be between 0.005 and 0.5")
    if model not in {"tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"}:
        raise HTTPException(400, f"Unsupported model: {model}")

    job_id = uuid.uuid4().hex[:12]
    (JOBS / job_id).mkdir(parents=True, exist_ok=True)

    files = [f for f in (files or []) if f and f.filename]
    reuse_segments = False
    parts: list[str] = []
    video_path = None
    filename = None
    merged_from = False
    if reuse_from and not files:
        src = get_job_access(reuse_from, user)
        seg_src = JOBS / reuse_from / "segments.json"
        if not seg_src.exists():
            raise HTTPException(400, "Source job has no reusable transcript")
        video_path = Path(src["video_path"])
        if not video_path.exists():
            raise HTTPException(400, "Source job video file no longer exists")
        filename = src["filename"]
        merged_from = bool(src.get("merged_from")) or \
            bool((src.get("result") or {}).get("merged"))
        shutil.copyfile(seg_src, JOBS / job_id / "segments.json")
        reuse_segments = True
    if files:
        for i, f in enumerate(files):
            ext = Path(f.filename).suffix.lower() or ".mp4"
            if ext not in VIDEO_EXTS:
                raise HTTPException(400, f"Unsupported file format: {ext}")
            part = UPLOADS / f"{job_id}_p{i:02d}{ext}"
            with part.open("wb") as fh:
                shutil.copyfileobj(f.file, fh)
            parts.append(str(part))
        names = [Path(f.filename).name for f in files]
        filename = names[0] if len(names) == 1 else f"{names[0]} (+{len(names) - 1} more)"
        video_path = Path(parts[0])
    if not filename:
        raise HTTPException(400, "Missing file")

    job = {
        "id": job_id,
        "owner": user["id"],
        "filename": Path(filename).name,
        "merged_from": merged_from,
        "export_stem": re.sub(r"[^\w\-]+", "_",
                              Path(files[0].filename if files
                                   else filename).stem)[:40] or "transcript",
        "video_path": str(video_path),
        "status": "queued", "step": "Queued", "progress": 0.0, "error": None,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "params": {"model": model, "language": language,
                   "scene_threshold": scene_threshold,
                   "reuse_segments": reuse_segments,
                   "reuse_from": reuse_from,
                   "correct": bool(correct),
                   "parts": parts},
        "result": None,
    }
    with jobs_lock:
        jobs[job_id] = job
    save_job(job)
    worker.submit(run_job, job_id)
    return JSONResponse({"id": job_id})


def get_job_or_404(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


def get_job_access(job_id: str, user: dict) -> dict:
    job = get_job_or_404(job_id)
    if user["role"] != "admin" and job.get("owner") != user["id"]:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/jobs")
def list_jobs(request: Request) -> JSONResponse:
    user = require_user(request)
    items = [j for j in jobs.values()
             if user["role"] == "admin" or j.get("owner") == user["id"]]
    items = sorted(items, key=lambda j: j["created_at"], reverse=True)
    return JSONResponse([{
        "id": j["id"], "filename": j["filename"], "status": j["status"],
        "created_at": j["created_at"],
        "owner": (users.get(j.get("owner") or "", {}) or {}).get("username"),
        "title": (j.get("result") or {}).get("title"),
        "meta": (j.get("result") or {}).get("meta"),
        "pages": len((j.get("result") or {}).get("pages", [])),
        "duration": (j.get("result") or {}).get("duration"),
    } for j in items])


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, request: Request) -> JSONResponse:
    return JSONResponse(get_job_access(job_id, require_user(request)))


@app.post("/api/jobs/{job_id}/meta")
def gen_job_meta(job_id: str, request: Request) -> JSONResponse:
    job = get_job_access(job_id, require_user(request))
    if job["status"] != "done":
        raise HTTPException(400, "Job is not finished")
    result = job.get("result") or {}
    if result.get("meta"):
        return JSONResponse(result["meta"])
    outline = "\n".join(
        f"{p.get('title') or ''}: " + " ".join(s["text"] for s in p["segments"])[:150]
        for p in result.get("pages") or [])
    if not outline and (JOBS / job_id / "full.srt").exists():
        outline = (JOBS / job_id / "full.srt").read_text(encoding="utf-8")
    if not outline:
        raise HTTPException(404, "No transcript available")
    meta = job_meta(outline[:4500])
    if not meta:
        raise HTTPException(502, "Meta generation failed")
    with jobs_lock:
        jobs[job_id]["result"] = {**(jobs[job_id].get("result") or {}), "meta": meta}
    save_job(jobs[job_id])
    return JSONResponse(meta)


@app.get("/api/jobs/{job_id}/thumb/{page_index}")
def job_thumb(job_id: str, page_index: int, request: Request) -> FileResponse:
    job = get_job_access(job_id, require_user(request))
    pages = (job.get("result") or {}).get("pages", [])
    if not 1 <= page_index <= len(pages):
        raise HTTPException(404, "Page not found")
    thumb = (JOBS / job_id / "pages" / pages[page_index - 1]["thumb"])
    if not thumb.exists():
        raise HTTPException(404, "No thumbnail")
    return FileResponse(thumb)


def range_response(path: Path, request: Request, media: str):
    file_size = path.stat().st_size
    header = request.headers.get("range")

    def iterfile(start: int, length: int):
        with path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    if not header:
        return StreamingResponse(
            iterfile(0, file_size), media_type=media,
            headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)})
    m = re.match(r"bytes=(\d*)-(\d*)", header)
    if not m:
        raise HTTPException(416, "Invalid Range header")
    start = int(m.group(1) or 0)
    end = int(m.group(2)) if m.group(2) else file_size - 1
    if start >= file_size or end >= file_size or start > end:
        return StreamingResponse([], status_code=416,
                                 headers={"Content-Range": f"bytes */{file_size}"})
    length = end - start + 1
    return StreamingResponse(
        iterfile(start, length), status_code=206, media_type=media,
        headers={"Content-Range": f"bytes {start}-{end}/{file_size}",
                 "Accept-Ranges": "bytes", "Content-Length": str(length)})


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str, request: Request, download: int = 0):
    job = get_job_access(job_id, require_user(request))
    path = Path(job["video_path"])
    if not path.exists():
        raise HTTPException(404, "Video file not found")
    if download:
        stem = job.get("export_stem") or re.sub(
            r"[^\w\-]+", "_", Path(job["filename"]).stem)[:40]
        return FileResponse(path, media_type="video/mp4",
                            filename=f"{stem}{path.suffix}")
    return range_response(path, request, "video/mp4")


@app.get("/api/jobs/{job_id}/export.{fmt}")
def job_export(job_id: str, fmt: str, request: Request) -> FileResponse:
    job = get_job_access(job_id, require_user(request))
    if job["status"] != "done":
        raise HTTPException(409, "Job not finished yet")
    mapping = {"srt": "full.srt", "md": "transcript.md"}
    if fmt not in mapping:
        raise HTTPException(400, "Only srt / md are supported")
    path = JOBS / job_id / mapping[fmt]
    if not path.exists():
        raise HTTPException(404, "File not found")
    stem = job.get("export_stem") or re.sub(
        r"[^\w\-]+", "_", Path(job["filename"]).stem)[:40]
    return FileResponse(path, filename=f"{stem}.{fmt}")


app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
