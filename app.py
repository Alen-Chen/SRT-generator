"""課程影片逐頁文字稿 - FastAPI 後端。"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pipeline import process, to_srt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("srt_gen.app")

BASE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SG_DATA_DIR", BASE / "data"))
UPLOADS = DATA / "uploads"
JOBS = DATA / "jobs"
for d in (UPLOADS, JOBS):
    d.mkdir(parents=True, exist_ok=True)

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "REDACTED")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
DEFAULT_WHISPER = os.environ.get("WHISPER_MODEL", "small")
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v", ".ts", ".mpg", ".mpeg", ".wmv"}

app = FastAPI(title="課程影片逐頁文字稿")
worker = ThreadPoolExecutor(max_workers=1)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


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
                job["error"] = "服務重啟導致任務中止"
            jobs[job["id"]] = job
        except Exception:  # noqa: BLE001
            log.exception("讀取 job 失敗: %s", jf)


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
        log.warning("取得模型清單失敗: %s", e)
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
            content = re.sub(r"<[^>]+>", "", content).strip().strip('"」『』「').strip()
            if content:
                return content
        except Exception as e:  # noqa: BLE001
            log.warning("LLM 呼叫失敗: %s", e)
    return None


def page_title(page: dict) -> str:
    text = " ".join(s["text"] for s in page["segments"])
    fallback = re.sub(r"\s+", " ", page["segments"][0]["text"])[:30]
    if not text:
        return "（無語音）"
    out = chat(
        f"以下是課程影片其中一段的文字稿。請用與內容相同的語言，"
        f"生成不超過 15 個字的話題標題，只輸出標題本身，不要引號、不要換行、結尾不加標點。\n\n"
        f"文字稿：{text[:1500]}", max_tokens=48)
    out = (out or "").splitlines()[0].strip() if out else ""
    return out[:60] if out else fallback


def video_title(pages: list[dict], filename: str) -> str:
    joined = "、".join(p.get("title", "") for p in pages)[:2000]
    if not joined.strip():
        return Path(filename).stem
    out = chat(
        f"以下是一堂課程各頁的話題清單。請用與內容相同的語言，"
        f"生成不超過 20 個字的課程總標題，只輸出標題本身。\n\n{joined}",
        max_tokens=40)
    out = (out or "").splitlines()[0].strip() if out else ""
    return out[:80] if out else Path(filename).stem


def build_markdown(job: dict) -> str:
    lines = [f"# {job['result']['title']}", "",
             f"- 來源檔案：{job['filename']}",
             f"- 影片長度：{fmt_ts(job['result']['duration'])}",
             f"- 偵測語言：{job['result']['language']}",
             f"- 頁數：{len(job['result']['pages'])}", ""]
    for p in job["result"]["pages"]:
        lines += [f"## 第 {p['index']} 頁 {p['title']}",
                  f"*{fmt_ts(p['start'])} – {fmt_ts(p['end'])}*", ""]
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
    video = Path(job["video_path"])
    outdir = JOBS / job_id
    try:
        update_job(job_id, status="running", step="準備中", progress=0.0)
        result = process(video, outdir,
                         model_size=job["params"]["model"],
                         language=job["params"]["language"],
                         scene_threshold=job["params"]["scene_threshold"],
                         reuse_segments=job["params"].get("reuse_segments", False),
                         progress=lambda step, pct: set_progress(job_id, step, pct))
        pages = result["pages"]

        set_progress(job_id, "生成頁面標題", 0.955)
        with ThreadPoolExecutor(max_workers=4) as pool:
            titles = list(pool.map(page_title, pages))
        for page, title in zip(pages, titles):
            page["title"] = title
            page["text"] = "\n".join(s["text"] for s in page["segments"])

        set_progress(job_id, "生成總標題", 0.985)
        result["title"] = video_title(pages, job["filename"])
        result["pages"] = pages

        (outdir / "full.srt").write_text(to_srt(result["segments"]), encoding="utf-8")
        with jobs_lock:
            jobs[job_id].update(status="done", step="完成", progress=1.0,
                                result=result)
        save_job(jobs[job_id])
        (outdir / "transcript.md").write_text(build_markdown(jobs[job_id]),
                                              encoding="utf-8")
        log.info("job %s 完成：%s 頁", job_id, len(pages))
    except Exception as e:  # noqa: BLE001
        log.exception("job %s 失敗", job_id)
        update_job(job_id, status="error", error=str(e))


@app.post("/api/jobs")
async def create_job(file: UploadFile | None = File(None),
                     model: str = Form(DEFAULT_WHISPER),
                     language: str = Form("auto"),
                     scene_threshold: float = Form(0.03),
                     reuse_from: str | None = Form(None)) -> JSONResponse:
    if not 0.005 <= scene_threshold <= 0.5:
        raise HTTPException(400, "scene_threshold 須介於 0.005–0.5")
    if model not in {"tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"}:
        raise HTTPException(400, f"不支援的模型：{model}")

    job_id = uuid.uuid4().hex[:12]
    (JOBS / job_id).mkdir(parents=True, exist_ok=True)

    reuse_segments = False
    video_path = None
    filename = None
    if reuse_from:
        src = get_job_or_404(reuse_from)
        seg_src = JOBS / reuse_from / "segments.json"
        if not seg_src.exists():
            raise HTTPException(400, "來源任務沒有可沿用的轉錄結果")
        video_path = Path(src["video_path"])
        if not video_path.exists():
            raise HTTPException(400, "來源任務的影片已不存在")
        filename = src["filename"]
        shutil.copyfile(seg_src, JOBS / job_id / "segments.json")
        reuse_segments = True
        if file is not None and file.filename:
            ext = Path(file.filename).suffix.lower() or ".mp4"
            if ext not in VIDEO_EXTS:
                raise HTTPException(400, f"不支援的檔案格式：{ext}")
            video_path = UPLOADS / f"{job_id}{ext}"
            with video_path.open("wb") as f:
                shutil.copyfileobj(file.file, f)
    else:
        if not file or not file.filename:
            raise HTTPException(400, "缺少檔案")
        ext = Path(file.filename).suffix.lower() or ".mp4"
        if ext not in VIDEO_EXTS:
            raise HTTPException(400, f"不支援的檔案格式：{ext}")
        filename = file.filename
        video_path = UPLOADS / f"{job_id}{ext}"
        with video_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

    job = {
        "id": job_id,
        "filename": Path(filename).name,
        "video_path": str(video_path),
        "status": "queued", "step": "排隊中", "progress": 0.0, "error": None,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "params": {"model": model, "language": language,
                   "scene_threshold": scene_threshold,
                   "reuse_segments": reuse_segments,
                   "reuse_from": reuse_from},
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
        raise HTTPException(404, "找不到任務")
    return job


@app.get("/api/jobs")
def list_jobs() -> JSONResponse:
    items = sorted(jobs.values(), key=lambda j: j["created_at"], reverse=True)
    return JSONResponse([{
        "id": j["id"], "filename": j["filename"], "status": j["status"],
        "created_at": j["created_at"],
        "title": (j.get("result") or {}).get("title"),
        "pages": len((j.get("result") or {}).get("pages", [])),
        "duration": (j.get("result") or {}).get("duration"),
    } for j in items])


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    return JSONResponse(get_job_or_404(job_id))


@app.get("/api/jobs/{job_id}/thumb/{page_index}")
def job_thumb(job_id: str, page_index: int) -> FileResponse:
    job = get_job_or_404(job_id)
    pages = (job.get("result") or {}).get("pages", [])
    if not 1 <= page_index <= len(pages):
        raise HTTPException(404, "找不到頁面")
    thumb = (JOBS / job_id / "pages" / pages[page_index - 1]["thumb"])
    if not thumb.exists():
        raise HTTPException(404, "無縮圖")
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
        raise HTTPException(416, "Range 格式錯誤")
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
def job_video(job_id: str, request: Request):
    job = get_job_or_404(job_id)
    path = Path(job["video_path"])
    if not path.exists():
        raise HTTPException(404, "影片檔不存在")
    return range_response(path, request, "video/mp4")


@app.get("/api/jobs/{job_id}/export.{fmt}")
def job_export(job_id: str, fmt: str) -> FileResponse:
    job = get_job_or_404(job_id)
    if job["status"] != "done":
        raise HTTPException(409, "任務尚未完成")
    mapping = {"srt": "full.srt", "md": "transcript.md"}
    if fmt not in mapping:
        raise HTTPException(400, "僅支援 srt / md")
    path = JOBS / job_id / mapping[fmt]
    if not path.exists():
        raise HTTPException(404, "檔案不存在")
    stem = re.sub(r"[^\w\-]+", "_", Path(job["filename"]).stem)[:40]
    return FileResponse(path, filename=f"{stem}.{fmt}")


app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
