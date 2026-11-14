"""影片 -> 逐頁文字稿 pipeline（本地 ffmpeg + faster-whisper）。"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
from bisect import bisect_right
from pathlib import Path
from typing import Callable, Optional

import imageio_ffmpeg

log = logging.getLogger("srt_gen.pipeline")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

_model_cache: dict = {}
_model_lock = threading.Lock()

ProgressFn = Callable[[str, float], None]


def _run(args: list[str], timeout: Optional[float] = 7200) -> subprocess.CompletedProcess:
    return subprocess.run(
        [FFMPEG, "-hide_banner", "-nostdin", "-y", *args],
        capture_output=True, text=True, timeout=timeout,
    )


def get_duration(video: Path) -> float:
    r = _run(["-i", str(video)], timeout=120)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr)
    if not m:
        raise ValueError("無法解析影片長度，可能不是有效的影片檔")
    h, mnt, sec = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(sec)


def extract_audio(video: Path, wav: Path) -> None:
    r = _run(["-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
              "-c:a", "pcm_s16le", str(wav)])
    if r.returncode != 0:
        raise RuntimeError(f"音訊擷取失敗: {r.stderr[-800:]}")


def _scaled_dims(video: Path) -> tuple[int, int]:
    """取得 320:-2 縮放後的灰階影格寬高。"""
    r = subprocess.run(
        [FFMPEG, "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
         "-i", str(video), "-vf", "fps=1,scale=320:-2,format=gray",
         "-frames:v", "1", "-f", "rawvideo", "-"],
        capture_output=True, timeout=120)
    n = len(r.stdout)
    if n < 1024:
        raise RuntimeError(f"無法解析影格尺寸: {r.stderr.decode()[-400:]}")
    return 320, n // (320 * 2) * 2


def detect_scenes(video: Path, threshold: float,
                  on_progress: Optional[Callable[[float], None]] = None) -> list[float]:
    """1fps 取樣灰階影格，以相鄰影格平均絕對差偵測簡報頁面切換（0–1 正規化）。"""
    import numpy as np

    w, h = _scaled_dims(video)
    frame_bytes = w * h
    proc = subprocess.Popen(
        [FFMPEG, "-hide_banner", "-nostdin", "-y", "-i", str(video),
         "-vf", "fps=1,scale=320:-2,format=gray", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None

    times: list[float] = []
    prev: Optional[np.ndarray] = None
    prev_idx = -1
    chunk_frames = 400
    while True:
        buf = proc.stdout.read(frame_bytes * chunk_frames)
        if not buf:
            break
        flat = np.frombuffer(buf, dtype=np.uint8)
        count = flat.size // frame_bytes
        if count == 0:
            break
        frames = flat[: count * frame_bytes].reshape(count, frame_bytes)
        if prev is None:
            prev = frames[0].astype(np.float32)
            prev_idx = 0
            frames = frames[1:]
        if frames.shape[0]:
            stack = np.concatenate([prev.reshape(1, frame_bytes), frames], axis=0)
            diffs = np.abs(np.diff(stack, axis=0)).mean(axis=1) / 255.0
            for j in np.nonzero(diffs >= threshold)[0]:
                times.append(float(prev_idx + int(j) + 1))
            prev = frames[-1].astype(np.float32)
            prev_idx += int(frames.shape[0])
        if on_progress:
            on_progress(float(prev_idx + 1))
    proc.wait()
    return sorted(set(times))


def load_whisper(size: str):
    with _model_lock:
        if size in _model_cache:
            return _model_cache[size]
        from faster_whisper import WhisperModel
        forced_device = os.environ.get("WHISPER_DEVICE")
        forced_compute = os.environ.get("WHISPER_COMPUTE")
        if forced_device:
            attempts = [{"device": forced_device,
                         "compute_type": forced_compute or "int8"}]
        else:
            attempts = [{"device": "auto",
                         "compute_type": forced_compute or "int8_float16"},
                        {"device": "cpu", "compute_type": "int8"}]
        last_err: Optional[Exception] = None
        for kwargs in attempts:
            try:
                model = WhisperModel(size, **kwargs)
                log.info("Whisper model %s loaded (%s)", size, kwargs)
                _model_cache[size] = model
                return model
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning("Whisper load failed %s: %s", kwargs, e)
        raise RuntimeError(f"無法載入 Whisper 模型 {size}: {last_err}")


def load_pcm16_mono(wav: Path) -> "np.ndarray":
    """以標準函式庫解碼 16-bit mono PCM WAV，避免 PyAV 版本相容問題。"""
    import numpy as np
    import wave

    with wave.open(str(wav), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("音訊需為 16-bit mono PCM")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(wav: Path, model_size: str, language: str, duration: float,
               on_frac: Optional[Callable[[float], None]] = None):
    model = load_whisper(model_size)
    audio = load_pcm16_mono(wav)
    seg_iter, info = model.transcribe(
        audio,
        language=None if (not language or language == "auto") else language,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
    )
    segments: list[dict] = []
    for s in seg_iter:
        text = s.text.strip()
        if not text:
            continue
        segments.append({"start": round(s.start, 2),
                         "end": round(s.end, 2), "text": text})
        if on_frac:
            on_frac(s.end / max(duration, 1.0))
    return segments, info.language


def build_pages(segments: list[dict], scene_times: list[float], duration: float,
                min_page: float = 6.0) -> list[dict]:
    bounds: list[float] = []
    last = 0.0
    for t in scene_times:
        if t <= 1.5 or duration - t <= 2:
            continue
        if t - last >= min_page:
            bounds.append(t)
            last = t
    edges = [0.0] + bounds + [max(duration, 1.0)]
    pages = [{"start": round(edges[i], 2), "end": round(edges[i + 1], 2),
              "segments": []} for i in range(len(edges) - 1)]
    for seg in segments:
        mid = (seg["start"] + seg["end"]) / 2
        idx = min(max(bisect_right(edges, mid) - 1, 0), len(pages) - 1)
        pages[idx]["segments"].append(seg)
    merged: list[dict] = []
    for p in pages:
        if p["segments"]:
            merged.append(p)
        elif merged:
            merged[-1]["end"] = p["end"]
    return merged


def make_thumbnail(video: Path, t: float, out: Path) -> bool:
    r = _run(["-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
              "-q:v", "3", str(out)], timeout=300)
    return r.returncode == 0 and out.exists() and out.stat().st_size > 0


def to_srt(segments: list[dict]) -> str:
    def ts(sec: float) -> str:
        ms = int(round(sec * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = []
    for i, seg in enumerate(segments, 1):
        blocks.append(f"{i}\n{ts(seg['start'])} --> {ts(seg['end'])}\n{seg['text']}\n")
    return "\n".join(blocks)


def process(video: Path, outdir: Path, model_size: str = "small",
            language: str = "auto", scene_threshold: float = 0.03,
            reuse_segments: bool = False,
            progress: Optional[ProgressFn] = None) -> dict:
    p = progress or (lambda step, pct: None)
    outdir.mkdir(parents=True, exist_ok=True)

    p("讀取影片資訊", 0.01)
    duration = get_duration(video)

    segfile = outdir / "segments.json"
    reused = reuse_segments and segfile.exists()

    p("偵測頁面切換", 0.05)
    scenes = detect_scenes(
        video, scene_threshold,
        lambda t: p("偵測頁面切換", 0.05 + 0.15 * min(t / max(duration, 1), 1.0)))

    if reused:
        data = json.loads(segfile.read_text(encoding="utf-8"))
        if isinstance(data, list):
            segments, lang = data, "unknown"
        else:
            segments, lang = data["segments"], data.get("language") or "unknown"
        p("沿用既有轉錄結果", 0.84)
    else:
        p("擷取音訊", 0.20)
        wav = outdir / "audio.wav"
        extract_audio(video, wav)

        p("語音轉錄", 0.22)
        segments, lang = transcribe(
            wav, model_size, language, duration,
            lambda f: p("語音轉錄", 0.22 + 0.60 * min(f, 1.0)))
        segfile.write_text(
            json.dumps({"language": lang, "segments": segments},
                       ensure_ascii=False, indent=1), encoding="utf-8")

    p("切割頁面", 0.84)
    pages = build_pages(segments, scenes, duration)

    p("產生頁面縮圖", 0.85)
    pages_dir = outdir / "pages"
    pages_dir.mkdir(exist_ok=True)
    for i, page in enumerate(pages, 1):
        span = page["end"] - page["start"]
        t = page["start"] + min(1.0, span / 2)
        t = max(page["start"], min(t, min(page["end"] - 0.1, duration - 0.1)))
        thumb = pages_dir / f"{i:03d}.jpg"
        page["index"] = i
        page["thumb"] = thumb.name if make_thumbnail(video, t, thumb) else None
    p("產生頁面縮圖", 0.95)

    return {"duration": round(duration, 2), "language": lang,
            "segments": segments, "pages": pages}
