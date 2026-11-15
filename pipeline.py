"""Video -> page-by-page transcript pipeline (local ffmpeg + faster-whisper)."""
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
        raise ValueError("Cannot parse video duration; not a valid video file?")
    h, mnt, sec = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(sec)


def extract_audio(video: Path, wav: Path) -> None:
    r = _run(["-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
              "-c:a", "pcm_s16le", str(wav)])
    if r.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {r.stderr[-800:]}")


def probe_streams(video: Path) -> dict:
    r = _run(["-i", str(video)], timeout=120)
    out = {"video": False, "audio": False}
    for line in r.stderr.splitlines():
        m = re.search(r"Stream #\d+:\d+.*?: (Video|Audio)", line)
        if m:
            out[m.group(1).lower()] = True
    return out


def merge_videos(parts: list[Path], out: Path) -> None:
    """Concatenate part files into one video.

    Fast path: container concat demuxer with stream copy. If codecs are
    incompatible, fall back to re-encoding with the concat filter."""
    if len(parts) < 2:
        raise ValueError("Merging requires at least two files")
    for p in parts:
        if not p.exists():
            raise FileNotFoundError(f"Missing part file: {p}")
    lst = out.with_name(out.name + ".concat")
    lst.write_text("".join(f"file '{p}'\n" for p in parts), encoding="utf-8")
    try:
        r = _run(["-f", "concat", "-safe", "0", "-i", str(lst),
                  "-c", "copy", str(out)])
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return
        log.warning("Concat copy failed, falling back to re-encode: %s",
                    r.stderr[-400:])
    finally:
        lst.unlink(missing_ok=True)

    info = [probe_streams(p) for p in parts]
    if not all(i["video"] for i in info):
        raise RuntimeError("Merge failed: some parts contain no video stream")
    pre, labels = [], []
    for i, inf in enumerate(info):
        if inf["audio"]:
            alab = f"[{i}:a]"
        else:
            alab = f"[sil{i}]"
            pre.append(f"anullsrc=channel_layout=stereo:sample_rate=48000[sil{i}]")
        labels.append(f"[{i}:v]{alab}")
    chain = ";".join(pre + ["".join(labels) +
                            f"concat=n={len(parts)}:v=1:a=1[v][a]"])
    args: list[str] = []
    for p in parts:
        args += ["-i", str(p)]
    args += ["-filter_complex", chain, "-map", "[v]", "-map", "[a]",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-c:a", "aac", str(out)]
    r2 = _run(args, timeout=None)
    if r2.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"Merge failed: {(r2.stderr or r.stderr)[-800:]}")


CUT_MIN_GAP = 6.0
CUT_PAD = 0.4


def compute_keep_blocks(segments: list[dict], duration: float,
                        min_gap: float = CUT_MIN_GAP,
                        pad: float = CUT_PAD) -> list[list[float]]:
    """Merge transcript speech segments into keep blocks (padded by `pad`).
    Speech separated by less than `min_gap` of silence is kept contiguously;
    gaps of `min_gap` or more (plus leading/trailing dead air) become cuts."""
    blocks: list[list[float]] = []
    for s in sorted(segments, key=lambda x: x["start"]):
        b0 = max(0.0, float(s["start"]) - pad)
        b1 = min(duration, float(s["end"]) + pad)
        if blocks and (b0 <= blocks[-1][1] or b0 - blocks[-1][1] < min_gap):
            blocks[-1][1] = max(blocks[-1][1], b1)
        else:
            blocks.append([b0, b1])
    return blocks


def compute_drop_blocks(keep: list[list[float]], duration: float,
                        min_len: float = 0.05) -> list[tuple[float, float]]:
    drops: list[tuple[float, float]] = []
    prev = 0.0
    for b in keep:
        if b[0] - prev > min_len:
            drops.append((prev, b[0]))
        prev = max(prev, b[1])
    if duration - prev > min_len:
        drops.append((prev, duration))
    return drops


def shift_time(t: float, drops: list[tuple[float, float]]) -> float:
    """Map an original-timeline time onto the cut timeline. Times that fall
    inside a removed block clamp to that block's start position."""
    acc = 0.0
    for ds, de in drops:
        if de <= t:
            acc += de - ds
        elif t >= ds:
            return ds - acc
        else:
            break
    return t - acc


def cut_video(video: Path, out: Path, keep: list[list[float]]) -> None:
    """Re-encode `video` keeping only the given time ranges (precise cuts)."""
    if not keep:
        raise ValueError("Nothing to keep")
    streams = probe_streams(video)
    has_a = bool(streams["audio"])
    parts = []
    for i, (a, b) in enumerate(keep):
        parts.append(f"[0:v]trim={a:.3f}:{b:.3f},setpts=PTS-STARTPTS[v{i}]")
        if has_a:
            parts.append(f"[0:a]atrim={a:.3f}:{b:.3f},asetpts=PTS-STARTPTS[a{i}]")
    labels = [f"[v{i}]" + (f"[a{i}]" if has_a else "") for i in range(len(keep))]
    parts.append("".join(labels) +
                 f"concat=n={len(keep)}:v=1:a={1 if has_a else 0}[outv]" +
                 ("[outa]" if has_a else ""))
    args: list[str] = ["-i", str(video), "-filter_complex", ";".join(parts),
                       "-map", "[outv]"]
    if has_a:
        args += ["-map", "[outa]", "-c:a", "aac"]
    args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-movflags", "+faststart", str(out)]
    r = _run(args, timeout=None)
    if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"Smart cut failed: {(r.stderr or '')[-800:]}")


def _scaled_dims(video: Path) -> tuple[int, int]:
    """Return grayscale frame size after scale=320:-2."""
    r = subprocess.run(
        [FFMPEG, "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
         "-i", str(video), "-vf", "fps=1,scale=320:-2,format=gray",
         "-frames:v", "1", "-f", "rawvideo", "-"],
        capture_output=True, timeout=120)
    n = len(r.stdout)
    if n < 1024:
        raise RuntimeError(f"Cannot probe frame size: {r.stderr.decode()[-400:]}")
    return 320, n // (320 * 2) * 2


def detect_scenes(video: Path, threshold: float,
                  on_progress: Optional[Callable[[float], None]] = None) -> list[float]:
    """Sample frames at 1fps and detect slide changes via mean absolute frame
    difference (normalized 0-1) between consecutive samples."""
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


def load_whisper(size: str, force_kwargs: Optional[dict] = None):
    with _model_lock:
        if force_kwargs is not None:
            _model_cache.pop(size, None)
            import gc
            gc.collect()
            attempts = [force_kwargs]
        elif size in _model_cache:
            return _model_cache[size]
        from faster_whisper import WhisperModel
        forced_device = os.environ.get("WHISPER_DEVICE")
        forced_compute = os.environ.get("WHISPER_COMPUTE")
        if force_kwargs is not None:
            pass
        elif forced_device:
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
        raise RuntimeError(f"Failed to load Whisper model {size}: {last_err}")


def load_pcm16_mono(wav: Path) -> "np.ndarray":
    """Decode a 16-bit mono PCM WAV with the stdlib wave module to avoid
    PyAV version-compatibility issues."""
    import numpy as np
    import wave

    with wave.open(str(wav), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("Audio must be 16-bit mono PCM")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _is_gpu_alloc_error(e: Exception) -> bool:
    msg = str(e).lower()
    return (("cublas" in msg or "cudnn" in msg or "cuda" in msg
             or "out of memory" in msg)
            and ("alloc" in msg or "failed with status" in msg
                 or "out of memory" in msg))


_SR = 16000  # the WAV we extract is always 16 kHz mono PCM


def transcribe(wav: Path, model_size: str, language: str, duration: float,
               on_frac: Optional[Callable[[float], None]] = None,
               checkpoint: Optional[Path] = None):
    """Transcribe `wav`.

    When `checkpoint` is a file containing a partial transcript (written by a
    run that was interrupted), transcription resumes from the last segment
    instead of starting over, so a long lecture survives a server restart.
    """
    model = load_whisper(model_size)
    audio = load_pcm16_mono(wav)

    done: list[dict] = []
    offset = 0.0
    if checkpoint and checkpoint.exists():
        try:
            data = json.loads(checkpoint.read_text(encoding="utf-8"))
            done = [s for s in (data.get("segments") or [])
                    if isinstance(s, dict) and s.get("text")]
            if done:
                offset = max(float(s.get("end", 0.0)) for s in done)
                log.info("Resuming transcription of %s from %.1fs "
                         "(%d segments saved)", wav, offset, len(done))
        except Exception as e:  # noqa: BLE001
            log.warning("Unreadable checkpoint %s: %s", checkpoint, e)
            done, offset = [], 0.0
    start = min(int(offset * _SR), len(audio))

    def _save(seg_list: list[dict], lang: str, partial: bool) -> None:
        if not checkpoint:
            return
        payload = {"language": lang, "segments": seg_list}
        if partial:
            payload["partial"] = True
        checkpoint.write_text(json.dumps(payload, ensure_ascii=False),
                             encoding="utf-8")

    def _run(m):
        seg_iter, info = m.transcribe(
            audio[start:],
            language=None if (not language or language == "auto") else language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
        )
        segments: list[dict] = list(done)
        saved = len(done)
        for s in seg_iter:
            text = s.text.strip()
            if not text:
                continue
            segments.append({"start": round(s.start + offset, 2),
                             "end": round(s.end + offset, 2), "text": text})
            if on_frac:
                on_frac((s.end + offset) / max(duration, 1.0))
            if checkpoint and len(segments) - saved >= 200:
                saved = len(segments)
                _save(segments, info.language, True)
        _save(segments, info.language, False)
        return segments, info.language

    try:
        return _run(model)
    except Exception as e:  # noqa: BLE001
        if not _is_gpu_alloc_error(e):
            raise
        log.warning("Whisper GPU inference failed (%s); retrying on CPU", e)
        model = load_whisper(model_size,
                             force_kwargs={"device": "cpu",
                                           "compute_type": "int8"})
        return _run(model)


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

    p("Reading video info", 0.01)
    duration = get_duration(video)

    segfile = outdir / "segments.json"
    partial = False
    complete = False
    if segfile.exists():
        try:
            saved = json.loads(segfile.read_text(encoding="utf-8"))
            saved_segs = saved if isinstance(saved, list) else (saved.get("segments") or [])
            partial = bool(saved_segs) and isinstance(saved, dict) and bool(saved.get("partial"))
            complete = bool(saved_segs) and not partial
        except Exception:  # noqa: BLE001
            pass
    # A finished transcript in the job folder is never thrown away: a resumed
    # job therefore skips transcription entirely if it had already completed.
    reused = reuse_segments or complete

    p("Detecting slide changes", 0.05)
    scenes = detect_scenes(
        video, scene_threshold,
        lambda t: p("Detecting slide changes",
                    0.05 + 0.15 * min(t / max(duration, 1), 1.0)))

    if reused:
        data = json.loads(segfile.read_text(encoding="utf-8"))
        if isinstance(data, list):
            segments, lang = data, "unknown"
        else:
            segments, lang = data["segments"], data.get("language") or "unknown"
        p("Reusing existing transcript", 0.84)
    else:
        wav = outdir / "audio.wav"
        if not partial:
            segfile.unlink(missing_ok=True)
            if wav.exists() and wav.stat().st_size > 1000:
                log.info("Reusing already extracted audio %s", wav)
            else:
                p("Extracting audio", 0.20)
                extract_audio(video, wav)
        elif not wav.exists():
            raise RuntimeError("Cannot resume: the extracted audio is missing")

        p("Transcribing speech", 0.22)
        segments, lang = transcribe(
            wav, model_size, language, duration,
            lambda f: p("Transcribing speech", 0.22 + 0.60 * min(f, 1.0)),
            checkpoint=segfile)

    p("Splitting pages", 0.84)
    pages = build_pages(segments, scenes, duration)

    p("Generating page thumbnails", 0.85)
    pages_dir = outdir / "pages"
    pages_dir.mkdir(exist_ok=True)
    for i, page in enumerate(pages, 1):
        span = page["end"] - page["start"]
        t = page["start"] + min(1.0, span / 2)
        t = max(page["start"], min(t, min(page["end"] - 0.1, duration - 0.1)))
        thumb = pages_dir / f"{i:03d}.jpg"
        page["index"] = i
        page["thumb"] = thumb.name if make_thumbnail(video, t, thumb) else None
    p("Generating page thumbnails", 0.95)

    return {"duration": round(duration, 2), "language": lang,
            "segments": segments, "pages": pages}
