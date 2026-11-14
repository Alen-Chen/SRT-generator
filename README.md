# Course Video · Page Transcripts (srt_gen)

Upload a course video and get a **page-by-page transcript**: slide-change detection
splits the video into pages (slides), and every page gets a thumbnail, a time range,
an LLM-generated topic title, and timestamped speech segments. Timestamps are
clickable to seek the video; SRT / Markdown exports are available.

## How it works

1. `ffmpeg` (bundled via imageio-ffmpeg) samples frames at 1fps; slide changes are
   detected from the mean absolute difference between consecutive grayscale frames
   (default threshold 0.03, adjustable in the UI)
2. Audio is extracted as 16 kHz mono
3. `faster-whisper` transcribes the speech locally (GPU/CPU auto-selected; falls
   back to CPU int8 when the GPU is full)
4. (Optional, on by default) Two-round LLM transcript correction:
   - Round 1: ASR output is kept as-is as a draft transcript
   - The draft is sent to the LLM to write a correction guide (lecture summary,
     domain background knowledge, inferred glossary and expected ASR error
     patterns)
   - Round 2: the whole draft (numbered lines in batches) plus the guide are
     sent back to the LLM to produce the corrected transcript; the draft is
     overwritten, only the corrected version is kept and offered for download
5. Each page's text is sent to an OpenAI-compatible `/v1/chat/completions` endpoint
   to generate per-page topic titles, an overall course title, and a bilingual
   (Chinese/English) title + summary shown in the right-hand history panel
   (summaries are collapsed; click to expand; older jobs backfill on first view)
6. (Optional, off by default) Smart silence-cut: using the transcript as the
   basis, stretches with no speech for 6 s or more (pre-start preparation,
   dead air between merged parts) are cut out of the video with a precise
   re-encode; segment/page timestamps and page thumbnails are shifted to the
   new timeline, and the result (`{id}_clip.mp4`) becomes the playable/
   downloadable video. Removed blocks and seconds are reported on the result
   page.
7. Outputs: page-by-page transcript, page thumbnails, `full.srt`, `transcript.md`

Speech transcription always runs locally; the LLM API is used for transcript
proofreading and titles.
The "Repaginate" control on the result page reuses an existing transcript
(`reuse_from` parameter) and re-runs only page detection — finishes in minutes.

## Install & run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp config.example.json config.json   # then edit api_key
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8080
```

Open <http://127.0.0.1:8080> and drag a video in.

Multiple part files of one course session can be selected/dropped together:
they are merged into a single video first (ffmpeg concat, stream copy with
re-encode fallback), then processed as one; the merged video is downloadable
from the result page.

## UI languages

English, 繁體中文, 简体中文 and 日本語. The default language is auto-detected
from the browser settings (zh-TW/HK/MO/Hant → Traditional, other zh →
Simplified, ja → Japanese, otherwise English) and can be overridden with the
selector in the header (persisted in localStorage).

## Configuration

Copy `config.example.json` to `config.json` and fill in your API credentials.
`config.json` is gitignored and holds the secret API key; never commit it.

```json
{
  "base_url": "http://127.0.0.1:8000/v1",
  "api_key": "<your-api-key>",
  "llm_model": ""
}
```

If the API key is missing, the app still works but titles fall back to plain text.

Environment variables override the config file:

| Variable | Default | Description |
|---|---|---|
| `SG_CONFIG` | `./config.json` | config file path |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `LLM_MODEL` | from config file | OpenAI-compatible endpoint / key / title model (default: first id from `/v1/models`) |
| `WHISPER_MODEL` | `small` | default transcription model |
| `WHISPER_DEVICE` / `WHISPER_COMPUTE` | auto / int8_float16→cpu int8 | force transcription device |
| `SG_DATA_DIR` | `./data` | uploads and results directory |

## Accounts & roles

- Every page requires sign-in with a registered username/password. Sessions last
  7 days (HttpOnly cookie); users are stored in `data/users.json`.
- The **first registered account becomes the administrator** and is signed in
  immediately. **Every later registration starts as `pending`** and cannot sign
  in until an administrator approves it (Users panel → Approve / Enable).
  Administrators can also disable accounts and promote/demote roles
  (self-deactivation and self-demotion are blocked).
- Each user only sees their own job history, videos and exports; administrators
  see everything and can open any job. Any listed record can be deleted (History
  panel → Delete): the job entry, its uploaded/merged videos and all generated
  subtitle/transcript files are removed permanently. Jobs still queued or
  running cannot be deleted. Videos shared via re-use stay with their source job.
- Uploaded originals are kept per user (`data/uploads/`); when several parts
  are merged, only the merged file is kept. Generated subtitles
  (`full.srt`) and the Markdown transcript are stored under `data/jobs/{id}/`.

## API

All `/api/jobs*` endpoints require a signed-in session cookie (401 otherwise).

- `POST /api/auth/register` / `POST /api/auth/login` (JSON `{username,password}`),
  `POST /api/auth/logout`, `GET /api/auth/me`
- `GET /api/admin/users`, `POST /api/admin/users/{id}/role`,
  `POST /api/admin/users/{id}/status` (admin only)
- `POST /api/jobs/{id}/meta` — generate bilingual (Chinese/English) title and
  summary from the transcript (used by the history panel; cached in the job)
- `POST /api/jobs` (multipart: `files` (repeatable), `model`, `language`,
  `scene_threshold`
  (default 0.03); optional `reuse_from`=<job id> to reuse a transcript and only
  re-split pages)
- `GET /api/jobs` / `GET /api/jobs/{id}` — job list and status
- `GET /api/jobs/{id}/video` (HTTP Range streaming), `/thumb/{n}`
- `GET /api/jobs/{id}/export.srt`, `/export.md`

## Test video

Place test videos in `test_pattern/` (excluded from git due to size; a 2.5-hour
course video takes roughly 30 minutes to process).
