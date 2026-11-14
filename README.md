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
4. Each page's text is sent to an OpenAI-compatible `/v1/chat/completions` endpoint
   to generate per-page topic titles and an overall course title
5. Outputs: page-by-page transcript, page thumbnails, `full.srt`, `transcript.md`

Speech transcription always runs locally; the LLM API is only used for titles.
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

## API

- `POST /api/jobs` (multipart: `file`, `model`, `language`, `scene_threshold`
  (default 0.03); optional `reuse_from`=<job id> to reuse a transcript and only
  re-split pages)
- `GET /api/jobs` / `GET /api/jobs/{id}` — job list and status
- `GET /api/jobs/{id}/video` (HTTP Range streaming), `/thumb/{n}`
- `GET /api/jobs/{id}/export.srt`, `/export.md`

## Test video

Place test videos in `test_pattern/` (excluded from git due to size; a 2.5-hour
course video takes roughly 30 minutes to process).
