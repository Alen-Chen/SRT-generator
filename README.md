# 課程影片．逐頁文字稿（srt_gen）

上傳課程影片，自動產出「逐頁」文字稿：以場景切換偵測將影片切成一頁一頁（投影片），
每頁附縮圖、時間範圍、LLM 生成的話題標題與逐段語音文字，可點時間戳跳轉播放，
並可下載 SRT / Markdown。

## 原理

1. `ffmpeg`（imageio-ffmpeg 內建版）以 1fps 取樣灰階影格，用相鄰影格平均絕對差偵測簡報頁面切換（預設門檻 0.03，UI 可调）
2. `ffmpeg` 擷取 16kHz mono 音訊
3. `faster-whisper`（本地 GPU/CPU 自動切換，GPU 不足時自動 fallback CPU int8）轉錄語音
4. 每頁段落 → 呼叫 OpenAI-compatible `/v1/chat/completions` 生成頁面標題與課程總標題
5. 輸出：逐頁文字稿、頁面縮圖、`full.srt`、`transcript.md`

結果頁的「重新分頁」可沿用已完成的轉錄結果（`reuse_from` 參數），只重跑頁面偵測，數分鐘內完成。

語音轉錄一律在本地進行（API server 若提供語音模型可自行擴充）；
LLM 僅用於產生標題/摘要。

## 安裝與執行

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8080
```

開啟 <http://127.0.0.1:8080> ，拖曳影片上傳即可。

## 環境變數

| 變數 | 預設 | 說明 |
|---|---|---|
| `OPENAI_BASE_URL` | `http://127.0.0.1:8000/v1` | OpenAI-compatible 端點 |
| `OPENAI_API_KEY` | `sk-strata-...` | API key |
| `LLM_MODEL` | 自動取 `/v1/models` 第一個 | 標題生成模型 |
| `WHISPER_MODEL` | `small` | 預設轉錄模型 |
| `WHISPER_DEVICE` / `WHISPER_COMPUTE` | auto / int8_float16→cpu int8 | 強制轉錄裝置 |
| `SG_DATA_DIR` | `./data` | 上傳與結果目錄 |

## API

- `POST /api/jobs`（multipart: `file`, `model`, `language`, `scene_threshold`(預設 0.03), 可選 `reuse_from`=既有 job id 以沿用轉錄結果重新分頁）
- `GET /api/jobs` / `GET /api/jobs/{id}` 任務清單與狀態
- `GET /api/jobs/{id}/video`（支援 Range 串流）、`/thumb/{n}`
- `GET /api/jobs/{id}/export.srt`、`/export.md`

## 測試影片

`test_pattern/` 放置測試影片（體積過大未納入 git，自行放入 2.5 小時課程影片約 30 分鐘完成處理）。
