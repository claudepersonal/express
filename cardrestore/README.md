# Card Restoration Lab

Batch Real-ESRGAN restoration for wrestling card scans. Replaces the
`Rbeachg93/card-restoration-batch-demo` Gradio Space.

- **client/**: Vite (vanilla JS). File tray with per-file checks (type, size, pixel limit) before upload, settings, live per-file progress with time estimates, before/after comparison (drag, tap or arrow keys), per-card PNG download, OCR text with copy, manifest table and ZIP. A batch survives reloads (`#job=<id>` in the URL).
- **server/**: Express 5 API. Queues jobs, runs one Python worker at a time, serves the Vite build from `dist/`.
- **worker/restore_worker.py**: Real-ESRGAN x4plus via spandrel (checksum-verified weights), EXIF orientation applied, tiled 4x inference with 64 px context padding, Lanczos resize for 2x, optional `detailEnhance` ("Standard"), Tesseract OCR, `manifest.csv` + `restored_cards.zip`.
- **sea/** + **server/sea.js**: Open Sea, a WebGPU/three.js ocean demo at `/sea/`. three.js is served from the pinned npm package (no CDN) under a hash-pinned CSP; only an allowlist of five module files is reachable.

Originals are never modified. Inputs over `MAX_INPUT_PIXELS` (default 1600×1600 worth of pixels) are shrunk to fit with Lanczos before restoring (`fit=true`, the default; the manifest's `fitted_from` column records the original size) or rejected with `fit=false`. CPU inference takes roughly 1–3 minutes per card at the limit.

## API

| Method | Path | |
|---|---|---|
| POST | `/api/jobs` | multipart: `files` (≤12 images, ≤15 MB each), `profile` (`conservative`\|`standard`), `scale` (2\|4), `tile` (64–512), `ocr` (`true`\|`false`), `fit` (`true`\|`false`). Returns 202 + job. |
| GET | `/api/jobs/:id` | status (`queued`/`running`/`done`/`failed`), rows, per-file errors |
| GET | `/api/jobs/:id/original/:i`, `/output/:i`, `/zip` | files |
| GET | `/health`, `/api/health` | liveness; model status + limits (`maxFiles`, `maxFileMB`, `maxInputPixels`) |
| GET | `/sea/` | Open Sea WebGPU demo (`/sea` redirects) |

Results are kept for 24 hours (`JOB_TTL_HOURS`).

## Local

```sh
npm install
curl -L -o models/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
pip install torch spandrel==0.4.2 opencv-python-headless numpy Pillow pytesseract
npm run dev        # Vite on :5173, API on :8080
npm test           # HTTP tests (fake worker) incl. /sea routing, CSP and allowlist
python3 -m unittest tests/test_worker.py   # tiled == single-pass check (needs weights)
```

## Deploy

Railway service `card-restoration-lab` in project `cardcrop-ai-suite`, root directory `cardrestore`, Dockerfile build.
