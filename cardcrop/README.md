# CardChop — Vite + FastAPI

The dense wrestling-card Batch Editor, built by Vite and served with its FastAPI backend.

## Production

The Docker build installs frontend dependencies, runs Vite, installs Python dependencies,
and tests real upload/edit/OpenCV enhancement/registry recovery before starting FastAPI.
`PORT=8080 npm start` starts the production API and built frontend. Persistent originals,
PNGs and `jobs.json` live in `STORAGE_DIR` (Railway: `/data`). Use one Uvicorn worker:
the durable registry has one process owner.

## Development

1. `python3 -m pip install -r requirements.txt` and install the Tesseract system binary.
2. `npm install` and `npm --prefix frontend install`.
3. `npm run dev`: Vite on 5173 proxies API/images to FastAPI on 8000.
4. `npm run build` produces `frontend/dist`; `npm start` serves it with FastAPI.

## Verification

`npm test` runs state/VLM tests and Python geometry/API tests. `npm run test:gui`
uses Chromium against the real FastAPI server and OpenCV worker after a Vite build.
Install the browser with `npx playwright install chromium` before that check.

The UI keeps the screenshot layout, per-card crop/rotation/settings, stop/resume,
original/enhanced comparison, actual PNG exports, and applied-settings audits.
The original screenshot amber remains default; Workbench offers the supplied neon palette.
No eBay publication occurs here. The VLM bridge reuses the existing wrestling-only provider
module and server credentials; credentials never enter the Vite bundle.

The production build fetches the existing pinned YOLO card detector and verifies its SHA-256;
missing/unloadable runtime weights are reported as whole-frame fallback. OpenCV
resampling is reported as OpenCV, never as learned super-resolution or NVIDIA inference.
The older Express source is retained as migration reference; it is not the start command.
