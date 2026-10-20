# CardChop GUI merge — implemented source changes

## Source and target

Read all four requested current files: `CardChop-LIVE-PREVIEW.html`, `CardEnhance-main(2).zip`, `card-enhancer-handoff-main(2).zip`, and `GUI_MERGE_NOTES.md`. Also recovered `CardChop-exact-batch-editor-git.zip` and `express-master(2).zip` to avoid reconstructing the backend from notes. The runnable archive is `cardchop-exact-gui`; GitHub target is `claudepersonal/express`, directory `cardcrop/`.

The supplied images are accessible. `IMG_5361.jpeg` is the dense amber batch layout. `IMG_5362.jpeg` is the Obsidian/cyan/green palette. The two pasted text files are an earlier archive-inspection transcript and malformed Python workspace-repair code, not replacement application modules. That repair script was not executed.

## Implemented mapping

| Canonical source | Resulting target (relative to cardcrop/) |
|---|---|
| Preview HTML and merge notes: 52px header, fixed rail, exact slider defaults, six-column grid, landscape spans, amber selection, compare, exports | `public/index.html`, `public/workbench.css`, `public/batch-workbench.js` |
| Handoff `batch-workbench.tsx`: serial run, stop-after-current, resume, upload selection, audit log | `public/batch-workbench.js`, `public/workbench-state.js` |
| Handoff `batch-store.ts`: per-ID records, custom settings, status, output, timing, error | Existing server jobs registry + persisted `workbench` and `workbenchRevision` in `server.js` |
| Handoff item editor / inspect: normalized crop corners, rotation, overrides, per-card evidence | Edit/inspect dialogs in `public/index.html`; validated API in `workbench-contract.js`; actual worker geometry in `enhance_worker.py` |
| CardEnhance `geometry.ts`, `detect.ts`, `pipeline.ts`, `enhance.ts`: geometry/original-output separation, explicit processing provenance | Manual perspective transform in the existing OpenCV worker; separate original and enhanced URLs, applied-options and sharpness telemetry. Original TypeScript pipeline files remain intact in their archive. They are not falsely presented as imported or running inside Express. |
| Supplied palette image | Optional persisted neon theme; screenshot amber remains default |

## Explicit conflict resolutions

- One Express backend and one plain-JavaScript state model. React/Zustand/TanStack shells are not added to the existing Express app.
- API contrast remains a CLAHE luminance blend in `[0, .5]`; the handoff's contrast multiplier is not passed through. Unsupported restoration-strength, micro-dust, vibrance, anti-glare, xAI grading, WebGL/HF upscale, and JPEG/print-DPI controls are not displayed as functioning controls.
- Geometry is normalized TL/TR/BR/BL in the **rotated** original. Rotation precedes the manual crop. Rotating resets the quad explicitly. Full frame means `[0,0]..[1,1]`, without the handoff's 2% inset.
- Manual quad bypasses detection and auto-orientation so saved geometry is honored. Detector crop remains available as a separate choice.
- Existing OpenCV PNG output and 2200px long-edge cap remain. Resampling is not described as learned super-resolution. Measured Laplacian sharpness delta is processing telemetry, not a grading or authenticity score.
- Resume requires this card's successful output and matching applied options. Duplicate filenames are never resume keys. New edits clear output and reject stale revisions; batch runs own an immutable settings/ID snapshot and cannot overlap.
- Poll failures retain the last successful queue. Upload selection uses returned IDs, not guessed queue positions. Selection updates preserve DOM elements for double-click comparison.
- Fixture mode is read-only and cannot manufacture completed results or downloadable enhanced originals.
- The newer GitHub `only=failed` VLM retry is retained. The previous GitHub GUI is retained as `public/index.original.html` (verified matching blob).

## Verification

- Passed: 9 focused Node tests (`tests/contracts.test.js`, `tests/workbench.test.js`); six existing Python training-tool tests; JavaScript syntax; Python compilation; patch whitespace check.
- Added Python geometry tests for rotation-before-crop, manual-crop provenance, full-frame output, and invalid quads.
- Added `scripts/gui-smoke.cjs` with real server/OpenCV assertions: duplicate filenames, upload, saved edits, stale-revision rejection, actual PNG output, stop/resume, compare, audit settings, persistence, and mobile overflow.
- Local full HTTP/OCR/enhancement and browser execution are **blocked**, not passed: Express/Multer/OpenCV are absent; npm/pip dependency installation is denied/unavailable; Chromium download returns HTTP 403. Existing VLM tests that generate OpenCV fixtures fail at that missing dependency.
- Browser smoke and complete pipeline tests must pass in CI or a provisioned runtime before treating the result as release-verified.

## Available integrations / missing pieces

- GitHub source is readable and writable. The implementation is proposed on a draft branch, not merged or deployed.
- Hugging Face identity is authenticated; the donor's `hlky/RealESRGAN_x2plus` model repository is confirmed. This does not establish a callable inference service. The Space search operation is unavailable on the current connector, and no target Space was supplied. No Space was guessed or changed.
- NVIDIA catalog was checked. The current source and runtime do not establish CUDA/TensorRT/GPU execution. No NVIDIA skill was installed and no GPU performance claim is made.
- Context7 Express documentation was consulted for static-file and same-origin API integration.
- `goofy-main(1).zip` and the commerce archive referenced by older notes were not required to implement the four-file GUI merge and were not read this turn. Their additional behaviors and eBay publication are not claimed as merged. No source application for the pasted ELAN/Gigapixel repair script is established by these four files.
- The `models/card_detector.pt` weights are absent in the recovered exact-editor project. The backend reports whole-frame fallback. CardEnhance's ONNX assets are a different contract and were not renamed to pretend they are PyTorch detector weights.

## Run

From the standalone source folder: install Node dependencies with `npm install`; install Python `opencv-python-headless numpy Pillow pytesseract` and the system Tesseract binary; then `PORT=8080 npm start`. Use `npm test` and, after `npx playwright install chromium`, `npm run test:gui`. These are prerequisites, not claims of successful installation in this session.
