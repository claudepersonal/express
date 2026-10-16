# Card cleanup pipeline

The existing CardCrop GUI and `POST /api/cards/:id/enhance` now share the
same worker with the batch CLI. No Streamlit service is required.

## GUI

In the controls, turn **Conservative processing** off, then select
**Full classical cleanup** or **CNN / classical fallback**. Select
**Contour + perspective** to rectify a card outline, **YOLO bounding box**
to use the existing detector, or **Whole image** to disable cropping.
Use **Enhance selected** to process the selection. Audit shows the actual
crop method, cleanup engine, warnings, preview, and enhanced PNG link.

The classical profile performs the uploaded sequence: blackhat mark mask,
Telea inpaint blend, light colored denoise, unsharp mask, full LAB-L CLAHE
(`clipLimit=1.5`, grid 8×8), and uniform brightness gain/offset. Blackhat
detects dark narrow marks, not bright glare. The final gain is brightness
adjustment, not white-balance correction. Surface marks and print can change.
The profile scratch-repair slider controls only the inpaint blend. Other
profile stages are fixed; manual descratch/denoise/sharpen/contrast sliders
are disabled for profiles. Scale and auto-rotation still apply.

Manual mode retains the existing slider pipeline. Any positive contrast
value enables the full CLAHE operation; zero bypasses it. Conservative mode
preserves its previous restrictions and rejects full cleanup profiles.
Original uploads are never overwritten. Enhanced PNG writes are atomic.

## API

Example query:

```text
?cleanup=classical&cleanupStrength=0.45&detection=contour&conservative=false&scale=2
```

`cleanup`: `manual` (default), `classical`, `cnn`.
`detection`: `yolo` (default), `contour`, `none`.
`cleanupStrength`: 0–1, default 0.45.
All existing numeric and boolean options remain validated. Outputs are capped
at a 2200px long edge. Contour detection accepts convex quadrilaterals above
the confidence threshold; an absent contour is reported as whole-frame output.
Audit metadata includes requested settings, actual engine/device, fallback
warnings, detection mode, inpaint pixel count, and CLAHE configuration.

## Optional CNN

Set `CHROME_CNN_WEIGHTS` on the server to a trained `ChromeArtifactCNN`
state-dictionary file. `CHROME_CNN_DEVICE=auto` uses CUDA when Torch reports
it available, otherwise CPU; `cpu` and `cuda` are also accepted. The existing
Docker image installs CPU Torch. A GPU deployment needs a CUDA-capable Torch
installation and GPU runtime. Selecting CNN does not provision hardware.

Inference uses the shared `chrome_model.py` architecture, strict checkpoint
loading with `weights_only=True`, and 256px tiles with a 40px halo to bound
memory. Missing or incompatible weights and failed inference fall back to
classical cleanup with an explicit audit warning. No trained weights are
bundled, and no quality claim is made for an untrained model.

Train on real clean card images with synthetic corruptions:

```sh
python train_chrome_cnn.py --clean_dir ./clean_cards --output_dir ./checkpoints --epochs 20 --batch-size 4 --synthetic
```

The trainer shares the inference architecture. It rejects missing/empty image
datasets instead of manufacturing random training images. Training is an
explicit command; it never runs during a web request.

## Batch PNG and ZIP export

```sh
python batch_pipeline.py --input_dir ./scans --output_dir ./enhanced --cleanup classical --detection contour --scale 1 --zip
```

The CLI uses the same worker, preserves source files, writes collision-safe
PNG names, and records source hashes, failures, and actual processing metadata
in `manifest.json`. ZIP output includes successful images and the manifest.
A partial failure exits nonzero and does not stop other images from processing.

## Checks

```sh
npm test
python -m unittest discover -s tests -p 'test_cnn.py' -v
```

The second command requires Torch; the CI CNN job installs CPU Torch. Tests
cover API validation, crop geometry, classical output, CNN fallback, tiled
inference, checkpoint compatibility, original preservation and batch export.
