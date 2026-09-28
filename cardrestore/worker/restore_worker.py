#!/usr/bin/env python3
"""Card Restoration Lab worker: Real-ESRGAN x4plus batch restore.

Reads one job as JSON on stdin and writes newline-delimited JSON events to
stdout (stdout is JSON-only; library noise goes to stderr):

  {"event": "ready", "device": "cpu"}
  {"event": "file", "index": 0, "row": {...}}           # one per input
  {"event": "error", "index": 0, "file": "...", "message": "..."}
  {"event": "done", "zip": "restored_cards.zip", "manifest": "manifest.csv"}

Originals are never modified. Output PNGs, manifest.csv and the zip are
written to job["outdir"].
"""
import contextlib
import csv
import hashlib
import json
import math
import os
import sys
import time
import zipfile

import cv2
import numpy as np
import torch
from PIL import Image

MODEL_PATH = os.environ.get('RESTORE_MODEL', os.path.join(os.path.dirname(__file__), '..', 'models', 'RealESRGAN_x4plus.pth'))
MODEL_SHA256 = '4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1'
MAX_INPUT_PIXELS = int(os.environ.get('MAX_INPUT_PIXELS', str(1600 * 1600)))
TILE_PAD = int(os.environ.get('TILE_PAD', '64'))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def available_cpus():
    """CPUs this container may actually use. os.cpu_count() reports the host's
    cores (dozens on Railway), and that many torch threads on an 8-vCPU quota
    thrash; honour the cgroup quota and CPU affinity instead."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    try:
        with open('/sys/fs/cgroup/cpu.max') as fh:
            quota, period = fh.read().split()[:2]
        if quota != 'max':
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, n)


THREADS = int(os.environ.get('TORCH_THREADS') or available_cpus())
torch.set_num_threads(THREADS)

_model = None


def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + '\n')
    sys.stdout.flush()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def model():
    """Load RealESRGAN_x4plus (RRDBNet, 4x) once via spandrel after verifying its checksum."""
    global _model
    if _model is None:
        from spandrel import ImageModelDescriptor, ModelLoader
        if not os.path.isfile(MODEL_PATH):
            raise RuntimeError(f'model weights missing at {MODEL_PATH}')
        if os.environ.get('SKIP_MODEL_CHECKSUM') != '1' and sha256_file(MODEL_PATH) != MODEL_SHA256:
            raise RuntimeError('RealESRGAN_x4plus.pth checksum mismatch; refusing to load')
        with contextlib.redirect_stdout(sys.stderr):
            descriptor = ModelLoader().load_from_file(MODEL_PATH)
        if not isinstance(descriptor, ImageModelDescriptor) or descriptor.scale != 4:
            raise RuntimeError('unexpected model file (expected a 4x image model)')
        _model = descriptor.model.eval().to(DEVICE)
    return _model


@torch.inference_mode()
def upscale_x4(image_rgb, tile, pad=TILE_PAD, progress=None):
    """Tiled 4x inference. Each tile is run with `pad` px of real context on every
    side and only its centre is kept, so tile seams are not visible. The output is
    assembled directly as uint8 to keep memory at ~48 bytes per input pixel."""
    net, s = model(), 4
    h, w = image_rgb.shape[:2]
    x = torch.from_numpy(np.array(image_rgb, dtype=np.uint8, copy=True)).permute(2, 0, 1).float().div_(255).unsqueeze(0)
    out = np.empty((h * s, w * s, 3), dtype=np.uint8)
    step = max(int(tile), 32)
    total = math.ceil(h / step) * math.ceil(w / step)
    done = 0
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1, x1 = min(y0 + step, h), min(x0 + step, w)
            py0, py1, px0, px1 = max(y0 - pad, 0), min(y1 + pad, h), max(x0 - pad, 0), min(x1 + pad, w)
            o = net(x[:, :, py0:py1, px0:px1].to(DEVICE)).float().clamp_(0, 1).mul_(255).round_().byte().cpu()
            oy, ox = (y0 - py0) * s, (x0 - px0) * s
            crop = o[0, :, oy:oy + (y1 - y0) * s, ox:ox + (x1 - x0) * s].permute(1, 2, 0).numpy()
            out[y0 * s:y1 * s, x0 * s:x1 * s] = crop
            done += 1
            if progress:
                progress(done, total)
    return out


def edge_metric(rgb):
    """Variance of the Laplacian: a sharpness diagnostic, not a quality score."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)


def ocr_text(rgb):
    try:
        import pytesseract
        with contextlib.redirect_stdout(sys.stderr):
            return pytesseract.image_to_string(Image.fromarray(rgb), config='--psm 3').strip()
    except Exception as exc:  # OCR is a review aid; never fail the restore over it
        return f'OCR unavailable: {exc}'


def restore_one(src, name, outdir, profile, scale, tile, want_ocr, index=0):
    started = time.perf_counter()
    with Image.open(src) as im:
        original = np.asarray(im.convert('RGB'))
    h, w = original.shape[:2]
    if h * w > MAX_INPUT_PIXELS:
        raise ValueError(f'{w}x{h} is larger than the {MAX_INPUT_PIXELS:,}-pixel input limit; '
                         'this tool is for low-resolution scans, resize before restoring')
    restored = upscale_x4(original, tile, progress=lambda d, t: emit({'event': 'progress', 'index': index, 'tile': d, 'tiles': t}))
    if scale == 2:
        restored = np.asarray(Image.fromarray(restored).resize((w * 2, h * 2), Image.Resampling.LANCZOS))
    if profile == 'standard':
        restored = cv2.detailEnhance(np.ascontiguousarray(restored), sigma_s=5, sigma_r=0.08)
    stem = os.path.splitext(os.path.basename(name))[0] or 'card'
    out_name = f'{stem}_restored_{scale}x.png'
    n = 2
    while os.path.exists(os.path.join(outdir, out_name)):  # two inputs with the same name
        out_name = f'{stem}_{n}_restored_{scale}x.png'
        n += 1
    out_path = os.path.join(outdir, out_name)
    Image.fromarray(restored).save(out_path, optimize=True)
    text = ocr_text(restored) if want_ocr else None
    return {
        'file': name,
        'output': out_name,
        'input_pixels': f'{w}x{h}',
        'output_pixels': f'{restored.shape[1]}x{restored.shape[0]}',
        'scale': scale,
        'seconds': round(time.perf_counter() - started, 2),
        'input_edge_metric': edge_metric(original),
        'output_edge_metric': edge_metric(restored),
        'output_bytes': os.path.getsize(out_path),
        'sha256_prefix': sha256_file(out_path)[:12],
        'ocr': text,
    }


MANIFEST_FIELDS = ['file', 'output', 'input_pixels', 'output_pixels', 'scale', 'seconds',
                   'input_edge_metric', 'output_edge_metric', 'output_bytes', 'sha256_prefix']


def run(job):
    outdir = job['outdir']
    os.makedirs(outdir, exist_ok=True)
    profile = 'standard' if job.get('profile') == 'standard' else 'conservative'
    scale = 4 if int(job.get('scale', 2)) == 4 else 2
    tile = min(max(int(job.get('tile', 256)), 64), 512)
    want_ocr = bool(job.get('ocr', True))
    model()
    emit({'event': 'ready', 'device': DEVICE.type, 'threads': THREADS})
    rows = []
    for i, item in enumerate(job['inputs']):
        try:
            row = restore_one(item['path'], item['name'], outdir, profile, scale, tile, want_ocr, i)
            rows.append(row)
            emit({'event': 'file', 'index': i, 'row': row})
        except Exception as exc:
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            # Never leak server paths to the client.
            message = str(exc).replace(item['path'], item['name']).replace(outdir, '')
            emit({'event': 'error', 'index': i, 'file': item['name'], 'message': message})
    manifest = os.path.join(outdir, 'manifest.csv')
    with open(manifest, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    archive = os.path.join(outdir, 'restored_cards.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for row in rows:
            bundle.write(os.path.join(outdir, row['output']), row['output'])
        bundle.write(manifest, 'manifest.csv')
    emit({'event': 'done', 'restored': len(rows), 'failed': len(job['inputs']) - len(rows),
          'zip': 'restored_cards.zip', 'manifest': 'manifest.csv'})


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        model()
        emit({'event': 'ready', 'device': DEVICE.type, 'threads': THREADS})
        return
    try:
        run(json.load(sys.stdin))
    except Exception as exc:
        emit({'event': 'fatal', 'message': str(exc)})
        sys.exit(1)


if __name__ == '__main__':
    main()
