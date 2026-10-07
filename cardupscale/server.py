#!/usr/bin/env python3
"""CardCrop upscale tier: batch queue API + UI.

One worker thread runs batches in order (inference is CPU/GPU bound); within a batch a
failed card is recorded and skipped, never stopping the rest. Clients poll
GET /api/batches/{id}. Batches are kept for BATCH_TTL_HOURS, then deleted.

Run:  uvicorn server:app --host 0.0.0.0 --port 8800
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import queue
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

import upscale_tier as ut

log = logging.getLogger('cardcrop.server')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

DATA = Path(os.environ.get('CARDCROP_DATA_DIR', Path(__file__).resolve().parent / 'data'))
MAX_FILE_MB = int(os.environ.get('CARDCROP_MAX_FILE_MB', '40'))
BATCH_TTL_HOURS = float(os.environ.get('CARDCROP_BATCH_TTL_HOURS', '24'))
MAX_QUEUED = int(os.environ.get('CARDCROP_MAX_QUEUED', '20'))
ALLOWED_FORMATS = {'JPEG', 'PNG', 'WEBP', 'TIFF', 'BMP'}
ID_RE = re.compile(r'^[0-9a-f]{32}$')


@dataclasses.dataclass
class Batch:
    id: str
    dir: Path
    inputs: list[ut.CardInput]
    opts: ut.Options
    cards: list[dict]
    status: str = 'queued'        # queued | running | done
    created: float = dataclasses.field(default_factory=time.time)
    finished: Optional[float] = None
    results: list[ut.CardResult] = dataclasses.field(default_factory=list)


batches: dict[str, Batch] = {}
lock = threading.Lock()
work: 'queue.Queue[str]' = queue.Queue()


def _decode_problem(path: Path) -> Optional[str]:
    """Fully decode an upload once; return why it cannot be used, or None."""
    try:
        with Image.open(path) as im:
            ImageOps.exif_transpose(im).load()
        return None
    except Exception as exc:
        return f'Cannot decode this file ({type(exc).__name__}: {exc}). Re-export or re-scan it.'


def _card_view(i: int, name: str, comp: Optional[float]) -> dict:
    return {'index': i, 'file': name, 'comp_usd': comp, 'status': 'waiting', 'stage': '', 'progress': 0.0,
            'error': None, 'flags': [], 'high_value': comp is not None and comp >= ut.HIGH_VALUE_USD,
            'model': None, 'backend': None, 'input_pixels': '', 'output_pixels': '', 'seconds': None,
            'psnr_vs_input': None, 'text_retention': None, 'output': None, 'reasons': [], 'preview': True}


def _worker() -> None:
    while True:
        bid = work.get()
        b = batches.get(bid)
        if b is None:
            continue
        with lock:
            b.status = 'running'
        started = time.time()

        def update(i: int, res: ut.CardResult, stage: str, frac: float) -> None:
            with lock:
                c = b.cards[i]
                c.update(status=res.status, stage=stage, progress=round(frac, 3), error=res.error)
                if res.status in ('done', 'failed'):
                    c.update({k: getattr(res, k) for k in ('flags', 'high_value', 'model', 'backend', 'input_pixels',
                              'output_pixels', 'seconds', 'psnr_vs_input', 'text_retention', 'output', 'reasons')})

        # Cards that failed at intake are not sent to the worker; results keep batch order.
        todo = [i for i, c in enumerate(b.cards) if c['status'] == 'waiting']
        try:
            if todo:
                done = ut.process_batch([b.inputs[i] for i in todo], b.dir / 'out', b.opts,
                                        on_update=lambda j, res, st, fr: update(todo[j], res, st, fr))
            else:
                done = []
            by_index = dict(zip(todo, done))
            b.results = [by_index.get(i) or ut.CardResult(file=c['file'], status='failed', error=c['error'],
                                                          comp_usd=c['comp_usd']) for i, c in enumerate(b.cards)]
            ut.write_manifest(b.results, b.dir / 'out')
            ut.build_zip(b.results, b.dir / 'out', b.dir / 'cards.zip')
        except Exception:  # a batch-level failure (e.g. disk) must not kill the worker
            log.exception('batch %s crashed', bid)
            with lock:
                for c in b.cards:
                    if c['status'] in ('waiting', 'processing'):
                        c.update(status='failed', error='batch aborted by a server error')
        with lock:
            b.status, b.finished = 'done', time.time()
        n_ok = sum(c['status'] == 'done' for c in b.cards)
        log.info('batch %s: %d/%d done in %.0fs', bid, n_ok, len(b.cards), time.time() - started)


def _sweeper() -> None:
    while True:
        time.sleep(600)
        cutoff = time.time() - BATCH_TTL_HOURS * 3600
        with lock:
            stale = [k for k, b in batches.items() if b.status == 'done' and (b.finished or 0) < cutoff]
            for k in stale:
                shutil.rmtree(batches.pop(k).dir, ignore_errors=True)
        if stale:
            log.info('expired %d batch(es)', len(stale))


app = FastAPI(title='CardCrop upscale tier', docs_url=None, redoc_url=None)
threading.Thread(target=_worker, name='upscale-worker', daemon=True).start()
threading.Thread(target=_sweeper, name='batch-sweeper', daemon=True).start()


@app.middleware('http')
async def security_headers(request, call_next):
    resp = await call_next(request)
    resp.headers.update({
        'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer', 'X-Frame-Options': 'DENY',
        'Content-Security-Policy': "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; "
                                   "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
    })
    return resp


@app.get('/api/health')
def health() -> dict:
    models = {}
    for key, spec in ut.REGISTRY.items():
        if spec.task == 'benchmark':  # measured in bench/, never routed: not required in production
            continue
        try:
            ut.model_path(spec)
            onnx = bool(spec.onnx and (ut.MODEL_DIR / 'onnx' / spec.onnx).is_file())
            models[key] = {'task': spec.task, 'license': spec.license, 'ok': True, 'backend': 'onnxruntime' if onnx else 'torch'}
        except ut.ModelIntegrityError as exc:
            models[key] = {'task': spec.task, 'license': spec.license, 'ok': False, 'error': str(exc)}
    with lock:
        waiting = sum(b.status == 'queued' for b in batches.values())
        running = any(b.status == 'running' for b in batches.values())
    return {'ok': all(m['ok'] for m in models.values()), 'models': models, 'threads': ut.THREADS,
            'gemini': ut.GeminiInpainter().available, 'queue': waiting, 'running': running,
            'limits': {'maxBatch': ut.MAX_BATCH, 'maxFileMB': MAX_FILE_MB, 'highValueUsd': ut.HIGH_VALUE_USD,
                       'standardLongSide': ut.STANDARD_LONG_SIDE}}


@app.post('/api/batches', status_code=202)
async def create_batch(files: list[UploadFile] = File(...), comps: str = Form('{}'), model: str = Form('auto'),
                       damage: str = Form('off'), faces: str = Form('off')) -> dict:
    if not 1 <= len(files) <= ut.MAX_BATCH:
        raise HTTPException(413, f'Send 1 to {ut.MAX_BATCH} cards per batch.')
    if model not in ('auto', 'general', 'text') or damage not in ('off', 'local', 'gemini', 'auto') \
            or faces not in ('on', 'off'):
        raise HTTPException(400, 'Unknown option value.')
    try:
        comp_map = {str(k): (None if v in (None, '') else float(v)) for k, v in json.loads(comps).items()}
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(400, 'comps must be a JSON object of index -> USD.')
    with lock:
        if sum(b.status == 'queued' for b in batches.values()) >= MAX_QUEUED:
            raise HTTPException(503, 'The queue is full. Try again when a batch finishes.')
    bid = uuid.uuid4().hex
    bdir = DATA / bid
    (bdir / 'in').mkdir(parents=True)
    names = ut.unique_names(f.filename or f'card{i}.jpg' for i, f in enumerate(files))
    inputs, cards = [], []
    try:
        for i, (f, name) in enumerate(zip(files, names)):
            dest = bdir / 'in' / f'{i:03d}{Path(name).suffix.lower()[:6] or ".img"}'
            size = 0
            with open(dest, 'wb') as out:
                while chunk := await f.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_FILE_MB << 20:
                        raise HTTPException(413, f'{name}: larger than {MAX_FILE_MB} MB.')
                    out.write(chunk)
            try:
                with Image.open(dest) as im:
                    fmt = im.format
            except Exception:
                raise HTTPException(415, f'{name}: not an image.')
            if fmt not in ALLOWED_FORMATS:
                raise HTTPException(415, f'{name}: {fmt} is not supported (JPEG, PNG, WebP, TIFF, BMP).')
            comp = comp_map.get(str(i))
            inputs.append(ut.CardInput(dest, name, comp))
            cards.append(_card_view(i, name, comp))
            # A file that is an image but cannot be decoded (truncated or damaged upload) fails
            # on its own now; the rest of the batch still runs.
            problem = _decode_problem(dest)
            if problem:
                cards[-1].update(status='failed', error=problem, preview=False, progress=1.0)
    except HTTPException:
        shutil.rmtree(bdir, ignore_errors=True)
        raise
    b = Batch(bid, bdir, inputs, ut.Options(model=model, damage=damage, faces=faces), cards)
    with lock:
        batches[bid] = b
    work.put(bid)
    log.info('batch %s queued: %d card(s), model=%s damage=%s faces=%s', bid, len(cards), model, damage, faces)
    return batch_status(bid)


def _get(bid: str) -> Batch:
    b = batches.get(bid) if ID_RE.match(bid) else None
    if b is None:
        raise HTTPException(404, 'Batch not found (batches are kept for 24 hours).')
    return b


@app.get('/api/batches/{bid}')
def batch_status(bid: str) -> dict:
    b = _get(bid)
    with lock:
        cards = [dict(c) for c in b.cards]
        position = 0
        if b.status == 'queued':
            position = 1 + sum(o.status == 'queued' and o.created < b.created for o in batches.values())
        return {'id': b.id, 'status': b.status, 'position': position, 'created': b.created, 'finished': b.finished,
                'options': dataclasses.asdict(b.opts), 'cards': cards,
                'counts': {s: sum(c['status'] == s for c in cards) for s in ('waiting', 'processing', 'done', 'failed')}}


def _preview(src: Path, dest: Path, long_side: int = 1600) -> Path:
    if not dest.exists():
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            im.thumbnail((long_side, long_side), Image.Resampling.LANCZOS)
            im.save(dest, 'JPEG', quality=90)
    return dest


@app.get('/api/batches/{bid}/cards/{index}/{which}')
def card_image(bid: str, index: int, which: str) -> FileResponse:
    b = _get(bid)
    if not 0 <= index < len(b.inputs) or which not in ('before.jpg', 'after.jpg'):
        raise HTTPException(404, 'No such image.')
    prev = b.dir / 'preview'
    prev.mkdir(exist_ok=True)
    if which == 'before.jpg':
        if not b.cards[index].get('preview', True):
            raise HTTPException(422, 'This file cannot be decoded, so there is no preview.')
        try:
            path = _preview(b.inputs[index].path, prev / f'{index:03d}_before.jpg')
        except OSError:
            raise HTTPException(422, 'This file cannot be decoded, so there is no preview.')
    else:
        out = b.cards[index].get('output')
        if b.cards[index]['status'] != 'done' or not out:
            raise HTTPException(404, 'Not restored (yet).')
        path = _preview(b.dir / 'out' / out, prev / f'{index:03d}_after.jpg')
    return FileResponse(path, media_type='image/jpeg', headers={'Cache-Control': 'private, max-age=3600'})


@app.get('/api/batches/{bid}/zip')
def batch_zip(bid: str) -> FileResponse:
    b = _get(bid)
    if b.status != 'done':
        raise HTTPException(409, 'The batch is still running.')
    if not any(c['status'] == 'done' for c in b.cards):
        raise HTTPException(409, 'No card in this batch finished.')
    return FileResponse(b.dir / 'cards.zip', media_type='application/zip', filename=f'cardcrop-{bid[:8]}.zip')


@app.get('/api/batches/{bid}/manifest.csv')
def batch_manifest(bid: str) -> Response:
    b = _get(bid)
    path = b.dir / 'out' / 'manifest.csv'
    if not path.exists():
        raise HTTPException(409, 'The manifest is written when the batch finishes.')
    return FileResponse(path, media_type='text/csv', filename='manifest.csv')


@app.exception_handler(Exception)
async def unhandled(_request, exc: Exception):
    log.exception('unhandled error')
    return JSONResponse({'detail': 'Internal error'}, status_code=500)


app.mount('/', StaticFiles(directory=Path(__file__).resolve().parent / 'static', html=True), name='static')
