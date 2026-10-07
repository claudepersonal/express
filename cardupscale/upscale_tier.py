#!/usr/bin/env python3
"""CardCrop upscale / restoration tier: an open-weight replacement for Topaz Gigapixel.

Slots into the CardCrop pipeline as the "enhance" step:

    photo -> [WebGL filters] -> enhance_card() -> identify -> comps -> eBay CSV

Stages run in this order on every card (each is optional and routed per card):

  1. damage   (opt-in) surface specks/scratches are found, text is masked off, and the
              remaining pixels are repaired: locally (OpenCV Telea, deterministic) or
              with Gemini image editing. Gemini output is composited inside the mask
              only and must pass the TextGuard (OCR of every word near an edit is
              unchanged, nothing outside the mask moved) or the region is rejected.
              Retouched output is flagged: condition shown to buyers must stay true.
  2. upscale  4x super-resolution. "text" tier = SwinIR-M real-world x4, PSNR-trained
              (Apache-2.0): the most faithful model measured on the test cards, clean
              letterforms without invented strokes. "general" tier = Real-ESRGAN x4plus
              (BSD-3): 1.5x faster, sharper look, for cards with no readable text.
              ESRGAN-family models run on ONNX Runtime (27% faster than PyTorch, same
              output to 1e-6); SwinIR runs on PyTorch (ORT needs 3.3 GB for it, PyTorch
              1.0 GB). All are tiled with context padding, so seams never show.
              Standard cards already >= EBAY_READY_LONG_SIDE skip this stage.
  3. faces    (opt-in) GFPGAN v1.4 (Apache-2.0) on YuNet-detected faces (sideways
              cards included), aligned to the FFHQ template, blended at a weight.
              Off by default: it sharpens faces cosmetically but measured further
              from the real card than the plain upscale.
  4. verify   the output is downsampled back to the input grid and compared (PSNR /
              SSIM), and every confidently read input word is re-read at its own spot
              in the output (verify_words); illegible words are flagged.

Value routing: cards with a comp >= HIGH_VALUE_USD get the max-fidelity chain:
text-tier model, back-projection (the output is corrected until it downsamples
back to the original pixels, so no low-frequency content is invented), lossless
PNG, always the full 4x pass (never downsampled), larger tile context, no
generative face restore, no automatic retouching. They are flagged HIGH_VALUE.

Licensing: only weights whose licence allows commercial use are registered.
4x-UltraSharp (CC BY-NC-SA 4.0), CodeFormer (S-Lab, non-commercial) and SUPIR
(non-commercial, 24 GB+ GPU) are deliberately excluded; see EXCLUDED_MODELS.

CLI:  python upscale_tier.py cards/*.jpg --out out/ --comp 0819.jpg=650 --zip
"""
from __future__ import annotations

import argparse
import base64
import csv
import dataclasses
import difflib
import hashlib
import io
import json
import logging
import math
import os
import re
import threading
import time
import zipfile
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger('cardcrop.upscale')

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get('CARDCROP_MODEL_DIR', HERE / 'models'))

HIGH_VALUE_USD = float(os.environ.get('CARDCROP_HIGH_VALUE_USD', '500'))
MAX_BATCH = 50
# eBay-ready target for standard cards: long side in px (ample for zoom; 500x700 is the floor).
STANDARD_LONG_SIDE = int(os.environ.get('CARDCROP_STANDARD_LONG_SIDE', '2400'))
EBAY_MIN_LONG_SIDE = 700
# Standard cards whose long side already reaches this are eBay-ready: no AI pass (saves ~10 CPU-min/card).
EBAY_READY_LONG_SIDE = int(os.environ.get('CARDCROP_EBAY_READY_LONG_SIDE', '1600'))
MAX_OUTPUT_PIXELS = int(os.environ.get('CARDCROP_MAX_OUTPUT_PIXELS', str(64_000_000)))
MAX_INPUT_PIXELS = int(os.environ.get('CARDCROP_MAX_INPUT_PIXELS', str(40_000_000)))
# Auto routing: cards with at least this many OCR-confirmed words go to the text tier.
TEXT_ROUTE_MIN_WORDS = int(os.environ.get('CARDCROP_TEXT_ROUTE_MIN_WORDS', '3'))
# Back-projection passes for the max-fidelity chain (0 = off). See back_project().
HIGH_VALUE_CONSISTENCY_ITERS = 4
LOCAL_DAMAGE_MAX_COVERAGE = 0.05   # above this the damage is "heavy": Gemini or human review
FIDELITY_MIN_PSNR = 26.0           # output downsampled vs input; below this -> review flag
TEXT_DRIFT_MAX = 0.10              # share of the input's confidently read words not legible in the output


# --------------------------------------------------------------------------- models
@dataclasses.dataclass(frozen=True)
class ModelSpec:
    key: str
    task: str            # general | text | face | face_detect
    file: str
    sha256: str
    url: str
    license: str
    scale: int = 4
    onnx: Optional[str] = None       # exported graph in models/onnx/
    onnx_tile: Optional[int] = None  # fixed input size of that graph (None = dynamic)


REGISTRY: dict[str, ModelSpec] = {s.key: s for s in [
    ModelSpec('realesrgan_x4plus', 'general', 'RealESRGAN_x4plus.pth',
              '4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1',
              'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
              'BSD-3-Clause', onnx='realesrgan_x4plus.onnx'),
    ModelSpec('swinir_m_x4_psnr', 'text', '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth',
              '1fd8fed99684bd271db55563e4906d36459cf535446820053f7a1081d4781dc5',
              'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR.pth',
              # PyTorch, not ONNX: ORT constant-folds SwinIR's attention tables to a 3.3 GB
              # session (measured) vs 1.0 GB peak in PyTorch at ~10% lower speed.
              'Apache-2.0'),
    # Measured in bench/ but not routed: lower fidelity than the PSNR SwinIR on every test card.
    ModelSpec('swinir_m_x4_gan', 'benchmark', '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth',
              'b9afb61e65e04eb7f8aba5095d070bbe9af28df76acd0c9405aeb33b814bcfc6',
              'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth',
              'Apache-2.0'),
    ModelSpec('realesrnet_x4plus', 'benchmark', 'RealESRNet_x4plus.pth',
              'a820b9bde89a874d7599d545567308ce6c128fc8754a53208eda016d40aa81df',
              'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth',
              'BSD-3-Clause', onnx='realesrnet_x4plus.onnx'),
    ModelSpec('gfpgan_v1_4', 'face', 'GFPGANv1.4.pth',
              'e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad',
              'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
              'Apache-2.0', scale=1),
    ModelSpec('yunet_2023mar', 'face_detect', 'face_detection_yunet_2023mar.onnx',
              '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4',
              'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
              'MIT', scale=1),
]}
TIER_MODEL = {'general': 'realesrgan_x4plus', 'text': 'swinir_m_x4_psnr'}

EXCLUDED_MODELS = {
    '4x-UltraSharp': 'CC BY-NC-SA 4.0: non-commercial only; SwinIR-M (Apache-2.0) is the text tier instead',
    'CodeFormer': 'S-Lab License 1.0: non-commercial only; GFPGAN v1.4 (Apache-2.0) is the face tier instead',
    'SUPIR': 'non-commercial licence and needs a 24 GB+ GPU (SDXL backbone); heavy damage routes to Gemini masked edit or review',
}


class ModelIntegrityError(RuntimeError):
    pass


def available_cpus() -> int:
    """CPUs this process may use (cgroup quota and affinity, not the host's core count)."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()[:2]
        if quota != 'max':
            n = min(n, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return max(1, n)


THREADS = int(os.environ.get('CARDCROP_THREADS') or available_cpus())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


_verified: set[Path] = set()


def model_path(spec: ModelSpec) -> Path:
    """Path to verified weights. Every file is SHA-256 checked once per process."""
    path = MODEL_DIR / spec.file
    if not path.is_file():
        raise ModelIntegrityError(f'{spec.key}: weights missing at {path} (download: {spec.url})')
    if path not in _verified:
        got = sha256_file(path)
        if got != spec.sha256:
            raise ModelIntegrityError(f'{spec.key}: checksum mismatch ({got[:12]}...), refusing to load')
        _verified.add(path)
    return path


class Upscaler:
    """A 4x model behind one call: ONNX Runtime if the exported graph exists, else PyTorch."""

    def __init__(self, spec: ModelSpec, prefer_onnx: bool = True):
        self.spec = spec
        model_path(spec)  # integrity check against the reference weights either way
        onnx_path = MODEL_DIR / 'onnx' / spec.onnx if spec.onnx else None
        if prefer_onnx and onnx_path and onnx_path.is_file():
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = THREADS
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._sess = ort.InferenceSession(str(onnx_path), opts, providers=['CPUExecutionProvider'])
            self.backend = 'onnxruntime'
            self.fixed_tile = spec.onnx_tile
        else:
            import torch
            from spandrel import ModelLoader
            torch.set_num_threads(THREADS)
            self._torch = torch
            self._net = ModelLoader().load_from_file(str(model_path(spec))).model.eval()
            self.backend = 'torch'
            self.fixed_tile = None
        log.info('loaded %s via %s (%d threads)', spec.key, self.backend, THREADS)

    def _run(self, chw: np.ndarray) -> np.ndarray:
        x = chw[None].astype(np.float32)
        if self.backend == 'onnxruntime':
            return self._sess.run(None, {'input': x})[0][0]
        with self._torch.inference_mode():
            return self._net(self._torch.from_numpy(x)).numpy()[0]

    def upscale(self, rgb: np.ndarray, tile: int = 192, pad: int = 16,
                progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
        """Tiled 4x inference. Each tile sees `pad` px of real neighbouring context on every
        side (mirrored at the image border); only its centre is kept, so no seams."""
        tile = self.fixed_tile or tile
        step = tile - 2 * pad
        if step < 16:
            raise ValueError('tile must exceed 2*pad by at least 16 px')
        s = self.spec.scale
        h, w = rgb.shape[:2]
        src = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        # Mirror the whole image once so every tile window is in-bounds and full size.
        mode = 'reflect' if min(h, w) > pad + tile else 'symmetric'
        padded = np.pad(src, ((0, 0), (pad, pad + tile), (pad, pad + tile)), mode=mode)
        out = np.empty((h * s, w * s, 3), dtype=np.uint8)
        ys, xs = range(0, h, step), range(0, w, step)
        total, done = len(ys) * len(xs), 0
        for y0 in ys:
            for x0 in xs:
                win = padded[:, y0:y0 + tile, x0:x0 + tile]
                res = self._run(win)
                th, tw = min(step, h - y0), min(step, w - x0)
                core = res[:, pad * s:(pad + th) * s, pad * s:(pad + tw) * s]
                out[y0 * s:(y0 + th) * s, x0 * s:(x0 + tw) * s] = (
                    np.clip(core, 0, 1).transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
                done += 1
                if progress:
                    progress(done, total)
        return out


_cache: dict[str, object] = {}
_cache_lock = threading.Lock()


def get_upscaler(key: str) -> Upscaler:
    with _cache_lock:
        if key not in _cache:
            _cache[key] = Upscaler(REGISTRY[key])
        return _cache[key]  # type: ignore[return-value]


# --------------------------------------------------------------------------- text (OCR)
@dataclasses.dataclass
class Word:
    text: str
    box: tuple[int, int, int, int]   # x, y, w, h in the coordinates of the analysed image
    conf: float
    turn: int = 0                     # counter-clockwise turn (deg) that made the text upright


def _ocr_pass(gray: Image.Image, min_conf: float) -> list[tuple[str, tuple[int, int, int, int], float]]:
    import pytesseract
    d = pytesseract.image_to_data(gray, config='--psm 11', output_type=pytesseract.Output.DICT)
    out = []
    for t, c, x, y, w, h in zip(d['text'], d['conf'], d['left'], d['top'], d['width'], d['height']):
        t = t.strip()
        if float(c) >= min_conf and sum(ch.isalnum() for ch in t) >= 2:
            out.append((t, (int(x), int(y), int(w), int(h)), float(c)))
    return out


def ocr_words(rgb: np.ndarray, min_conf: float = 60.0) -> list[Word]:
    """Tesseract words with confidence, in `rgb` coordinates. Card scans are often
    stored sideways, so 90/270-degree turns are tried when the upright pass finds
    little text; the turn reading the most confident text wins. Images under
    1200 px are read at 2x."""
    h, w = rgb.shape[:2]
    k = 2.0 if max(h, w) < 1200 else 1.0
    gray = Image.fromarray(rgb).convert('L')
    if k != 1.0:
        gray = gray.resize((int(w * k), int(h * k)), Image.Resampling.LANCZOS)
    W, H = gray.size
    best: tuple[float, int, list] = (-1.0, 0, [])
    for turn in (0, 90, 270):
        img = gray if turn == 0 else gray.rotate(turn, expand=True)  # PIL rotates counter-clockwise
        found = _ocr_pass(img, min_conf)
        score = sum(c for _, _, c in found)
        if score > best[0]:
            best = (score, turn, found)
        if turn == 0 and len(found) >= 8:
            break
    _, turn, found = best
    words = []
    for t, (x, y, bw, bh), c in found:
        if turn == 90:      # rotated (x, y) -> original: col = W - (y + h), row = x
            x, y, bw, bh = W - (y + bh), x, bh, bw
        elif turn == 270:   # col = y, row = H - (x + w)
            x, y, bw, bh = y, H - (x + bw), bh, bw
        words.append(Word(t, (int(x / k), int(y / k), int(math.ceil(bw / k)), int(math.ceil(bh / k))), c, turn))
    return words


def _norm_token(t: str) -> str:
    return re.sub(r'[^0-9A-Za-z/#-]', '', t).upper()


def text_mask(shape: tuple[int, int], words: Iterable[Word], grow: int = 3) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    for wd in words:
        x, y, w, h = wd.box
        m[max(0, y - grow):y + h + grow, max(0, x - grow):x + w + grow] = True
    return m


SOLID_CONF = 75.0   # OCR confidence for a word to count as reliably read
SOLID_LEN = 3       # and at least this many characters after normalisation


def _plausible(w: Word) -> bool:
    """A real word's box is long enough along the reading direction for its letters
    (OCR occasionally 'reads' a confident short word off a speck)."""
    along, across = (w.box[2], w.box[3]) if w.turn == 0 else (w.box[3], w.box[2])
    return along >= 0.25 * len(_norm_token(w.text)) * across


def solid_words(words: list[Word]) -> list[Word]:
    """Words OCR read with confidence and plausible geometry: the ones a change could
    not be blamed on OCR jitter."""
    return [w for w in words if w.conf >= SOLID_CONF and len(_norm_token(w.text)) >= SOLID_LEN and _plausible(w)]


def solid_tokens(words: list[Word]) -> list[str]:
    return [_norm_token(w.text) for w in solid_words(words)]


def text_changes(before: list[Word], after: list[Word]) -> tuple[list[str], list[str]]:
    """(lost, added): confidently read words of `before` that `after` does not contain at
    any confidence, and confidently read words of `after` that `before` never had."""
    from collections import Counter
    all_before = Counter(_norm_token(w.text) for w in before)
    all_after = Counter(_norm_token(w.text) for w in after)
    lost = list((Counter(solid_tokens(before)) - all_after).elements())
    added = list((Counter(solid_tokens(after)) - all_before).elements())
    return lost, added


def text_retention(before: list[Word], after: list[Word]) -> float:
    """Share of the input's confidently read words that the output still contains."""
    solid = solid_tokens(before)
    if not solid:
        return 1.0
    lost, _ = text_changes(before, after)
    return 1.0 - len(lost) / len(solid)


def verify_words(words: list[Word], out_rgb: np.ndarray, scale: float) -> tuple[float, list[str]]:
    """Check that every confidently read input word is still legible in the output.

    Each word's box is scaled onto the output, cropped with a margin, turned upright,
    sized to ~48 px text height and read as a single line (psm 7, then psm 8). A word passes when
    its letters appear in that reading. Far more reliable than comparing two whole-page
    OCR passes at different scales. Returns (share kept, words lost)."""
    import pytesseract
    solid = solid_words(words)
    if not solid:
        return 1.0, []
    H, W = out_rgb.shape[:2]
    lost = []
    for w in solid:
        x, y, bw, bh = (v * scale for v in w.box)
        m = 0.35 * min(bw, bh) + 2
        x0, y0 = int(max(0, x - m)), int(max(0, y - m))
        x1, y1 = int(min(W, x + bw + m)), int(min(H, y + bh + m))
        crop = Image.fromarray(out_rgb[y0:y1, x0:x1]).convert('L')
        if w.turn:
            crop = crop.rotate(w.turn, expand=True)
        text_h = (bh if w.turn == 0 else bw) or 1
        k = float(np.clip(48.0 / text_h, 0.5, 4.0))
        crop = crop.resize((max(1, int(crop.width * k)), max(1, int(crop.height * k))), Image.Resampling.LANCZOS)
        tok = _norm_token(w.text)
        need = max(SOLID_LEN, math.ceil(0.8 * len(tok)))
        for psm in (7, 8):  # single line first; single word for display lettering
            read = re.sub(r'[^0-9A-Z/#-]', '', pytesseract.image_to_string(crop, config=f'--psm {psm}').upper())
            if difflib.SequenceMatcher(None, tok, read, autojunk=False).find_longest_match(0, len(tok), 0, len(read)).size >= need:
                break
        else:
            lost.append(w.text)
    return 1.0 - len(lost) / len(solid), lost


def text_similarity(a: list[Word], b: list[Word]) -> float:
    """Order-insensitive token agreement (1.0 = same words)."""
    ta = sorted(_norm_token(w.text) for w in a if _norm_token(w.text))
    tb = sorted(_norm_token(w.text) for w in b if _norm_token(w.text))
    if not ta and not tb:
        return 1.0
    return difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()


# --------------------------------------------------------------------------- damage
PRINT_DENSITY = 0.12   # share of fine-scale edge pixels in a 15x15 window that marks dense print


def print_structure(rgb: np.ndarray, work_long_side: int = 600) -> np.ndarray:
    """Pixels inside or next to dense printing (fine text, logos, patterned borders).
    Small print at low resolution looks like a cluster of specks to any local test;
    what separates it from damage is density: print is packed with edges, scuffs sit
    on smooth fields. Measured at ~600 px on the long side, where halftone grain of a
    high-dpi scan averages out but lettering stays dense; mapped back to full size.
    Used by the defect detector and, independently, by TextGuard."""
    h, w = rgb.shape[:2]
    k = min(1.0, work_long_side / max(h, w))
    small = cv2.resize(rgb, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA) if k < 1 else rgb
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    edges = (cv2.Canny(cv2.GaussianBlur(gray, (0, 0), 0.8), 25, 70) > 0).astype(np.float32)
    dense = (cv2.boxFilter(edges, -1, (15, 15)) >= PRINT_DENSITY).astype(np.uint8)
    dense = cv2.dilate(dense, np.ones((5, 5), np.uint8))
    if k < 1:
        dense = cv2.resize(dense, (w, h), interpolation=cv2.INTER_NEAREST)
    return dense.astype(bool)


def detect_surface_defects(rgb: np.ndarray, protect: np.ndarray, sensitivity: float = 4.5,
                           exclude_print: bool = True) -> np.ndarray:
    """Mask of surface specks, scuffs and fine scratches.

    Defects on a card scan are small marks that are lighter and less saturated than
    the print around them (scuffed gloss, paper fibre, dust), or dark dust on light
    areas. The image is softened just enough to suppress halftone grain, compared with
    a local median in Lab, and thresholded at `sensitivity` robust standard deviations.
    Printed edges (lettering, borders, artwork outlines), dense print
    (print_structure) and `protect` (OCR text boxes) are excluded, so print is never
    treated as damage. `exclude_print=False` exists only to reproduce the failure the
    density rule fixes (bench/guard_and_damage.py)."""
    soft = cv2.GaussianBlur(rgb, (0, 0), 1.0)
    lab = cv2.cvtColor(soft, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]
    C = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    bgL = cv2.medianBlur(L.astype(np.uint8), 11).astype(np.float32)
    bgC = cv2.medianBlur(np.clip(C, 0, 255).astype(np.uint8), 11).astype(np.float32)
    dL, dC = L - bgL, bgC - C
    mad = float(np.median(np.abs(dL - np.median(dL)))) * 1.4826 + 1e-3
    t = max(10.0, sensitivity * mad)
    cand = ((dL > t) & (dC > -2)) | ((dL < -t) & (bgL > 170))
    printed = cv2.dilate(cv2.Canny(cv2.GaussianBlur(rgb, (0, 0), 2.5), 30, 90), np.ones((7, 7), np.uint8)) > 0
    keep_out = printed | protect
    if exclude_print:
        keep_out |= print_structure(rgb)
    cand &= ~keep_out
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    area_cap = max(40, int(rgb.shape[0] * rgb.shape[1] * 0.0004))
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= 3) & (stats[1:, cv2.CC_STAT_AREA] <= area_cap)
    mask = cv2.dilate(keep[labels].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return mask & ~(protect | print_structure(rgb)) if exclude_print else mask & ~protect


@dataclasses.dataclass
class DamageReport:
    mode: str                  # off | local | gemini | review
    coverage: float            # share of card pixels in the defect mask
    regions_total: int = 0
    regions_applied: int = 0
    regions_rejected: int = 0
    max_change_outside_mask: int = 0
    notes: list[str] = dataclasses.field(default_factory=list)


def repair_local(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Deterministic fill of masked pixels from their surroundings (Telea fast marching)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    fixed = cv2.inpaint(bgr, mask.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    out = cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB)
    out[~mask] = rgb[~mask]  # guarantee: nothing outside the mask changes
    return out


def mask_regions(mask: np.ndarray, margin: int = 24, max_regions: int = 8) -> list[tuple[int, int, int, int]]:
    """Group mask pixels into at most `max_regions` crop boxes (x0, y0, x1, y1) with context."""
    grown = cv2.dilate(mask.astype(np.uint8), np.ones((margin, margin), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
    boxes = sorted(((int(s[0]), int(s[1]), int(s[0] + s[2]), int(s[1] + s[3])) for s in stats[1:]),
                   key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    while len(boxes) > max_regions:  # merge the two smallest boxes' neighbourhoods
        a, b = boxes.pop(), boxes.pop()
        boxes.append((min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])))
        boxes.sort(key=lambda bx: -(bx[2] - bx[0]) * (bx[3] - bx[1]))
    h, w = mask.shape
    return [(max(0, x0), max(0, y0), min(w, x1), min(h, y1)) for x0, y0, x1, y1 in boxes]


class TextGuard:
    """Accepts an edited region only if:
      1. no pixel outside the mask changed;
      2. the mask does not touch dense print (print_structure), whatever produced it;
      3. the editor did not re-render the unmasked context (shifted/regenerated image);
      4. OCR near the edit loses or gains no confidently read word, and the number of
         characters OCR sees at any confidence stays within CHAR_TOLERANCE. Check 4b
         catches fine print erased at sizes OCR can only half-read."""

    CHAR_TOLERANCE = 0.10

    def __init__(self, min_context_ssim: float = 0.90, ocr_margin: int = 40):
        self.min_context_ssim = min_context_ssim
        self.ocr_margin = ocr_margin

    def check(self, original: np.ndarray, edited: np.ndarray, mask: np.ndarray,
              box: tuple[int, int, int, int], candidate: Optional[np.ndarray] = None) -> tuple[bool, str]:
        from skimage.metrics import structural_similarity
        outside = ~mask
        if np.any(original[outside] != edited[outside]):
            return False, 'pixels outside the damage mask changed'
        touched = mask & print_structure(original)
        if touched.any():
            return False, f'edit touches dense print ({int(touched.sum())} px)'
        if candidate is not None:  # raw editor output for the crop, before compositing
            x0, y0, x1, y1 = box
            ctx = ~mask[y0:y1, x0:x1]
            if ctx.sum() > 64:
                g0 = cv2.cvtColor(original[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
                g1 = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY)
                _, smap = structural_similarity(g0, g1, full=True, data_range=255)
                ctx_ssim = float(smap[ctx].mean())
                if ctx_ssim < self.min_context_ssim:
                    return False, f'editor changed unmasked context (SSIM {ctx_ssim:.3f})'
        x0, y0, x1, y1 = box
        m = self.ocr_margin
        H, W = mask.shape
        sl = (slice(max(0, y0 - m), min(H, y1 + m)), slice(max(0, x0 - m), min(W, x1 + m)))
        before, after = ocr_words(original[sl], min_conf=10), ocr_words(edited[sl], min_conf=10)
        lost, added = text_changes(before, after)
        if lost or added:
            return False, f'text near the edit changed: lost {lost[:6]}, added {added[:6]}'
        cb = sum(len(_norm_token(w.text)) for w in before)
        ca = sum(len(_norm_token(w.text)) for w in after)
        if cb >= 10 and abs(ca - cb) > self.CHAR_TOLERANCE * cb:
            return False, f'characters read near the edit went from {cb} to {ca}'
        return True, 'ok'


class GeminiInpainter:
    """Masked repair with Gemini image editing (the AI Studio project's API key).

    Only crops around damage are sent, with the mask as a second image. The returned
    image is resized to the crop and composited inside the mask only; every region
    then has to pass TextGuard or the original pixels are kept."""

    ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
    PROMPT = ('You are repairing a photo of a trading card. Image 1 is a crop of the card; image 2 is a '
              'mask where WHITE marks surface damage (scratches, specks, print dust). Remove only the '
              'damage under the white mask so it matches the surrounding surface. Do not change, redraw, '
              'sharpen or re-letter any text, numbers, names, logos, faces or borders. Return the full '
              'crop at the same framing and size.')

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, timeout: float = 90.0):
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        self.model = model or os.environ.get('GEMINI_IMAGE_MODEL', 'gemini-2.5-flash-image')
        self.timeout = timeout
        self.guard = TextGuard()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _png_b64(arr: np.ndarray) -> str:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, 'PNG')
        return base64.b64encode(buf.getvalue()).decode()

    def _edit(self, crop: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
        import requests
        body = {
            'contents': [{'parts': [
                {'text': self.PROMPT},
                {'inline_data': {'mime_type': 'image/png', 'data': self._png_b64(crop)}},
                {'inline_data': {'mime_type': 'image/png', 'data': self._png_b64(crop_mask.astype(np.uint8) * 255)}},
            ]}],
            'generationConfig': {'responseModalities': ['IMAGE'], 'temperature': 0.0},
        }
        url = self.ENDPOINT.format(model=self.model)
        for attempt in range(3):
            r = requests.post(url, json=body, headers={'x-goog-api-key': self.api_key}, timeout=self.timeout)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 ** attempt * 2)
                continue
            r.raise_for_status()
            for cand in r.json().get('candidates', []):
                for part in cand.get('content', {}).get('parts', []):
                    blob = part.get('inlineData') or part.get('inline_data')
                    if blob and blob.get('data'):
                        img = Image.open(io.BytesIO(base64.b64decode(blob['data']))).convert('RGB')
                        h, w = crop.shape[:2]
                        return np.asarray(img.resize((w, h), Image.Resampling.LANCZOS))
            raise RuntimeError('Gemini returned no image (possibly blocked by safety filters)')
        raise RuntimeError('Gemini edit failed after retries')

    def repair(self, rgb: np.ndarray, mask: np.ndarray, report: DamageReport) -> np.ndarray:
        out = rgb.copy()
        for box in mask_regions(mask):
            x0, y0, x1, y1 = box
            report.regions_total += 1
            region_mask = np.zeros_like(mask)
            region_mask[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
            try:
                cand = self._edit(rgb[y0:y1, x0:x1], mask[y0:y1, x0:x1])
            except Exception as exc:  # network, quota, safety: keep the original pixels
                report.regions_rejected += 1
                report.notes.append(f'region {box}: Gemini error: {exc}')
                log.warning('gemini region %s failed: %s', box, exc)
                continue
            trial = out.copy()
            sub = trial[y0:y1, x0:x1]
            sub[region_mask[y0:y1, x0:x1]] = cand[region_mask[y0:y1, x0:x1]]
            ok, why = self.guard.check(out, trial, region_mask, box, candidate=cand)
            if ok:
                out = trial
                report.regions_applied += 1
            else:
                report.regions_rejected += 1
                report.notes.append(f'region {box}: rejected by TextGuard: {why}')
                log.warning('gemini region %s rejected: %s', box, why)
        return out


def repair_damage(rgb: np.ndarray, words: list[Word], mode: str) -> tuple[np.ndarray, DamageReport, np.ndarray]:
    """mode: off | local | gemini | auto. Returns (image, report, mask)."""
    protect = text_mask(rgb.shape[:2], words, grow=4)
    mask = detect_surface_defects(rgb, protect)
    coverage = float(mask.mean())
    report = DamageReport(mode='off', coverage=round(coverage, 5))
    if mode == 'off' or not mask.any():
        return rgb, report, mask
    gemini = GeminiInpainter()
    heavy = coverage > LOCAL_DAMAGE_MAX_COVERAGE
    if mode == 'gemini' or (mode == 'auto' and heavy):
        if gemini.available:
            report.mode = 'gemini'
            out = gemini.repair(rgb, mask, report)
        else:
            report.mode = 'review'
            report.notes.append('heavy damage or Gemini requested, but no GEMINI_API_KEY is set: left unretouched for review')
            return rgb, report, mask
    else:
        report.mode = 'local'
        out = repair_local(rgb, mask)
        report.regions_total = report.regions_applied = len(mask_regions(mask))
        ok, why = TextGuard().check(rgb, out, mask, (0, 0, rgb.shape[1], rgb.shape[0]))
        if not ok:
            report.notes.append(f'local repair rejected by TextGuard: {why}')
            report.regions_rejected, report.regions_applied = report.regions_total, 0
            out = rgb
    report.max_change_outside_mask = int(np.abs(out.astype(int) - rgb.astype(int))[~mask].max(initial=0))
    return out, report, mask


# --------------------------------------------------------------------------- faces
FFHQ_512 = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                     [201.26117, 371.41043], [313.08905, 371.15118]], dtype=np.float32)


class FaceRestorer:
    def __init__(self):
        import torch
        from spandrel import ModelLoader
        torch.set_num_threads(THREADS)
        self._torch = torch
        self._net = ModelLoader().load_from_file(str(model_path(REGISTRY['gfpgan_v1_4']))).model.eval()
        self._det_path = str(model_path(REGISTRY['yunet_2023mar']))

    def _detect_upright(self, rgb: np.ndarray, min_size: int, min_score: float) -> list[np.ndarray]:
        h, w = rgb.shape[:2]
        k = min(1.0, 1600 / max(h, w))  # YuNet is fast and accurate at ~1600 px
        small = cv2.resize(rgb, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else rgb
        det = cv2.FaceDetectorYN.create(self._det_path, '', (small.shape[1], small.shape[0]), min_score, 0.3, 50)
        _, faces = det.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        out = []
        for f in faces if faces is not None else []:
            if f[2] / k >= min_size and f[3] / k >= min_size:
                out.append(f[4:14].reshape(5, 2) / k)
        return out

    def detect(self, rgb: np.ndarray, min_size: int = 48, min_score: float = 0.85) -> list[np.ndarray]:
        """Five-point landmarks of each face, in `rgb` coordinates. Card scans are often
        sideways, so quarter turns are tried when the upright pass finds nothing."""
        faces = self._detect_upright(rgb, min_size, min_score)
        if faces:
            return faces
        h, w = rgb.shape[:2]
        for k in (1, 3):  # np.rot90: k=1 counter-clockwise, k=3 clockwise
            found = self._detect_upright(np.ascontiguousarray(np.rot90(rgb, k)), min_size, min_score)
            if found:
                if k == 1:   # rotated (x', y') -> original (W-1-y', x')
                    return [np.stack([w - 1 - f[:, 1], f[:, 0]], axis=1) for f in found]
                return [np.stack([f[:, 1], h - 1 - f[:, 0]], axis=1) for f in found]
        return []

    def restore(self, rgb: np.ndarray, weight: float) -> tuple[np.ndarray, int]:
        """Restore every detected face, blending `weight` of the GFPGAN result back in."""
        out = rgb.astype(np.float32)
        faces = self.detect(rgb)
        for lm in faces:
            M, _ = cv2.estimateAffinePartial2D(lm.astype(np.float32), FFHQ_512, method=cv2.LMEDS)
            if M is None:
                continue
            crop = cv2.warpAffine(rgb, M, (512, 512), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
            x = self._torch.from_numpy(np.ascontiguousarray(crop.astype(np.float32).transpose(2, 0, 1)[None] / 127.5 - 1.0))
            with self._torch.inference_mode():
                y = self._net(x, return_rgb=False, randomize_noise=False)[0]  # deterministic
            face = ((y[0].clamp(-1, 1).numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.float32)
            inv = cv2.invertAffineTransform(M)
            h, w = rgb.shape[:2]
            back = cv2.warpAffine(face, inv, (w, h), flags=cv2.INTER_LANCZOS4)
            m = np.zeros((512, 512), np.float32)
            cv2.ellipse(m, (256, 290), (190, 230), 0, 0, 360, 1.0, -1)
            m = cv2.GaussianBlur(m, (0, 0), 18)
            alpha = cv2.warpAffine(m, inv, (w, h))[..., None] * weight
            out = out * (1 - alpha) + back * alpha
        return np.clip(out + 0.5, 0, 255).astype(np.uint8), len(faces)


def get_face_restorer() -> FaceRestorer:
    with _cache_lock:
        if 'faces' not in _cache:
            _cache['faces'] = FaceRestorer()
        return _cache['faces']  # type: ignore[return-value]


# --------------------------------------------------------------------------- routing
@dataclasses.dataclass
class Options:
    model: str = 'auto'           # auto | text | general
    damage: str = 'off'           # off | local | gemini | auto   (retouching is opt-in)
    faces: str = 'off'            # off | on. Opt-in: on the test cards GFPGAN moved faces
                                  # further from the real scan (bench/face_check.py).
    face_weight: float = 0.5      # 0 = original face, 1 = full GFPGAN
    jpeg_quality: int = 95


@dataclasses.dataclass
class Plan:
    value_tier: str               # standard | high
    model_key: Optional[str]      # None = no AI upscale needed
    tile: int
    pad: int
    consistency_iters: int        # back-projection passes (max-fidelity chain)
    damage: str
    faces: bool
    output_format: str            # jpeg | png
    target_long_side: Optional[int]
    reasons: list[str]


def make_plan(rgb: np.ndarray, words: list[Word], comp_usd: Optional[float], opts: Options) -> Plan:
    h, w = rgb.shape[:2]
    reasons: list[str] = []
    high = comp_usd is not None and comp_usd >= HIGH_VALUE_USD
    if high:
        reasons.append(f'comp ${comp_usd:,.0f} >= ${HIGH_VALUE_USD:,.0f}: max-fidelity chain')
    if opts.model in TIER_MODEL:
        tier = opts.model
        reasons.append(f'{tier} tier forced')
    elif high:
        tier = 'text'
        reasons.append('auto: high-value -> text tier (most faithful model)')
    else:
        tier = 'text' if len(words) >= TEXT_ROUTE_MIN_WORDS else 'general'
        reasons.append(f'auto: {len(words)} readable word(s) -> {tier} tier')
    model_key: Optional[str] = TIER_MODEL[tier]
    if not high and max(h, w) >= EBAY_READY_LONG_SIDE:
        model_key = None
        reasons.append(f'already {w}x{h} (>= {EBAY_READY_LONG_SIDE} px): eBay-ready, no AI upscale')
    if high and h * w * 16 > MAX_OUTPUT_PIXELS:
        reasons.append(f'4x would exceed {MAX_OUTPUT_PIXELS / 1e6:.0f} MP: output is capped')
    if high:
        reasons.append('back-projection on: output must downsample to the original pixels')
    damage = opts.damage
    if high and damage != 'off':
        damage = 'off'
        reasons.append('retouching disabled for high-value cards (condition must stay as photographed)')
    # Faces are restored only on request, only on upscaled output, never on high-value cards.
    faces = opts.faces == 'on' and not high and model_key is not None
    if high and opts.faces == 'on':
        reasons.append('generative face restore skipped: high-value card')
    return Plan(
        value_tier='high' if high else 'standard', model_key=model_key,
        tile=192, pad=32 if high else 16, consistency_iters=HIGH_VALUE_CONSISTENCY_ITERS if high else 0,
        damage=damage, faces=faces,
        output_format='png' if high else 'jpeg',
        target_long_side=None if high else STANDARD_LONG_SIDE, reasons=reasons)


def back_project(sr: np.ndarray, lr: np.ndarray, iters: int) -> np.ndarray:
    """Iterative back-projection: push the upscale toward the unique image whose
    area-downsample equals the input. Detail the model added above the input's
    Nyquist limit survives; anything that contradicts the real pixels (tone shifts,
    smoothed-away texture, invented low-frequency structure) is pulled back."""
    if iters <= 0:
        return sr
    out = sr.astype(np.float32)
    low = lr.astype(np.float32)
    (H, W), (h, w) = sr.shape[:2], lr.shape[:2]
    for _ in range(iters):
        err = low - cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
        out += cv2.resize(err, (W, H), interpolation=cv2.INTER_CUBIC)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- verification
def fidelity(original: np.ndarray, output: np.ndarray) -> dict:
    """Downsample the output onto the input grid and compare: a model that invents
    structure (or a shifted edit) cannot reproduce the input it started from."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    h, w = original.shape[:2]
    down = cv2.resize(output, (w, h), interpolation=cv2.INTER_AREA)
    g0 = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(down, cv2.COLOR_RGB2GRAY)
    return {'psnr_vs_input': round(float(peak_signal_noise_ratio(original, down, data_range=255)), 2),
            'ssim_vs_input': round(float(structural_similarity(g0, g1, data_range=255)), 4)}


# --------------------------------------------------------------------------- one card
@dataclasses.dataclass
class CardResult:
    file: str
    output: Optional[str] = None
    status: str = 'waiting'       # waiting | processing | done | failed
    error: Optional[str] = None
    comp_usd: Optional[float] = None
    value_tier: str = 'standard'
    high_value: bool = False
    model: Optional[str] = None
    backend: Optional[str] = None
    consistency_iters: int = 0
    input_pixels: str = ''
    output_pixels: str = ''
    seconds: float = 0.0
    damage_mode: str = 'off'
    damage_coverage: float = 0.0
    retouched: bool = False
    faces_restored: int = 0
    psnr_vs_input: Optional[float] = None
    ssim_vs_input: Optional[float] = None
    text_retention: Optional[float] = None
    flags: list[str] = dataclasses.field(default_factory=list)
    reasons: list[str] = dataclasses.field(default_factory=list)
    input_sha256: str = ''
    output_sha256: str = ''


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.width * im.height > MAX_INPUT_PIXELS:
            raise ValueError(f'{im.width}x{im.height} exceeds the {MAX_INPUT_PIXELS / 1e6:.0f} MP input limit')
        return np.asarray(im.convert('RGB'))


def _fit_long_side(rgb: np.ndarray, long_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    k = long_side / max(h, w)
    if k >= 1:
        return rgb
    return cv2.resize(rgb, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)


def enhance_card(path: Path, out_dir: Path, comp_usd: Optional[float] = None, opts: Optional[Options] = None,
                 out_name: Optional[str] = None,
                 progress: Optional[Callable[[str, float], None]] = None) -> CardResult:
    """Run the routed chain on one card and write the result. Raises on failure."""
    opts = opts or Options()
    t0 = time.perf_counter()
    path = Path(path)
    res = CardResult(file=out_name or path.name, comp_usd=comp_usd, status='processing')
    res.input_sha256 = sha256_file(path)
    note = progress or (lambda stage, frac: None)

    rgb = load_rgb(path)
    h, w = rgb.shape[:2]
    res.input_pixels = f'{w}x{h}'
    note('reading text', 0.02)
    words_in = ocr_words(rgb)
    plan = make_plan(rgb, words_in, comp_usd, opts)
    res.value_tier, res.high_value, res.reasons = plan.value_tier, plan.value_tier == 'high', list(plan.reasons)
    if res.high_value:
        res.flags.append('HIGH_VALUE')

    work = rgb
    if plan.damage != 'off':
        note('damage', 0.05)
        work, dmg, _ = repair_damage(rgb, words_in, plan.damage)
        res.damage_mode, res.damage_coverage = dmg.mode, dmg.coverage
        res.retouched = dmg.regions_applied > 0
        res.reasons.extend(dmg.notes)
        if res.retouched:
            res.flags.append('RETOUCHED')
        if dmg.mode == 'review':
            res.flags.append('DAMAGE_REVIEW')

    if plan.model_key:
        up = get_upscaler(plan.model_key)
        model_input = work
        res.model, res.backend = plan.model_key, up.backend
        res.consistency_iters = plan.consistency_iters
        note('upscaling', 0.1)
        work = up.upscale(work, tile=plan.tile, pad=plan.pad,
                          progress=lambda d, t: note('upscaling', 0.1 + 0.75 * d / t))
        work = back_project(work, model_input, plan.consistency_iters)
        if plan.value_tier == 'high' and work.shape[0] * work.shape[1] > MAX_OUTPUT_PIXELS:
            k = math.sqrt(MAX_OUTPUT_PIXELS / (work.shape[0] * work.shape[1]))
            work = cv2.resize(work, (int(work.shape[1] * k), int(work.shape[0] * k)), interpolation=cv2.INTER_AREA)
    if plan.faces:
        note('faces', 0.86)
        work, n = get_face_restorer().restore(work, opts.face_weight)
        res.faces_restored = n
        if n:
            res.flags.append('FACE_RESTORED')
    if plan.target_long_side:
        work = _fit_long_side(work, plan.target_long_side)

    note('verifying', 0.92)
    fid = fidelity(rgb, work)
    res.psnr_vs_input, res.ssim_vs_input = fid['psnr_vs_input'], fid['ssim_vs_input']
    kept, lost = verify_words(words_in, work, work.shape[1] / w)
    res.text_retention = round(kept, 3)
    if res.psnr_vs_input < FIDELITY_MIN_PSNR:
        res.flags.append('LOW_FIDELITY')
    if 1 - res.text_retention > TEXT_DRIFT_MAX:
        res.flags.append('TEXT_DRIFT')
        res.reasons.append(f'words read in the input but not the output: {lost[:8]}')
    if max(work.shape[:2]) < EBAY_MIN_LONG_SIDE:
        res.flags.append('BELOW_EBAY_MIN')

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(res.file).stem
    ext = '.png' if plan.output_format == 'png' else '.jpg'
    out_path = out_dir / f'{stem}{ext}'
    img = Image.fromarray(work)
    if ext == '.png':
        img.save(out_path, 'PNG', compress_level=6)
    else:
        img.save(out_path, 'JPEG', quality=opts.jpeg_quality, subsampling=0, optimize=True)
    res.output = out_path.name
    res.output_pixels = f'{work.shape[1]}x{work.shape[0]}'
    res.output_sha256 = sha256_file(out_path)
    res.seconds = round(time.perf_counter() - t0, 2)
    res.status = 'done'
    log.info('%s -> %s %s %s in %.1fs flags=%s', res.file, res.output, res.model, res.output_pixels,
             res.seconds, ','.join(res.flags) or '-')
    return res


# --------------------------------------------------------------------------- batches
@dataclasses.dataclass
class CardInput:
    path: Path
    name: str                     # original filename, kept in the output and the ZIP
    comp_usd: Optional[float] = None


MANIFEST_FIELDS = ['file', 'output', 'status', 'error', 'comp_usd', 'value_tier', 'high_value', 'model', 'backend',
                   'consistency_iters', 'input_pixels', 'output_pixels', 'seconds', 'damage_mode', 'damage_coverage', 'retouched',
                   'faces_restored', 'psnr_vs_input', 'ssim_vs_input', 'text_retention', 'flags',
                   'input_sha256', 'output_sha256']


def unique_names(names: Iterable[str]) -> list[str]:
    """Keep original filenames; disambiguate duplicates as 'name (2).jpg'."""
    seen: dict[str, int] = {}
    out = []
    for n in names:
        base = Path(n).name or 'card.jpg'
        key = Path(base).stem.lower()
        seen[key] = seen.get(key, 0) + 1
        out.append(base if seen[key] == 1 else f'{Path(base).stem} ({seen[key]}){Path(base).suffix}')
    return out


def process_batch(cards: list[CardInput], out_dir: Path, opts: Optional[Options] = None,
                  on_update: Optional[Callable[[int, CardResult, str, float], None]] = None,
                  should_stop: Callable[[], bool] = lambda: False) -> list[CardResult]:
    """Process 1..MAX_BATCH cards in order. A failed card is recorded and skipped;
    the batch always continues. Writes manifest.json and manifest.csv."""
    if not 1 <= len(cards) <= MAX_BATCH:
        raise ValueError(f'a batch holds 1 to {MAX_BATCH} cards, got {len(cards)}')
    names = unique_names(c.name for c in cards)
    results = [CardResult(file=n, comp_usd=c.comp_usd) for n, c in zip(names, cards)]
    upd = on_update or (lambda i, r, s, f: None)
    for i, (card, res) in enumerate(zip(cards, results)):
        if should_stop():
            break
        res.status = 'processing'
        upd(i, res, 'starting', 0.0)
        try:
            done = enhance_card(card.path, out_dir, card.comp_usd, opts, out_name=names[i],
                                progress=lambda stage, frac, i=i, res=res: upd(i, res, stage, frac))
            results[i] = done
            upd(i, done, 'done', 1.0)
        except Exception as exc:
            res.status, res.error = 'failed', f'{type(exc).__name__}: {exc}'[:400]
            log.exception('card %s failed', names[i])
            upd(i, res, 'failed', 1.0)
    write_manifest(results, out_dir)
    return results


def write_manifest(results: list[CardResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [dataclasses.asdict(r) for r in results]
    (out_dir / 'manifest.json').write_text(json.dumps({
        'high_value_threshold_usd': HIGH_VALUE_USD, 'models': {k: dataclasses.asdict(v) for k, v in REGISTRY.items()},
        'excluded_models': EXCLUDED_MODELS, 'cards': rows}, indent=2))
    with open(out_dir / 'manifest.csv', 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction='ignore')
        wr.writeheader()
        for r in rows:
            wr.writerow({**r, 'flags': ' '.join(r['flags'])})


def build_zip(results: list[CardResult], out_dir: Path, zip_path: Path) -> Path:
    """ZIP of finished cards under their original filenames (extension follows the
    output format) plus both manifests. Failed cards are listed in the manifest only."""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as z:  # images are already compressed
        for r in results:
            if r.status == 'done' and r.output:
                z.write(out_dir / r.output, r.output)
        for m in ('manifest.csv', 'manifest.json'):
            if (out_dir / m).exists():
                z.write(out_dir / m, m)
    return zip_path


# --------------------------------------------------------------------------- setup
def fetch_models(keys: Optional[Iterable[str]] = None) -> list[str]:
    """Download missing weights from their release URLs and verify SHA-256.
    A file that fails the checksum is deleted, never used. By default only routed
    models are fetched; pass keys to include the benchmark-only ones."""
    import requests
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    done = []
    for key in keys or [k for k, v in REGISTRY.items() if v.task != 'benchmark']:
        spec = REGISTRY[key]
        dest = MODEL_DIR / spec.file
        if dest.is_file() and sha256_file(dest) == spec.sha256:
            continue
        tmp = dest.with_suffix(dest.suffix + '.part')
        log.info('downloading %s (%s)', spec.key, spec.url)
        with requests.get(spec.url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp, 'wb') as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        got = sha256_file(tmp)
        if got != spec.sha256:
            tmp.unlink()
            raise ModelIntegrityError(f'{spec.key}: downloaded file has SHA-256 {got[:12]}..., expected {spec.sha256[:12]}...')
        tmp.replace(dest)
        done.append(key)
    return done


def export_onnx(keys: Optional[Iterable[str]] = None, check: bool = True) -> dict[str, float]:
    """Export the upscalers to ONNX (fixed tile for SwinIR, whose window attention
    bakes shapes in; dynamic H/W for ESRGAN) and verify each graph against PyTorch.
    Returns the max absolute difference per model."""
    import torch
    from spandrel import ModelLoader
    out_dir = MODEL_DIR / 'onnx'
    out_dir.mkdir(parents=True, exist_ok=True)
    diffs = {}
    for key in keys or [k for k, v in REGISTRY.items() if v.onnx and v.task != 'benchmark']:
        spec = REGISTRY[key]
        net = ModelLoader().load_from_file(str(model_path(spec))).model.eval()
        size = spec.onnx_tile or 64
        dynamic = None if spec.onnx_tile else {'input': {2: 'h', 3: 'w'}, 'output': {2: 'H', 3: 'W'}}
        with torch.no_grad():
            torch.onnx.export(net, torch.rand(1, 3, size, size), str(out_dir / spec.onnx), opset_version=17,
                              input_names=['input'], output_names=['output'], dynamic_axes=dynamic, dynamo=False)
        if check:
            import onnxruntime as ort
            x = torch.rand(1, 3, spec.onnx_tile or 96, spec.onnx_tile or 96)
            got = ort.InferenceSession(str(out_dir / spec.onnx), providers=['CPUExecutionProvider']).run(None, {'input': x.numpy()})[0]
            with torch.inference_mode():
                ref = net(x).numpy()
            diffs[key] = float(np.abs(got - ref).max())
            if diffs[key] > 1e-3:
                raise RuntimeError(f'{key}: ONNX output differs from PyTorch by {diffs[key]:.2e}')
        log.info('exported %s -> %s', key, spec.onnx)
    return diffs


# --------------------------------------------------------------------------- CLI
def _parse_comps(pairs: list[str], csv_path: Optional[str]) -> dict[str, float]:
    comps: dict[str, float] = {}
    if csv_path:
        with open(csv_path, newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                comps[Path(row['file']).name] = float(row['comp_usd'])
    for p in pairs:
        name, _, val = p.partition('=')
        comps[Path(name).name] = float(val)
    return comps


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('inputs', nargs='*', help='card images (up to %d)' % MAX_BATCH)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--fetch-models', action='store_true', help='download and verify missing weights, then exit')
    ap.add_argument('--export-onnx', action='store_true', help='export and verify ONNX graphs, then exit')
    ap.add_argument('--model', default='auto', choices=['auto', 'general', 'text'])
    ap.add_argument('--damage', default='off', choices=['off', 'local', 'gemini', 'auto'])
    ap.add_argument('--faces', default='off', choices=['off', 'on'])
    ap.add_argument('--face-weight', type=float, default=0.5)
    ap.add_argument('--comp', action='append', default=[], metavar='FILE=USD', help='comp value per card')
    ap.add_argument('--comp-csv', help='CSV with columns file,comp_usd')
    ap.add_argument('--zip', action='store_true', help='also write cards.zip')
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    if a.fetch_models or a.export_onnx:
        if a.fetch_models:
            print(json.dumps({'downloaded': fetch_models()}))
        if a.export_onnx:
            print(json.dumps({'onnx_max_abs_diff': export_onnx()}))
        return 0
    if not a.inputs or not a.out:
        ap.error('inputs and --out are required')
    comps = _parse_comps(a.comp, a.comp_csv)
    cards = [CardInput(Path(p), Path(p).name, comps.get(Path(p).name)) for p in a.inputs]
    opts = Options(model=a.model, damage=a.damage, faces=a.faces, face_weight=a.face_weight)
    results = process_batch(cards, a.out, opts)
    if a.zip:
        build_zip(results, a.out, a.out / 'cards.zip')
    failed = [r for r in results if r.status != 'done']
    print(json.dumps({'done': len(results) - len(failed), 'failed': len(failed)}))
    return 1 if failed and len(failed) == len(results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
