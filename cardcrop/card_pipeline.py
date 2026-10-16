"""Card perspective rectification and opt-in cleanup profiles.

Adapted from the supplied wrestling_full_pipeline.py. OpenCV is sufficient for
classical processing. Torch and operator-provided weights are optional for CNN.
"""
import contextlib
from functools import lru_cache
import os
from pathlib import Path
import sys
from typing import Optional, Tuple

import cv2
import numpy as np


def order_corners(pts):
    points = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(points).all() or len(np.unique(points, axis=0)) != 4:
        raise ValueError('four distinct finite corners required')
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    points = np.roll(points, -np.argmin(points.sum(axis=1)), axis=0)
    if not cv2.isContourConvex(points.reshape(-1, 1, 2)) or cv2.contourArea(points) < 4:
        raise ValueError('degenerate card quadrilateral')
    return points


def detect_card_quad(img_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    h, w = img_bgr.shape[:2]
    img_area = float(h * w)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    attempts = []
    edges = cv2.Canny(gray, 40, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    attempts.append(edges)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 25, 10)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=3)
    attempts.append(thr)

    best = None
    for mask in attempts:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.08 * img_area or area > 0.98 * img_area:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps in (0.015, 0.02, 0.03, 0.045):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    if best is None or area > best[0]:
                        best = (area, approx)
                    break

    if best is None:
        return None, 0.0

    corners = order_corners(best[1])
    confidence = float(min(1.0, best[0] / img_area * 1.15))
    return corners, confidence


def perspective_crop(img_bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    width_top = np.linalg.norm(corners[1] - corners[0])
    width_bot = np.linalg.norm(corners[2] - corners[3])
    height_l = np.linalg.norm(corners[3] - corners[0])
    height_r = np.linalg.norm(corners[2] - corners[1])
    out_w = max(2, round(max(width_top, width_bot)))
    out_h = max(2, round(max(height_l, height_r)))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img_bgr, M, (out_w, out_h))


def cleanup_classical(img_bgr: np.ndarray, strength: float = 0.45) -> np.ndarray:
    """Uploaded classical profile: inpaint, denoise, sharpen, CLAHE, brightness."""
    # BLACKHAT finds dark narrow marks; this preview can alter printed detail.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 21))
    bh_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_h)
    bh_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_v)
    _, mask_h = cv2.threshold(bh_h, 12, 255, cv2.THRESH_BINARY)
    _, mask_v = cv2.threshold(bh_v, 12, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(mask_h, mask_v)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    inpainted = cv2.inpaint(img_bgr, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
    result = cv2.addWeighted(img_bgr, 1.0 - strength, inpainted, strength, 0)

    # Light denoise (protect foil edges)
    result = cv2.fastNlMeansDenoisingColored(result, None, h=2, hColor=2,
                                              templateWindowSize=7, searchWindowSize=15)

    # Mild unsharp mask for text and detail
    blur = cv2.GaussianBlur(result, (0, 0), 1.2)
    result = cv2.addWeighted(result, 1.25, blur, -0.25, 0)

    # Gentle contrast via CLAHE on luminance (LAB)
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Uniform gain/offset brightens all channels; it is not white balance.
    result = cv2.convertScaleAbs(result, alpha=1.02, beta=2)

    return result, int(cv2.countNonZero(mask)) if strength > 0 else 0



@lru_cache(maxsize=2)
def _load_cnn(path, device, modified_ns):
    import torch
    from chrome_model import ChromeArtifactCNN
    with contextlib.redirect_stdout(sys.stderr):
        model = ChromeArtifactCNN()
        state = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()
    return model


def cleanup_cnn(img_bgr, model, device):
    """Bound memory with 256px tiles and a 40px receptive-field halo."""
    import torch
    h, w = img_bgr.shape[:2]
    result = np.empty_like(img_bgr)
    with torch.inference_mode():
        for y in range(0, h, 256):
            for x in range(0, w, 256):
                y2, x2 = min(y + 256, h), min(x + 256, w)
                top, left = max(0, y - 40), max(0, x - 40)
                bottom, right = min(h, y2 + 40), min(w, x2 + 40)
                rgb = cv2.cvtColor(img_bgr[top:bottom, left:right], cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255.0
                out = model(tensor)
                if out.shape != tensor.shape or not torch.isfinite(out).all().item():
                    raise ValueError('invalid CNN output')
                pixels = (out.clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
                bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                result[y:y2, x:x2] = bgr[y-top:y2-top, x-left:x2-left]
    return result


def run_cleanup(img, profile, strength):
    warnings = []
    if profile == 'cnn':
        path = os.environ.get('CHROME_CNN_WEIGHTS', '')
        if not path or not Path(path).is_file():
            warnings.append('cnn_weights_missing: used classical cleanup')
        else:
            try:
                import torch
                device = os.environ.get('CHROME_CNN_DEVICE', 'auto')
                if device == 'auto':
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                if device not in ('cpu', 'cuda'):
                    raise ValueError('CHROME_CNN_DEVICE must be auto, cpu or cuda')
                model = _load_cnn(path, device, Path(path).stat().st_mtime_ns)
                output = cleanup_cnn(img, model, device)
                return output, {'cleanup_engine': 'cnn', 'cleanup_device': device,
                                'cleanup_warnings': [], 'surface_altered': True,
                                'inpaint_pixels': 0}
            except Exception as exc:
                print(f'CNN unavailable: {exc}', file=sys.stderr)
                warnings.append('cnn_load_or_inference_failed: used classical cleanup')
    output, pixels = cleanup_classical(img, strength)
    return output, {'cleanup_engine': 'classical', 'cleanup_device': 'cpu',
                    'cleanup_warnings': warnings, 'surface_altered': pixels > 0,
                    'inpaint_pixels': pixels}
