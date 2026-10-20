#!/usr/bin/env python3
"""Wrestling-card scan enhancement. Originals are never overwritten.

Whole-frame fallback is explicitly reported, not called detection. No learned
restoration is implied by OpenCV resampling. Conservative mode preserves surface
marks and colour; inpainting is an explicit non-conservative preview option.
"""
import contextlib
import json
import math
import os
import sys
import time

import cv2
import numpy as np

MODEL_PATH = os.environ.get('YOLO_MODEL', os.path.join(os.path.dirname(__file__), 'models', 'card_detector.pt'))
_model = None
_model_tried = False
_model_error = 'not_checked'


def get_model():
    global _model, _model_tried, _model_error
    if _model_tried:
        return _model
    _model_tried = True
    if not os.path.isfile(MODEL_PATH):
        _model_error = 'weights_missing'
        return None
    try:
        # Ultralytics may print initialization messages; stdout is JSON-only.
        with contextlib.redirect_stdout(sys.stderr):
            from ultralytics import YOLO
            _model = YOLO(MODEL_PATH)
            _model.predict(np.zeros((320, 320, 3), dtype=np.uint8), device='cpu', verbose=False)
        _model_error = None
    except Exception as exc:
        _model = None
        _model_error = 'model_load_or_inference_failed'
        print(f'Detector unavailable: {exc}', file=sys.stderr)
    return _model


def detector_status():
    return {'available': get_model() is not None, 'reason': _model_error, 'path': MODEL_PATH}


def detect_card(img):
    model = get_model()
    if model is None:
        return None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            boxes = model(img, conf=0.55, device='cpu', verbose=False)[0].boxes
        h, w = img.shape[:2]
        accepted = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2, conf)):
                continue
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if conf >= 0.55 and x2-x1 >= w*0.70 and y2-y1 >= h*0.70:
                accepted.append((x1, y1, x2, y2, conf))
        return max(accepted, key=lambda b: b[4], default=None)
    except Exception as exc:
        print(f'Detection failed: {exc}', file=sys.stderr)
        return None


def _readability(img):
    try:
        import pytesseract
        h, w = img.shape[:2]
        k = min(1.0, 700.0 / max(h, w))
        small = cv2.resize(img, (max(1, round(w*k)), max(1, round(h*k))), interpolation=cv2.INTER_AREA)
        data = pytesseract.image_to_data(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
                                        config='--psm 6', output_type=pytesseract.Output.DICT, timeout=10)
        return sum(float(conf) for word, conf in zip(data['text'], data['conf'])
                   if float(conf) > 55 and len(word.strip()) >= 3 and any(c.isalpha() for c in word))
    except Exception:
        return 0


def best_orientation(img):
    variants = [(img, 'none'), (cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 'cw'),
                (cv2.rotate(img, cv2.ROTATE_180), '180'),
                (cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 'ccw')]
    scores = [_readability(v) for v, _ in variants]
    best = max(scores)
    # No readable evidence means preserve the scan, including landscape designs.
    if best < 200 or scores[0] >= best * 0.85:
        return variants[0]
    return variants[scores.index(best)]


def adaptive_descratch(img, strength):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    energy_map = np.abs(cv2.Laplacian(gray, cv2.CV_64F, ksize=3))
    threshold = max(25.0, float(np.percentile(energy_map, 92)) * (4.0 - 1.5*strength))
    mask = (energy_map > threshold).astype(np.uint8)*255
    lines = cv2.bitwise_or(
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))),
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lines, 8)
    keep = np.zeros_like(lines)
    candidates = 0
    for i in range(1, n):
        w, h, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if area >= 30 and max(w, h) / max(1, min(w, h)) >= 6:
            keep[labels == i] = 255
            candidates += 1
    if not candidates:
        return img, 0
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    repaired = cv2.inpaint(img, keep, 4, cv2.INPAINT_NS)
    return cv2.addWeighted(img, 1-strength, repaired, strength, 0), candidates


def validate_options(opts):
    if not isinstance(opts, dict):
        raise ValueError('options must be an object')
    out = {}
    for key, default, lo, hi in [('scale', 2, 1, 4), ('descratch', 0, 0, 1),
                               ('denoise', 0, 0, 1), ('sharpen', 0.25, 0, 1), ('contrast', 0, 0, 0.5)]:
        value = opts.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'{key} must be between {lo} and {hi}')
        out[key] = value
    for key in ['autoRotate', 'conservative', 'autoCrop']:
        value = opts.get(key, True)
        if not isinstance(value, bool):
            raise ValueError(f'{key} must be a boolean')
        out[key] = value
    rotation = opts.get('rotation', 0)
    if isinstance(rotation, bool) or rotation not in [0, 90, 180, 270]:
        raise ValueError('invalid rotation')
    out['rotation'] = rotation
    quad = opts.get('quad')
    if quad is not None:
        if not isinstance(quad, list) or len(quad) != 4 or any(
            not isinstance(p, list) or len(p) != 2 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
                for v in p) for p in quad):
            raise ValueError('invalid crop corners')
        pts = np.array(quad, dtype=np.float32)
        edges = np.roll(pts, -1, axis=0) - pts
        cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
        area = sum(p[0]*quad[(i+1)%4][1]-quad[(i+1)%4][0]*p[1] for i,p in enumerate(quad))/2
        if np.any(cross <= 0) or area < .005:
            raise ValueError('crop must be clockwise, convex and nonzero')
    out['quad'] = quad
    return out


def manual_geometry(img, rotation, quad):
    if rotation:
        img = cv2.rotate(img, {90:cv2.ROTATE_90_CLOCKWISE, 180:cv2.ROTATE_180, 270:cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation])
    if quad is None:
        return img
    h,w = img.shape[:2]
    src = np.array(quad, np.float32) * np.array([w-1,h-1], np.float32)
    tl,tr,br,bl = src
    out_w = max(2, round(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl)))+1)
    out_h = max(2, round(max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr)))+1)
    dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src,dst), (out_w,out_h))


def enhance(input_path, output_path, opts=None):
    started = time.perf_counter()
    opts = validate_options({} if opts is None else opts)
    if os.path.realpath(input_path) == os.path.realpath(output_path):
        raise ValueError('output must not overwrite the original')
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError('cannot read image')
    src_h, src_w = img.shape[:2]
    img = manual_geometry(img, opts['rotation'], opts['quad'])
    det = detect_card(img) if opts['quad'] is None and opts['autoCrop'] else None
    if det:
        x1, y1, x2, y2, _ = det
        img = img[max(0, y1-4):min(src_h, y2+4), max(0, x1-4):min(src_w, x2+4)]
    rot = 'none'
    if opts['autoRotate'] and not opts['rotation'] and opts['quad'] is None:
        img, rot = best_orientation(img)
    h, w = img.shape[:2]
    factor = min(opts['scale'], 2200 / max(h, w))
    size = (max(1, round(w*factor)), max(1, round(h*factor)))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LANCZOS4)
    sharpness_before = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    candidates = 0
    # Surface repair/colour changes require explicit non-conservative mode.
    if not opts['conservative']:
        if opts['descratch'] > 0:
            img, candidates = adaptive_descratch(img, opts['descratch'])
        if opts['denoise'] > 0:
            img = cv2.addWeighted(img, 1-opts['denoise'], cv2.GaussianBlur(img, (3, 3), 0), opts['denoise'], 0)
        if opts['contrast'] > 0:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
            l = cv2.addWeighted(l, 1-opts['contrast'], equalized, opts['contrast'], 0)
            img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    sharpen = min(opts['sharpen'], 0.3) if opts['conservative'] else opts['sharpen']
    if sharpen > 0:
        f = img.astype(np.float32)
        img = np.clip(f + (f-cv2.GaussianBlur(f, (3, 3), 1.0))*sharpen*1.7
                      + (f-cv2.GaussianBlur(f, (5, 5), 2.0))*sharpen*0.8, 0, 255).astype(np.uint8)
    # Commit a complete image atomically; a failure cannot advertise a stale file.
    temporary = output_path + '.tmp.png'
    try:
        if not cv2.imwrite(temporary, img, [cv2.IMWRITE_PNG_COMPRESSION, 4]):
            raise OSError('cannot write enhanced image')
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    status = detector_status()
    sharpness_after = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    return {'category': 'wrestling', 'rotation': rot, 'manual_rotation':opts['rotation'], 'source': f'{src_w}x{src_h}',
            'output_size': f'{size[0]}x{size[1]}', 'orientation': 'landscape' if size[0] > size[1] else 'portrait',
            'yolo_available': status['available'], 'yolo_conf': round(det[4], 3) if det else None,
            'yolo_cropped': bool(det), 'detection_mode': 'manual_quad' if opts['quad'] is not None else ('yolo_crop' if det else 'whole_frame'),
            'detection_warning': None if det or opts['quad'] is not None or not opts['autoCrop'] else (status['reason'] or 'no_accepted_detection'),
            'scratch_candidates': candidates, 'surface_altered': candidates > 0,
            'conservative': opts['conservative'], 'review_needed': True, 'options':opts,
            'backend':'opencv', 'ms':round((time.perf_counter()-started)*1000),
            'sharpness_before':sharpness_before, 'sharpness_after':sharpness_after,
            'sharpness_delta':sharpness_after-sharpness_before}


def main():
    try:
        if sys.argv[1:] == ['--capabilities']:
            result = detector_status()
        elif len(sys.argv) in (3, 4):
            result = enhance(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) == 4 else {})
        else:
            raise ValueError('usage: enhance_worker.py <input> <output> [opts_json]')
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({'error': str(exc)}))
        return 1


if __name__ == '__main__':
    sys.exit(main())
