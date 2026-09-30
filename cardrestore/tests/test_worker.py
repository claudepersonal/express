"""Worker checks against the real Real-ESRGAN weights (skipped when absent).

Run: RESTORE_MODEL=/path/RealESRGAN_x4plus.pth python3 -m unittest tests/test_worker.py
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'worker'))
import restore_worker as rw  # noqa: E402


@unittest.skipUnless(os.path.isfile(rw.MODEL_PATH), 'Real-ESRGAN weights not present')
class TiledInference(unittest.TestCase):
    def test_tiled_matches_single_pass(self):
        rng = np.random.default_rng(0)
        base = rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)
        img = np.kron(base, np.ones((6, 6, 1), dtype=np.uint8))  # 144x192, blocky like a low-res scan
        single = rw.upscale_x4(img, tile=512)          # one tile covers the whole image
        tiled = rw.upscale_x4(img, tile=64)            # 3x3 tiles with padding
        self.assertEqual(single.shape, (576, 768, 3))
        diff = np.abs(single.astype(int) - tiled.astype(int))
        # Random noise is the worst case for seams; real scans are smoother.
        self.assertLessEqual(int(diff.max()), 12, f'max diff {diff.max()}')
        self.assertLess(float(diff.mean()), 0.1)


class InputLoading(unittest.TestCase):
    """load_input: EXIF orientation and the shrink-to-fit rule (no model needed)."""

    def _save(self, size, orientation=None):
        import tempfile
        from PIL import Image
        path = os.path.join(tempfile.mkdtemp(), 'card.jpg')
        img = Image.new('RGB', size, (200, 30, 30))
        if orientation:
            exif = Image.Exif()
            exif[0x0112] = orientation
            img.save(path, exif=exif)
        else:
            img.save(path)
        return path

    def test_small_input_is_untouched(self):
        arr, fitted = rw.load_input(self._save((300, 420)), fit=True)
        self.assertEqual(arr.shape, (420, 300, 3))
        self.assertIsNone(fitted)

    def test_exif_rotation_is_applied(self):
        # Orientation 6 = stored landscape, displayed portrait.
        arr, _ = rw.load_input(self._save((420, 300), orientation=6), fit=True)
        self.assertEqual(arr.shape, (420, 300, 3))

    def test_large_input_shrinks_to_the_limit_keeping_aspect(self):
        w, h = 3000, 4000
        arr, fitted = rw.load_input(self._save((w, h)), fit=True)
        oh, ow = arr.shape[:2]
        self.assertEqual(fitted, f'{w}x{h}')
        self.assertLessEqual(ow * oh, rw.MAX_INPUT_PIXELS)
        self.assertGreater(ow * oh, rw.MAX_INPUT_PIXELS * 0.99)
        self.assertAlmostEqual(ow / oh, w / h, places=2)

    def test_large_input_is_refused_without_fit(self):
        with self.assertRaises(ValueError):
            rw.load_input(self._save((3000, 4000)), fit=False)


if __name__ == '__main__':
    unittest.main()
