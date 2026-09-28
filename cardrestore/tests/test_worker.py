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


if __name__ == '__main__':
    unittest.main()
