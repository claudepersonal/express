"""Regression coverage for full-strength luminance CLAHE."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location(
    'clahe_worker', Path(__file__).resolve().parents[1] / 'enhance_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class ClaheTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(19).integers(
            60, 180, (64, 96, 3), dtype=np.uint8)

    def enhance(self, contrast, conservative=False):
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory) / 'source.png')
            output = str(Path(directory) / 'output.png')
            self.assertTrue(cv2.imwrite(source, self.image))
            with patch.object(worker, 'detect_card', return_value=None), patch.object(
                worker, 'detector_status',
                return_value={'available': False, 'reason': 'test'}
            ):
                worker.enhance(source, output, {
                    'scale': 1, 'autoRotate': False, 'sharpen': 0,
                    'denoise': 0, 'descratch': 0, 'contrast': contrast,
                    'conservative': conservative,
                })
            np.testing.assert_array_equal(cv2.imread(source), self.image)
            return cv2.imread(output)

    def test_positive_contrast_applies_full_luminance_clahe(self):
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        expected = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
        self.assertFalse(np.array_equal(expected, self.image))
        for contrast in (0.08, 0.5):
            with self.subTest(contrast=contrast):
                np.testing.assert_array_equal(self.enhance(contrast), expected)

    def test_zero_contrast_bypasses_clahe(self):
        with patch.object(cv2, 'createCLAHE', side_effect=AssertionError('unexpected CLAHE')):
            np.testing.assert_array_equal(self.enhance(0), self.image)

    def test_conservative_mode_bypasses_clahe(self):
        with patch.object(cv2, 'createCLAHE', side_effect=AssertionError('unexpected CLAHE')):
            np.testing.assert_array_equal(self.enhance(0.5, conservative=True), self.image)


if __name__ == '__main__':
    unittest.main()
