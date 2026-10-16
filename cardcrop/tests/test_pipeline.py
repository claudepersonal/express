import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import card_pipeline as pipeline
import enhance_worker as worker


class PipelineTests(unittest.TestCase):
    def test_landscape_rectification_preserves_geometry(self):
        corners = np.array([[10, 10], [210, 10], [210, 110], [10, 110]], np.float32)
        image = np.full((130, 230, 3), 170, np.uint8)
        ordered = pipeline.order_corners(corners[[2, 0, 3, 1]])
        np.testing.assert_array_equal(ordered, corners)
        self.assertEqual(pipeline.perspective_crop(image, ordered).shape, (100, 200, 3))

    def test_degenerate_corners_rejected(self):
        with self.assertRaises(ValueError):
            pipeline.order_corners(np.zeros((4, 2)))

    def test_blank_image_is_not_a_detection(self):
        corners, confidence = pipeline.detect_card_quad(np.zeros((120, 180, 3), np.uint8))
        self.assertIsNone(corners)
        self.assertEqual(confidence, 0)

    def test_actual_card_outline_is_detected(self):
        image = np.zeros((320, 400, 3), np.uint8)
        cv2.rectangle(image, (40, 35), (360, 285), (230, 230, 230), -1)
        corners, confidence = pipeline.detect_card_quad(image)
        self.assertIsNotNone(corners)
        self.assertGreater(confidence, 0.35)

    def test_missing_weights_fall_back_with_identical_classical_output(self):
        image = np.random.default_rng(3).integers(50, 200, (48, 64, 3), dtype=np.uint8)
        original = image.copy()
        expected, _ = pipeline.cleanup_classical(image, 0.45)
        with patch.dict(os.environ, {'CHROME_CNN_WEIGHTS': ''}):
            actual, meta = pipeline.run_cleanup(image, 'cnn', 0.45)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(image, original)
        self.assertEqual(meta['cleanup_engine'], 'classical')
        self.assertTrue(meta['cleanup_warnings'])

    def test_invalid_profile_options_and_conservative_conflict(self):
        for opts in [{'cleanup': 'unknown'}, {'detection': []}, {'cleanupStrength': True},
                     {'cleanupStrength': 2}, {'cleanup': 'cnn'}, {'cleanup': 'classical'}]:
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                worker.validate_options(opts)

    def test_worker_full_profile_does_not_add_manual_sharpen(self):
        image = np.random.default_rng(7).integers(40, 210, (48, 64, 3), dtype=np.uint8)
        expected, _ = pipeline.cleanup_classical(image, 0.45)
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory)/'in.png', Path(directory)/'out.png'
            cv2.imwrite(str(source), image)
            with patch.object(worker, 'detector_status', return_value={'available': False, 'reason': 'test'}):
                result = worker.enhance(str(source), str(target), {
                    'cleanup': 'classical', 'conservative': False, 'detection': 'none',
                    'scale': 1, 'autoRotate': False, 'sharpen': 1,
                })
            np.testing.assert_array_equal(cv2.imread(str(target)), expected)
            np.testing.assert_array_equal(cv2.imread(str(source)), image)
            self.assertEqual(result['cleanup_engine'], 'classical')
            self.assertEqual(result['clahe']['clip_limit'], 1.5)

    def test_batch_exports_only_successful_images_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/'input', Path(directory)/'output'
            source.mkdir()
            cv2.imwrite(str(source/'card.png'), np.full((40, 50, 3), 100, np.uint8))
            (source/'broken.png').write_text('not an image')
            run = subprocess.run([sys.executable, str(ROOT/'batch_pipeline.py'),
                '--input_dir', str(source), '--output_dir', str(output),
                '--detection', 'none', '--no-auto-rotate', '--zip'], capture_output=True, text=True,
                env={**os.environ, 'YOLO_MODEL': str(source/'missing.pt')})
            self.assertEqual(run.returncode, 1, run.stderr)
            with zipfile.ZipFile(output/'enhanced.zip') as archive:
                self.assertEqual(set(archive.namelist()), {'card.png.enhanced.png', 'manifest.json'})


if __name__ == '__main__':
    unittest.main()
