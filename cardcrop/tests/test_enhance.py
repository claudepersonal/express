import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('worker', ROOT/'enhance_worker.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)

class EnhancementTests(unittest.TestCase):
    def test_weak_ocr_keeps_landscape(self):
        img = np.zeros((100, 200, 3), np.uint8)
        with patch.object(w, '_readability', return_value=0):
            actual, rotation = w.best_orientation(img)
        self.assertEqual(rotation, 'none')
        self.assertEqual(actual.shape, img.shape)

    def test_upside_down_is_considered(self):
        with patch.object(w, '_readability', side_effect=[20, 30, 500, 40]):
            _, rotation = w.best_orientation(np.zeros((200, 100, 3), np.uint8))
        self.assertEqual(rotation, '180')

    def test_decimal_ocr_confidence(self):
        with patch('pytesseract.image_to_data', return_value={'text':['WRESTLING'], 'conf':['91.75']}):
            self.assertEqual(w._readability(np.zeros((100, 100, 3), np.uint8)), 91.75)

    def test_aspect_ratio_and_scale(self):
        with tempfile.TemporaryDirectory() as d, patch.object(w, 'detect_card', return_value=None), patch.object(w, 'get_model', return_value=None):
            src, dst = str(Path(d)/'in.png'), str(Path(d)/'out.png')
            cv2.imwrite(src, np.zeros((100, 300, 3), np.uint8))
            result = w.enhance(src, dst, {'autoRotate':False, 'scale':2, 'sharpen':0})
            self.assertEqual(cv2.imread(dst).shape, (200, 600, 3))
            self.assertEqual(result['detection_mode'], 'whole_frame')
            self.assertTrue(result['review_needed'])

    def test_conservative_does_not_inpaint_or_change_colour(self):
        with tempfile.TemporaryDirectory() as d, patch.object(w, 'detect_card', return_value=None), patch.object(w, 'get_model', return_value=None), patch.object(w, 'adaptive_descratch', side_effect=AssertionError('inpainting called')):
            src, dst = str(Path(d)/'in.png'), str(Path(d)/'out.png')
            image=np.random.default_rng(4).integers(0, 256, (100, 70, 3), dtype=np.uint8)
            cv2.imwrite(src, image)
            w.enhance(src, dst, {'autoRotate':False, 'scale':1, 'sharpen':0, 'descratch':1, 'denoise':1, 'contrast':0.5})
            np.testing.assert_array_equal(image, cv2.imread(dst))

    def test_bad_options_fail(self):
        for opts in [[], {'scale':float('nan')}, {'sharpen':-1}, {'contrast':2}, {'autoRotate':'false'}]:
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                w.validate_options(opts)

    def test_manual_rotation_then_full_frame_uses_rotated_coordinates(self):
        image=np.arange(4*6*3,dtype=np.uint8).reshape(4,6,3)
        quad=[[0,0],[1,0],[1,1],[0,1]]
        actual=w.manual_geometry(image,90,quad)
        np.testing.assert_array_equal(actual,cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE))

    def test_manual_crop_bypasses_detector_and_is_reported(self):
        with tempfile.TemporaryDirectory() as d, patch.object(w,'detect_card',side_effect=AssertionError('detector overrides manual crop')), patch.object(w,'get_model',return_value=None):
            src,dst=str(Path(d)/'in.png'),str(Path(d)/'out.png')
            cv2.imwrite(src,np.zeros((100,200,3),np.uint8))
            quad=[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]
            result=w.enhance(src,dst,{'quad':quad,'rotation':90,'autoRotate':False,'scale':1,'sharpen':0})
            self.assertEqual(result['detection_mode'],'manual_quad')
            self.assertEqual(result['manual_rotation'],90)
            self.assertEqual(result['options']['quad'],quad)
            self.assertEqual(result['backend'],'opencv')

    def test_rejects_invalid_manual_geometry(self):
        for opts in [{'rotation':45},{'rotation':True},{'quad':[[0,0],[1,1],[1,0],[0,1]]}]:
            with self.subTest(opts=opts),self.assertRaises(ValueError): w.validate_options(opts)

    def test_failed_output_write_is_not_success(self):
        with tempfile.TemporaryDirectory() as d, patch.object(w, 'detect_card', return_value=None):
            src = str(Path(d)/'in.png')
            cv2.imwrite(src, np.zeros((100, 70, 3), np.uint8))
            with self.assertRaises(OSError):
                w.enhance(src, str(Path(d)/'missing'/'out.png'), {'autoRotate':False})

    def test_cli_error_nonzero_and_json(self):
        for args in [[], ['missing.png','out.png'], ['x','y','{broken']]:
            r=subprocess.run([sys.executable,str(ROOT/'enhance_worker.py'),*args],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)
            self.assertIn('error',json.loads(r.stdout))

if __name__ == '__main__':
    unittest.main()
