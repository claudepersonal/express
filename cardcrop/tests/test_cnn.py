import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import card_pipeline as pipeline


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional torch dependency is not installed')
class CnnTests(unittest.TestCase):
    def test_tiled_inference_matches_whole_image(self):
        import torch
        from chrome_model import ChromeArtifactCNN
        torch.set_num_threads(1)
        torch.manual_seed(4)
        model = ChromeArtifactCNN(num_features=4, num_blocks=1).eval()
        image = np.random.default_rng(5).integers(0, 256, (45, 280, 3), dtype=np.uint8)
        actual = pipeline.cleanup_cnn(image, model, 'cpu')
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255
        with torch.inference_mode():
            pixels = (model(tensor).clamp(0, 1)[0].numpy().transpose(1, 2, 0)*255).round().astype(np.uint8)
        expected = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        self.assertLessEqual(np.abs(actual.astype(int)-expected.astype(int)).max(), 1)

    def test_shared_checkpoint_loads_for_inference(self):
        import torch
        from chrome_model import ChromeArtifactCNN
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pth'
            model = ChromeArtifactCNN()
            torch.save(model.state_dict(), path)
            loaded = pipeline._load_cnn(str(path), 'cpu', path.stat().st_mtime_ns)
            self.assertFalse(loaded.training)
            self.assertEqual(set(loaded.state_dict()), set(model.state_dict()))


if __name__ == '__main__':
    unittest.main()
