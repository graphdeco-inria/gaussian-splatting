import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np


path = Path(__file__).resolve().parents[1] / "scene" / "colmap_loader.py"
spec = importlib.util.spec_from_file_location("colmap_loader", path)
colmap_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colmap_loader)


class TextCameraTest(unittest.TestCase):
    def test_simple_pinhole_text_matches_binary_export(self):
        with tempfile.TemporaryDirectory() as directory:
            text = Path(directory) / "cameras.txt"
            binary = Path(directory) / "cameras.bin"
            text.write_text("# Camera list\n7 SIMPLE_PINHOLE 640 480 500 320 240\n")
            binary.write_bytes(struct.pack("<QiiQQddd", 1, 7, 0, 640, 480, 500, 320, 240))
            actual = colmap_loader.read_intrinsics_text(text)[7]
            expected = colmap_loader.read_intrinsics_binary(binary)[7]
            self.assertEqual(actual[:4], expected[:4])
            np.testing.assert_array_equal(actual.params, expected.params)

    def test_mixed_undistorted_camera_models(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.txt"
            path.write_text("# Two different intrinsic models\n\n7 SIMPLE_PINHOLE 640 480 500 320 240\n9 PINHOLE 800 600 700 710 400 300\n")
            cameras = colmap_loader.read_intrinsics_text(path)
            self.assertEqual(set(cameras), {7, 9})
            self.assertEqual(cameras[7].model, "SIMPLE_PINHOLE")
            np.testing.assert_array_equal(cameras[9].params, [700, 710, 400, 300])

    def test_distorted_camera_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.txt"
            path.write_text("1 SIMPLE_RADIAL 640 480 500 320 240 0.1\n")
            with self.assertRaises(AssertionError):
                colmap_loader.read_intrinsics_text(path)


if __name__ == "__main__":
    unittest.main()
