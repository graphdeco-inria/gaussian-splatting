import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import cv2
import numpy as np

utils_path = Path(__file__).resolve().parents[1] / "utils"
sys.path.insert(0, str(utils_path))
try:
    spec = importlib.util.spec_from_file_location("native_depth_scale", utils_path / "make_depth_scale.py")
    depth_scale = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(depth_scale)
finally:
    sys.path.remove(str(utils_path))


class DepthCalibrationTest(unittest.TestCase):
    def make_scene(self, directory, constant=False, invalid=False):
        mono = (4000 + 600 * np.arange(80)).reshape(8, 10).astype(np.uint16)
        if constant:
            mono[:] = 30000
        path = Path(directory) / "frame.png"
        self.assertTrue(cv2.imwrite(str(path), mono))
        pixels = np.array([(x, y) for y in range(1, 5) for x in range(1, 6)])
        xys = pixels * np.array([4., 2.])
        inv_depth = (4000 + 600 * (pixels[:, 1] * 10 + pixels[:, 0])) / 65536. * 2. + .1
        xyz = np.zeros((len(pixels), 3))
        xyz[:, 2] = 1. / inv_depth
        ids = np.arange(len(pixels))
        if invalid:
            xyz = np.concatenate([xyz, [[0, 0, 0], [0, 0, np.nan], [0, 0, -1], [0, 0, np.inf]]])
            ids = np.arange(len(xyz))
            xys = np.concatenate([xys, [[0, 0]] * 4])
        camera = SimpleNamespace(width=40, height=16)
        image = SimpleNamespace(camera_id=1, point3D_ids=ids, xys=xys, qvec=np.array([1., 0, 0, 0]),
                                tvec=np.zeros(3), name="frame.jpg")
        images = {3: image}
        # A CLI invocation creates this global; retain it while comparing original behavior.
        depth_scale.images_metas = images
        return {1: camera}, images, xyz, SimpleNamespace(depths_dir=directory)

    def test_independent_xy_scales_recover_known_affine_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_scene(directory)
            result = depth_scale.get_scales(3, *args)
            self.assertAlmostEqual(result["scale"], 2., places=5)
            self.assertAlmostEqual(result["offset"], .1, places=5)

    def test_constant_monocular_depth_returns_disabled_finite_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_scene(directory, constant=True)
            result = depth_scale.get_scales(3, *args)
            self.assertEqual(result["scale"], 0)
            self.assertEqual(result["offset"], 0)
            self.assertTrue(np.isfinite([result["scale"], result["offset"]]).all())

    def test_zero_nan_negative_and_infinite_colmap_depths_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_scene(directory, invalid=True)
            with np.errstate(divide="raise", invalid="raise"):
                result = depth_scale.get_scales(3, *args)
            self.assertAlmostEqual(result["scale"], 2., places=5)
            self.assertAlmostEqual(result["offset"], .1, places=5)


if __name__ == "__main__":
    unittest.main()
