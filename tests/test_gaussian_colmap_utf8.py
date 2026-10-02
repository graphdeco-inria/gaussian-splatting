import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]


def native_module(name, path):
    spec = importlib.util.spec_from_file_location(name, root / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


loader = native_module("native_colmap_loader", "scene/colmap_loader.py")
model_io = native_module("native_model_io", "utils/read_write_model.py")


class ColmapUtf8Test(unittest.TestCase):
    def test_training_loader_and_utility_read_independent_utf8_fixture(self):
        name = "场景/été_🚀.jpg"
        binary = struct.pack("<Qidddddddi", 1, 7, 1, 0, 0, 0, 2, 3, 4, 9)
        binary += name.encode("utf-8") + b"\x00" + struct.pack("<Qddq", 1, 1.25, 2.5, -1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.bin"
            path.write_bytes(binary)
            for read in [loader.read_extrinsics_binary, model_io.read_images_binary]:
                image = read(path)[7]
                self.assertEqual(image.name, name)
                np.testing.assert_array_equal(image.xys, [[1.25, 2.5]])
                np.testing.assert_array_equal(image.point3D_ids, [-1])

    def test_writer_uses_complete_utf8_payload_and_both_readers_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.bin"
            name = "模型/naïve.png"
            image = model_io.Image(7, np.array([1., 0, 0, 0]), np.array([2., 3, 4]), 9,
                                  name, np.empty((0, 2)), np.empty(0, dtype=np.int64))
            model_io.write_images_binary({7: image}, path)
            expected = struct.pack("<Qidddddddi", 1, 7, 1, 0, 0, 0, 2, 3, 4, 9)
            expected += name.encode("utf-8") + b"\x00" + struct.pack("<Q", 0)
            self.assertEqual(path.read_bytes(), expected)
            for read in [loader.read_extrinsics_binary, model_io.read_images_binary]:
                self.assertEqual(read(path)[7].name, name)

    def test_ascii_multi_record_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.bin"
            images = {
                i: model_io.Image(i, np.array([1., 0, 0, 0]), np.zeros(3), 1, name,
                                  np.array([[i, i + .5]]), np.array([-1], dtype=np.int64))
                for i, name in enumerate(["one.jpg", "two/sub.png"], 1)
            }
            model_io.write_images_binary(images, path)
            for read in [loader.read_extrinsics_binary, model_io.read_images_binary]:
                recovered = read(path)
                self.assertEqual([x.name for x in recovered.values()], [x.name for x in images.values()])
                for i in images:
                    np.testing.assert_array_equal(recovered[i].xys, images[i].xys)


if __name__ == "__main__":
    unittest.main()
