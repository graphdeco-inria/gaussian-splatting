import copy
import importlib.util
import inspect
import io
from pathlib import Path
from types import ModuleType
import unittest
from unittest import mock

import torch

from arguments import OptimizationParams
from argparse import ArgumentParser


def load_gaussian_model():
    # Checkpointing does not call the CUDA nearest-neighbour extension.
    extension = ModuleType("simple_knn._C")
    extension.distCUDA2 = mock.Mock(side_effect=AssertionError("unexpected CUDA call"))
    path = Path(__file__).resolve().parents[1] / "scene" / "gaussian_model.py"
    spec = importlib.util.spec_from_file_location("checkpoint_gaussian_model", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict("sys.modules", {"simple_knn._C": extension}):
        spec.loader.exec_module(module)
    return module.GaussianModel


GaussianModel = load_gaussian_model()


class CheckpointExposureTest(unittest.TestCase):
    def setUp(self):
        self.args = OptimizationParams(ArgumentParser())
        self.zeros = torch.zeros

    def cpu_zeros(self, *args, **kwargs):
        # Only redirect the two CUDA-only training-statistic allocations.
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return self.zeros(*args, **kwargs)

    def make_model(self):
        model = GaussianModel(0)
        model._xyz = torch.nn.Parameter(torch.tensor([[0.1, 0.2, 0.3]]))
        model._features_dc = torch.nn.Parameter(torch.ones(1, 1, 3))
        model._features_rest = torch.nn.Parameter(torch.empty(1, 0, 3))
        model._scaling = torch.nn.Parameter(torch.ones(1, 3))
        model._rotation = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        model._opacity = torch.nn.Parameter(torch.ones(1, 1))
        model._exposure = torch.nn.Parameter(torch.eye(3, 4).repeat(2, 1, 1))
        model.exposure_mapping = {"left.png": 0, "right.png": 1}
        model.pretrained_exposures = None
        model.spatial_lr_scale = 1.0
        model.max_radii2D = torch.ones(1)
        with mock.patch("torch.zeros", self.cpu_zeros):
            model.training_setup(self.args)
        return model

    def advance(self, model, step):
        model.update_learning_rate(step)
        model.optimizer.zero_grad()
        model.exposure_optimizer.zero_grad()
        loss = model._xyz.square().sum()
        loss = loss + (model.get_exposure_from_name("left.png") - 0.3).square().sum()
        loss = loss + (model.get_exposure_from_name("right.png") + 0.7).square().sum()
        loss.backward()
        model.optimizer.step()
        model.exposure_optimizer.step()

    def round_trip(self, model):
        stream = io.BytesIO()
        torch.save(model.capture(), stream)
        stream.seek(0)
        kwargs = {"weights_only": False} if "weights_only" in inspect.signature(torch.load).parameters else {}
        return torch.load(stream, **kwargs)

    def restore(self, state, model):
        with mock.patch("torch.zeros", self.cpu_zeros):
            model.restore(state, self.args)

    def test_resumed_step_matches_uninterrupted_exposure_and_geometry(self):
        model = self.make_model()
        for step in range(1, 4):
            self.advance(model, step)
        state = self.round_trip(model)
        resumed = self.make_model()
        resumed.exposure_mapping = {"right.png": 0, "left.png": 1}
        self.restore(state, resumed)
        self.assertEqual(resumed.exposure_mapping, model.exposure_mapping)
        torch.testing.assert_close(resumed.get_exposure, model.get_exposure)
        for step in range(4, 7):
            self.advance(model, step)
            self.advance(resumed, step)
            torch.testing.assert_close(resumed.get_exposure, model.get_exposure, rtol=0, atol=0)
            torch.testing.assert_close(resumed.get_xyz, model.get_xyz, rtol=0, atol=0)

    def test_exposure_optimizer_owns_restored_parameter_and_moments(self):
        model = self.make_model()
        self.advance(model, 1)
        resumed = self.make_model()
        self.restore(self.round_trip(model), resumed)
        self.assertIs(resumed.exposure_optimizer.param_groups[0]["params"][0], resumed.get_exposure)
        expected = model.exposure_optimizer.state[model.get_exposure]
        actual = resumed.exposure_optimizer.state[resumed.get_exposure]
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_legacy_checkpoint_keeps_fresh_exposure_initialization(self):
        model = self.make_model()
        self.advance(model, 1)
        legacy = self.round_trip(model)[:12]
        resumed = self.make_model()
        initial = resumed.get_exposure.detach().clone()
        mapping = copy.copy(resumed.exposure_mapping)
        self.restore(legacy, resumed)
        torch.testing.assert_close(resumed.get_xyz, model.get_xyz)
        torch.testing.assert_close(resumed.get_exposure, initial)
        self.assertEqual(resumed.exposure_mapping, mapping)
        self.assertEqual(len(resumed.exposure_optimizer.state), 0)

    def test_pretrained_exposures_are_preserved(self):
        model = self.make_model()
        model.pretrained_exposures = {
            "left.png": torch.full((3, 4), 0.25),
            "right.png": torch.full((3, 4), 0.75),
        }
        resumed = self.make_model()
        self.restore(self.round_trip(model), resumed)
        torch.testing.assert_close(resumed.get_exposure_from_name("left.png"), model.get_exposure_from_name("left.png"))
        torch.testing.assert_close(resumed.get_exposure_from_name("right.png"), model.get_exposure_from_name("right.png"))


if __name__ == "__main__":
    unittest.main()
