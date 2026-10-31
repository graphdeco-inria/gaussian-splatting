import unittest
from unittest import mock

import torch
from torch import nn

from lpipsPyTorch.modules.lpips import LPIPS


class FeaturePyramid(nn.Module):
    n_channels_list = [3, 3]

    def forward(self, image):
        return [image, torch.nn.functional.avg_pool2d(image, 2)]


class LPIPSBatchTest(unittest.TestCase):
    def make_metric(self):
        weights = {
            "0.1.weight": torch.full((1, 3, 1, 1), 0.2),
            "1.1.weight": torch.full((1, 3, 1, 1), 0.3),
        }
        with mock.patch("lpipsPyTorch.modules.lpips.get_network", return_value=FeaturePyramid()), \
             mock.patch("lpipsPyTorch.modules.lpips.get_state_dict", return_value=weights):
            return LPIPS()

    def test_batched_distances_match_individual_images(self):
        metric = self.make_metric()
        x = torch.arange(3 * 3 * 4 * 4, dtype=torch.float32).reshape(3, 3, 4, 4) / 100
        y = torch.zeros_like(x)
        individual = torch.cat([metric(a[None], b[None]) for a, b in zip(x, y)])
        batched = metric(x, y)
        self.assertEqual(batched.shape, (3, 1, 1, 1))
        torch.testing.assert_close(batched, individual)

    def test_image_order_only_reorders_distances(self):
        metric = self.make_metric()
        x = torch.tensor([0.1, 0.5, 0.9])[:, None, None, None].expand(3, 3, 4, 4)
        y = torch.zeros_like(x)
        expected = metric(x, y)
        order = torch.tensor([2, 0, 1])
        torch.testing.assert_close(metric(x[order], y[order]), expected[order])

    def test_one_distance_has_no_gradient_to_another_image(self):
        metric = self.make_metric()
        x = torch.full((2, 3, 4, 4), 0.5, requires_grad=True)
        metric(x, torch.zeros_like(x))[0].sum().backward()
        self.assertGreater(torch.count_nonzero(x.grad[0]).item(), 0)
        self.assertEqual(torch.count_nonzero(x.grad[1]).item(), 0)

    def test_single_image_retains_layer_sum(self):
        metric = self.make_metric()
        x = torch.ones(1, 3, 4, 4)
        torch.testing.assert_close(metric(x, torch.zeros_like(x)), torch.full((1, 1, 1, 1), 1.5))


if __name__ == "__main__":
    unittest.main()
