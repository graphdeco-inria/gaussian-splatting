import unittest

import torch

from utils.image_utils import composite_with_background


class CompositeWithBackgroundTest(unittest.TestCase):

    def test_composites_alpha_over_background(self):
        image = torch.zeros((3, 1, 3))
        image[:, :, 1:] = 1.0
        alpha = torch.tensor([[[0.0, 0.5, 1.0]]])
        background = torch.tensor([1.0, 0.5, 0.0])

        composited = composite_with_background(image, alpha, background)

        expected = torch.tensor([
            [[1.0, 1.0, 1.0]],
            [[0.5, 0.75, 1.0]],
            [[0.0, 0.5, 1.0]],
        ])
        self.assertTrue(torch.allclose(composited, expected))

    def test_returns_image_when_alpha_is_missing(self):
        image = torch.rand((3, 2, 2))

        composited = composite_with_background(image, None, torch.ones(3))

        self.assertIs(composited, image)


if __name__ == "__main__":
    unittest.main()
