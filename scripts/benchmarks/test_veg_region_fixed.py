#!/usr/bin/env python3
"""Focused invariants for the region-constrained VEG paper baseline."""

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scene.gaussian_model_factorsplat import GaussianModel


def bare_model():
    return GaussianModel.__new__(GaussianModel)


class RegionConstrainedVEGTests(unittest.TestCase):
    def test_interpolation_never_crosses_region_blocks(self):
        model = bare_model()
        model._tf_bank_lookup = torch.tensor([[
            [[0.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2],
             [0.4, 0.4, 0.4, 0.4], [0.6, 0.6, 0.6, 0.6]],
            [[10.0, 10.0, 10.0, 1.0], [20.0, 20.0, 20.0, 1.0],
             [30.0, 30.0, 30.0, 1.0], [40.0, 40.0, 40.0, 1.0]],
        ]])
        model._tf_veg_label = torch.tensor([0, 1], dtype=torch.int16)
        model._tf_veg_v = nn.Parameter(torch.zeros(2, 1))

        rgba = model._veg_rgba(0)

        torch.testing.assert_close(rgba[0], torch.full((4,), 0.3))
        torch.testing.assert_close(
            rgba[1], torch.tensor([25.0, 25.0, 25.0, 1.0]))

    def test_visibility_uses_fixed_region_and_preserves_no_support(self):
        model = bare_model()
        model._xyz = torch.zeros(3, 3)
        model._tf_base_index = 0
        model._tf_label_visible = torch.tensor([
            [True, True],
            [False, True],
        ])
        model._tf_veg_label = torch.tensor([0, 1, 0], dtype=torch.int16)
        model._tf_veg_has_support = torch.tensor([True, True, False])

        torch.testing.assert_close(
            model._veg_visibility(1).squeeze(1),
            torch.tensor([0.0, 1.0, 1.0]),
        )


if __name__ == "__main__":
    unittest.main()
