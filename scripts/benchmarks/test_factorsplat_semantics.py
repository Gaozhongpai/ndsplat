#!/usr/bin/env python3
"""Focused invariants for the experimental FactorSplat OOD corrections."""

import types
import unittest
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scene.gaussian_model_factorsplat import GaussianModel


def bare_model():
    return GaussianModel.__new__(GaussianModel)


class FactorSplatSemanticTests(unittest.TestCase):
    def test_dual_code_keeps_total_rank_and_exact_identity(self):
        model = bare_model()
        model.tf_global_context_rank = 3
        model._local_tf_code = types.MethodType(
            lambda self, index, opacity_only=False: torch.zeros(5, 5), model)
        model._context_code_for_index = types.MethodType(
            lambda self, index: torch.zeros(3), model)
        code = model._local_context_code(0)
        self.assertEqual(tuple(code.shape), (5, 8))
        self.assertTrue(torch.equal(code, torch.zeros_like(code)))

    def test_global_context_is_label_order_invariant(self):
        model = bare_model()
        model.tf_global_context_rank = 2
        model._tf_context_descriptors = torch.tensor([[
            [1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
        ]])
        model.tf_context_encoder = nn.Linear(2, 4, bias=False)
        model.tf_context_encoder_psi = nn.Linear(4, 2, bias=False)
        before = model._context_code_for_index(0)
        model._tf_context_descriptors = model._tf_context_descriptors[:, [2, 0, 1]]
        after = model._context_code_for_index(0)
        torch.testing.assert_close(before, after)

    def test_veg_interpolation_never_crosses_region_blocks(self):
        model = bare_model()
        model._tf_bank_lookup = torch.tensor([[
            [[0.0, 0.0, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2],
             [0.4, 0.4, 0.4, 0.4], [0.6, 0.6, 0.6, 0.6]],
            [[10.0, 10.0, 10.0, 1.0], [20.0, 20.0, 20.0, 1.0],
             [30.0, 30.0, 30.0, 1.0], [40.0, 40.0, 40.0, 1.0]],
        ]])
        model._tf_veg_label = torch.tensor([0, 1], dtype=torch.int16)
        # Both primitives sit halfway through their own four-bin curve.
        model._tf_veg_v = nn.Parameter(torch.zeros(2, 1))
        rgba = model._veg_rgba(0)
        torch.testing.assert_close(rgba[0], torch.full((4,), 0.3))
        torch.testing.assert_close(
            rgba[1], torch.tensor([25.0, 25.0, 25.0, 1.0]))

    def test_veg_visibility_uses_fixed_region_and_preserves_no_support(self):
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
            torch.tensor([0.0, 1.0, 1.0]))

    def test_training_envelope_is_per_gaussian(self):
        model = bare_model()
        model._xyz = torch.zeros(2, 3)
        model._tf_train_rows = [0, 1, 2]
        model._tf_opacity_envelope = None
        model.tf_opacity_alpha_identity_gate = False
        offsets = {
            0: torch.tensor([0.0, 0.0]),
            1: torch.tensor([-2.0, 3.0]),
            2: torch.tensor([1.0, -4.0]),
        }
        model._opacity_factor_offset = types.MethodType(
            lambda self, row: offsets[row], model)
        lower, upper = model._opacity_training_envelope()
        torch.testing.assert_close(lower, torch.tensor([-2.0, -4.0]))
        torch.testing.assert_close(upper, torch.tensor([1.0, 3.0]))

    def test_opacity_residual_clip_is_in_logit_units(self):
        model = bare_model()
        model.tf_opacity_scale = 4.0
        model.tf_opacity_residual_clip = 1.0
        offset = model._bounded_opacity_offset(torch.tensor([-2.0, 0.1, 2.0]))
        torch.testing.assert_close(offset, torch.tensor([-1.0, 0.4, 1.0]))
        model.tf_opacity_residual_clip = 0.0
        torch.testing.assert_close(
            model._bounded_opacity_offset(torch.tensor([-2.0, 0.1, 2.0])),
            torch.tensor([-8.0, 0.4, 8.0]))

    def test_alpha_identity_gate_detects_only_alpha_edits(self):
        model = bare_model()
        model._tf_base_index = 0
        model._tf_bank_lookup = torch.tensor([
            [[[0.1, 0.2, 0.3, 0.4]], [[0.3, 0.2, 0.1, 0.6]]],
            [[[0.8, 0.1, 0.4, 0.4]], [[0.2, 0.7, 0.6, 0.6]]],
            [[[0.1, 0.2, 0.3, 0.4]], [[0.3, 0.2, 0.1, 0.7]]],
        ])
        self.assertFalse(model._alpha_curve_changed(0))
        self.assertFalse(model._alpha_curve_changed(1))
        self.assertTrue(model._alpha_curve_changed(2))

    def test_alpha_only_code_is_zero_for_color_only_edit(self):
        model = bare_model()
        model._tf_base_index = 0
        model.tf_log_ratio_encoder = False
        model._tf_bank_log_alpha = None
        # Two material entries, one primitive gathering both equally. Preset 1
        # changes RGB only and therefore must have a zero opacity code.
        model._tf_bank_lookup = torch.tensor([
            [[[[0.1, 0.2, 0.3, 0.4]]], [[[0.3, 0.2, 0.1, 0.6]]]],
            [[[[0.8, 0.1, 0.4, 0.4]]], [[[0.2, 0.7, 0.6, 0.6]]]],
        ]).reshape(2, 2, 1, 4)
        model._tf_lookup_ids = torch.tensor([[0, 1]], dtype=torch.int16)
        model._tf_lookup_counts = torch.tensor([2], dtype=torch.uint8)
        model._tf_lookup_w = torch.tensor([[0.5, 0.5]])
        model.tf_encoder = nn.Sequential(
            nn.Linear(4, 4, bias=False), nn.ReLU(),
            nn.Linear(4, 4, bias=False),
        )
        with torch.no_grad():
            model.tf_encoder[0].weight.copy_(torch.eye(4))
            model.tf_encoder[2].weight.copy_(torch.eye(4))

        def contract(self, tables):
            table = tables[0]
            return (table[self._tf_lookup_ids.long()]
                    * self._tf_lookup_w[..., None]).sum(dim=1)

        model._contract_local = types.MethodType(contract, model)
        color_code = model._local_tf_code(1)
        opacity_code = model._local_tf_code(1, opacity_only=True)
        self.assertGreater(color_code.detach().abs().max().item(), 0.0)
        self.assertTrue(torch.equal(opacity_code, torch.zeros_like(opacity_code)))

        # An authored alpha edit must still reach the opacity code.
        model._tf_bank_lookup[1, :, :, 3] += 0.2
        opacity_code = model._local_tf_code(1, opacity_only=True)
        self.assertGreater(opacity_code.detach().abs().max().item(), 0.0)

    def test_soft_visibility_returns_surviving_descriptor_mass(self):
        model = bare_model()
        model._tf_base_index = 0
        model._tf_bank_lookup = torch.zeros(2, 2, 1, 4)
        model._tf_label_visible = torch.tensor([[True, True], [False, True]])
        # Removed-label masses are 0, .25, .5, and 1.
        model._tf_lookup_ids = torch.tensor(
            [[1, 0], [0, 1], [0, 1], [0, 0]], dtype=torch.int16)
        model._tf_lookup_counts = torch.tensor([1, 2, 2, 2], dtype=torch.uint8)
        model._tf_lookup_w = torch.tensor(
            [[1.0, 0.0], [0.25, 0.75], [0.5, 0.5], [0.5, 0.5]])
        model._tf_lookup_p = None
        model.tf_gate_removed_mass = 0.5

        model.tf_soft_visibility_gate = True
        soft = model._authored_visibility(1).squeeze(1)
        torch.testing.assert_close(soft, torch.tensor([1.0, 0.75, 0.5, 0.0]))

        model.tf_soft_visibility_gate = False
        hard = model._authored_visibility(1).squeeze(1)
        torch.testing.assert_close(hard, torch.tensor([1.0, 1.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
