"""
Test suite for slice_gaussian_ndgs_v2: Forward and Backward verification.

This script tests the slice_gaussian_ndgs_v2 function (both PyTorch and CUDA implementations).
It verifies:
1. Forward pass: Correctness of output shapes and mathematical properties
2. Backward pass: Gradient computation via torch.autograd.gradcheck
3. V2-specific features: lambda interpolation, optional position shift, v_11 return

Usage:
    python test_slice_gaussian_ndgs_v2.py
"""

import torch
import torch.nn.functional as F
from torch.autograd import gradcheck
import sys

# Import both PyTorch and CUDA implementations
from gsplat.cuda._torch_impl import _slice_gaussian_ndgs_v2 as pytorch_slice_v2

try:
    from gsplat.cuda._wrapper import slice_gaussian_ndgs_v2 as cuda_slice_v2
    HAS_CUDA = True
except ImportError as e:
    print(f"WARNING: CUDA implementation not available: {e}")
    print("Only PyTorch implementation will be tested.\n")
    HAS_CUDA = False


def create_test_inputs(N: int, C: int, device: str = "cuda", requires_grad: bool = True):
    """
    Create valid test inputs for slice_gaussian_ndgs_v2.

    Args:
        N: Number of Gaussians
        C: Conditioning dimension (3 for view direction, 4 for view+time)
        device: Device to create tensors on
        requires_grad: Whether to require gradients

    Returns:
        Tuple of (m_1, m_2, query, covars, lambda_view, lambda_time)
    """
    D = 3 + C  # Total dimensions (spatial + conditional)

    # Spatial means
    m_1 = torch.randn(N, 3, device=device, dtype=torch.float64, requires_grad=requires_grad)

    # Conditional means (normalized for view direction interpretation)
    m_2 = F.normalize(torch.randn(N, C, device=device, dtype=torch.float64), dim=-1)
    m_2.requires_grad_(requires_grad)

    # Query values (normalized)
    query = F.normalize(torch.randn(N, C, device=device, dtype=torch.float64), dim=-1)

    # Create valid positive semi-definite covariance matrices using L @ L^T construction
    L = torch.randn(N, D, D, device=device, dtype=torch.float64) * 0.5
    L = torch.tril(L)  # Make lower triangular
    L = L + torch.eye(D, device=device, dtype=torch.float64).unsqueeze(0) * 0.5  # Ensure positive definiteness
    covars = torch.bmm(L, L.transpose(-1, -2))
    covars.requires_grad_(requires_grad)

    # Lambda parameters (ALREADY sigmoid-activated, in [0,1])
    lambda_view = torch.rand(N, device=device, dtype=torch.float64) * 0.8 + 0.1  # [0.1, 0.9]
    lambda_time = torch.rand(N, device=device, dtype=torch.float64) * 0.8 + 0.1  # [0.1, 0.9]
    if requires_grad:
        lambda_view.requires_grad_(True)
        lambda_time.requires_grad_(True)

    return m_1, m_2, query, covars, lambda_view, lambda_time


# ==============================================================================
# Forward Pass Tests
# ==============================================================================

def test_forward_output_shapes():
    """Test that forward pass produces correct output shapes."""
    print("\n" + "=" * 70)
    print("TEST: Forward Pass Output Shapes")
    print("=" * 70)

    test_cases = [
        {"N": 10, "C": 3, "desc": "6DGS (view direction only)"},
        {"N": 10, "C": 4, "desc": "7DGS (view direction + time)"},
        {"N": 1, "C": 3, "desc": "Single Gaussian"},
        {"N": 100, "C": 4, "desc": "Batch of 100 Gaussians"},
    ]

    all_passed = True

    for case in test_cases:
        N, C = case["N"], case["C"]
        desc = case["desc"]

        m_1, m_2, query, covars, lambda_view, lambda_time = create_test_inputs(
            N, C, device="cuda", requires_grad=False
        )

        # Forward pass - PyTorch
        m_cond_py, cov3D_py, opacity_scale_py = pytorch_slice_v2(
            m_1, m_2, query, covars,
            use_view_dependent_pos=True,
            lambda_view=lambda_view,
            lambda_time=lambda_time,
            lambda_opc=0.35,
            zero_view_time_cross_terms=True
        )

        # Check shapes
        expected_m_cond_shape = (N, 3)
        expected_cov3D_shape = (N, 6)  # Upper triangular format
        expected_opacity_scale_shape = (N, 1)

        shapes_correct = (
            m_cond_py.shape == expected_m_cond_shape and
            cov3D_py.shape == expected_cov3D_shape and
            opacity_scale_py.shape == expected_opacity_scale_shape
        )

        status = "✓ PASS" if shapes_correct else "✗ FAIL"
        if not shapes_correct:
            all_passed = False

        print(f"\n{desc} (PyTorch):")
        print(f"  m_cond: expected {expected_m_cond_shape}, got {tuple(m_cond_py.shape)} {status}")
        print(f"  cov3D:  expected {expected_cov3D_shape}, got {tuple(cov3D_py.shape)} {status}")
        print(f"  opacity_scale:  expected {expected_opacity_scale_shape}, got {tuple(opacity_scale_py.shape)} {status}")

        # Test CUDA if available
        if HAS_CUDA:
            m_cond_cuda, cov3D_cuda, opacity_scale_cuda = cuda_slice_v2(
                m_1, m_2, query, covars,
                use_view_dependent_pos=True,
                lambda_view=lambda_view,
                lambda_time=lambda_time,
                lambda_opc=0.35,
                zero_view_time_cross_terms=True
            )

            shapes_correct_cuda = (
                m_cond_cuda.shape == expected_m_cond_shape and
                cov3D_cuda.shape == expected_cov3D_shape and
                opacity_scale_cuda.shape == expected_opacity_scale_shape
            )

            status_cuda = "✓ PASS" if shapes_correct_cuda else "✗ FAIL"
            if not shapes_correct_cuda:
                all_passed = False

            print(f"\n{desc} (CUDA):")
            print(f"  m_cond: expected {expected_m_cond_shape}, got {tuple(m_cond_cuda.shape)} {status_cuda}")
            print(f"  cov3D:  expected {expected_cov3D_shape}, got {tuple(cov3D_cuda.shape)} {status_cuda}")
            print(f"  opacity_scale:  expected {expected_opacity_scale_shape}, got {tuple(opacity_scale_cuda.shape)} {status_cuda}")

    print("\n" + "-" * 70)
    if all_passed:
        print("All shape tests PASSED!")
    else:
        print("Some shape tests FAILED!")

    return all_passed


def test_v2_specific_features():
    """Test V2-specific features: use_view_dependent_pos and v_11 return."""
    print("\n" + "=" * 70)
    print("TEST: V2-Specific Features")
    print("=" * 70)

    device = "cuda"
    N = 50
    C = 4

    all_passed = True

    m_1, m_2, query, covars, lambda_view, lambda_time = create_test_inputs(
        N, C, device=device, requires_grad=False
    )

    # Test 1: use_view_dependent_pos=False should return m_cond == m_1
    print("\n1. use_view_dependent_pos=False should return m_cond == m_1:")
    m_cond, _, _ = pytorch_slice_v2(
        m_1, m_2, query, covars,
        use_view_dependent_pos=False,
        lambda_view=lambda_view,
        lambda_time=lambda_time,
        lambda_opc=0.35
    )

    diff = (m_cond - m_1).abs().max().item()
    matches = diff < 1e-10
    status = "✓" if matches else "✗"
    all_passed = all_passed and matches
    print(f"   PyTorch: max diff={diff:.2e} {status}")

    if HAS_CUDA:
        m_cond_cuda, _, _ = cuda_slice_v2(
            m_1, m_2, query, covars,
            use_view_dependent_pos=False,
            lambda_view=lambda_view,
            lambda_time=lambda_time,
            lambda_opc=0.35
        )
        diff_cuda = (m_cond_cuda - m_1).abs().max().item()
        matches_cuda = diff_cuda < 1e-10
        status_cuda = "✓" if matches_cuda else "✗"
        all_passed = all_passed and matches_cuda
        print(f"   CUDA: max diff={diff_cuda:.2e} {status_cuda}")

    # Test 2: cov3D should be v_11 (not Schur complement)
    print("\n2. cov3D should equal v_11 from input covariance:")
    m_cond, cov3D, _ = pytorch_slice_v2(
        m_1, m_2, query, covars,
        use_view_dependent_pos=True,
        lambda_view=lambda_view,
        lambda_time=lambda_time,
        lambda_opc=0.35
    )

    # Extract v_11 from input covars
    v_11_input = covars[:, :3, :3]  # [N, 3, 3]

    # Reconstruct from upper triangular
    cov3D_reconstructed = torch.zeros(N, 3, 3, device=device, dtype=cov3D.dtype)
    cov3D_reconstructed[:, 0, 0] = cov3D[:, 0]
    cov3D_reconstructed[:, 0, 1] = cov3D_reconstructed[:, 1, 0] = cov3D[:, 1]
    cov3D_reconstructed[:, 0, 2] = cov3D_reconstructed[:, 2, 0] = cov3D[:, 2]
    cov3D_reconstructed[:, 1, 1] = cov3D[:, 3]
    cov3D_reconstructed[:, 1, 2] = cov3D_reconstructed[:, 2, 1] = cov3D[:, 4]
    cov3D_reconstructed[:, 2, 2] = cov3D[:, 5]

    diff_v11 = (cov3D_reconstructed - v_11_input).abs().max().item()
    matches_v11 = diff_v11 < 1e-10
    status_v11 = "✓" if matches_v11 else "✗"
    all_passed = all_passed and matches_v11
    print(f"   PyTorch: max diff from v_11={diff_v11:.2e} {status_v11}")

    if HAS_CUDA:
        _, cov3D_cuda, _ = cuda_slice_v2(
            m_1, m_2, query, covars,
            use_view_dependent_pos=True,
            lambda_view=lambda_view,
            lambda_time=lambda_time,
            lambda_opc=0.35
        )

        cov3D_cuda_reconstructed = torch.zeros(N, 3, 3, device=device, dtype=cov3D_cuda.dtype)
        cov3D_cuda_reconstructed[:, 0, 0] = cov3D_cuda[:, 0]
        cov3D_cuda_reconstructed[:, 0, 1] = cov3D_cuda_reconstructed[:, 1, 0] = cov3D_cuda[:, 1]
        cov3D_cuda_reconstructed[:, 0, 2] = cov3D_cuda_reconstructed[:, 2, 0] = cov3D_cuda[:, 2]
        cov3D_cuda_reconstructed[:, 1, 1] = cov3D_cuda[:, 3]
        cov3D_cuda_reconstructed[:, 1, 2] = cov3D_cuda_reconstructed[:, 2, 1] = cov3D_cuda[:, 4]
        cov3D_cuda_reconstructed[:, 2, 2] = cov3D_cuda[:, 5]

        diff_v11_cuda = (cov3D_cuda_reconstructed - v_11_input).abs().max().item()
        matches_v11_cuda = diff_v11_cuda < 1e-5  # Allow slightly more tolerance for CUDA
        status_v11_cuda = "✓" if matches_v11_cuda else "✗"
        all_passed = all_passed and matches_v11_cuda
        print(f"   CUDA: max diff from v_11={diff_v11_cuda:.2e} {status_v11_cuda}")

    # Test 3: Lambda interpolation effect
    print("\n3. Lambda interpolation: lambda=0 vs lambda=1 should give different results:")

    # Lambda = 0 (decoupled)
    m_cond_0, _, _ = pytorch_slice_v2(
        m_1, m_2, query, covars,
        use_view_dependent_pos=True,
        lambda_view=torch.zeros(N, device=device, dtype=torch.float64),
        lambda_time=torch.zeros(N, device=device, dtype=torch.float64),
        lambda_opc=0.35
    )

    # Lambda = 1 (fully coupled)
    m_cond_1, _, _ = pytorch_slice_v2(
        m_1, m_2, query, covars,
        use_view_dependent_pos=True,
        lambda_view=torch.ones(N, device=device, dtype=torch.float64),
        lambda_time=torch.ones(N, device=device, dtype=torch.float64),
        lambda_opc=0.35
    )

    diff_lambda = (m_cond_0 - m_cond_1).abs().mean().item()
    has_effect = diff_lambda > 1e-6
    status_lambda = "✓" if has_effect else "✗"
    all_passed = all_passed and has_effect
    print(f"   PyTorch: mean diff between lambda=0 and lambda=1: {diff_lambda:.6f} {status_lambda}")

    print("\n" + "-" * 70)
    if all_passed:
        print("All V2-specific feature tests PASSED!")
    else:
        print("Some V2-specific feature tests FAILED!")

    return all_passed


def test_cuda_vs_pytorch():
    """Compare CUDA and PyTorch implementations."""
    if not HAS_CUDA:
        print("\n" + "=" * 70)
        print("TEST: CUDA vs PyTorch - SKIPPED (CUDA not available)")
        print("=" * 70)
        return True

    print("\n" + "=" * 70)
    print("TEST: CUDA vs PyTorch Implementation Consistency")
    print("=" * 70)

    device = "cuda"
    N = 100

    all_passed = True

    test_cases = [
        {"C": 3, "use_pos": True, "zero_cross": True, "desc": "6DGS, use_pos=True, zero_cross=True"},
        {"C": 4, "use_pos": True, "zero_cross": True, "desc": "7DGS, use_pos=True, zero_cross=True"},
        {"C": 4, "use_pos": True, "zero_cross": False, "desc": "7DGS, use_pos=True, zero_cross=False"},
        {"C": 4, "use_pos": False, "zero_cross": True, "desc": "7DGS, use_pos=False, zero_cross=True"},
    ]

    for case in test_cases:
        C = case["C"]
        use_pos = case["use_pos"]
        zero_cross = case["zero_cross"]
        desc = case["desc"]

        print(f"\n{desc}:")

        m_1, m_2, query, covars, lambda_view, lambda_time = create_test_inputs(
            N, C, device=device, requires_grad=False
        )

        # PyTorch implementation
        m_cond_py, cov3D_py, opacity_scale_py = pytorch_slice_v2(
            m_1, m_2, query, covars,
            use_view_dependent_pos=use_pos,
            lambda_view=lambda_view,
            lambda_time=lambda_time,
            lambda_opc=0.35,
            zero_view_time_cross_terms=zero_cross
        )

        # CUDA implementation
        m_cond_cuda, cov3D_cuda, opacity_scale_cuda = cuda_slice_v2(
            m_1, m_2, query, covars,
            use_view_dependent_pos=use_pos,
            lambda_view=lambda_view,
            lambda_time=lambda_time,
            lambda_opc=0.35,
            zero_view_time_cross_terms=zero_cross
        )

        # Compare outputs
        m_cond_diff = (m_cond_py - m_cond_cuda).abs().max().item()
        cov3D_diff = (cov3D_py - cov3D_cuda).abs().max().item()
        opacity_scale_diff = (opacity_scale_py - opacity_scale_cuda).abs().max().item()

        tolerance = 1e-5
        m_cond_match = m_cond_diff < tolerance
        cov3D_match = cov3D_diff < tolerance
        opacity_scale_match = opacity_scale_diff < tolerance

        all_match = m_cond_match and cov3D_match and opacity_scale_match
        all_passed = all_passed and all_match

        print(f"   m_cond max diff: {m_cond_diff:.2e} {'✓' if m_cond_match else '✗'}")
        print(f"   cov3D max diff:  {cov3D_diff:.2e} {'✓' if cov3D_match else '✗'}")
        print(f"   opacity_scale max diff:  {opacity_scale_diff:.2e} {'✓' if opacity_scale_match else '✗'}")

    print("\n" + "-" * 70)
    if all_passed:
        print("CUDA vs PyTorch consistency tests PASSED!")
    else:
        print("CUDA vs PyTorch consistency tests FAILED!")

    return all_passed


# ==============================================================================
# Backward Pass Tests
# ==============================================================================

def test_backward_gradcheck():
    """Use torch.autograd.gradcheck to verify backward implementation."""
    print("\n" + "=" * 70)
    print("TEST: Backward Pass - torch.autograd.gradcheck")
    print("=" * 70)

    device = "cuda"

    test_cases = [
        {"N": 3, "C": 3, "use_pos": True, "desc": "6DGS, use_pos=True"},
        {"N": 3, "C": 4, "use_pos": True, "zero_cross": True, "desc": "7DGS, use_pos=True, zero_cross=True"},
        {"N": 3, "C": 4, "use_pos": False, "zero_cross": True, "desc": "7DGS, use_pos=False"},
    ]

    all_passed = True

    for case in test_cases:
        N, C = case["N"], case["C"]
        use_pos = case["use_pos"]
        zero_cross = case.get("zero_cross", True)
        desc = case["desc"]

        print(f"\n{desc}:")

        m_1, m_2, query, covars, lambda_view, lambda_time = create_test_inputs(
            N, C, device=device, requires_grad=True
        )

        # Define wrapper function for gradcheck
        def func(m_1, m_2, covars, lambda_view, lambda_time):
            m_cond, cov3D, opacity_scale = pytorch_slice_v2(
                m_1, m_2, query, covars,
                use_view_dependent_pos=use_pos,
                lambda_view=lambda_view,
                lambda_time=lambda_time,
                lambda_opc=0.35,
                zero_view_time_cross_terms=zero_cross
            )
            return m_cond, cov3D, opacity_scale

        try:
            # Note: gradcheck requires double precision for numerical stability
            passed = gradcheck(
                func,
                (m_1, m_2, covars, lambda_view, lambda_time),
                eps=1e-6,
                atol=1e-4,
                rtol=1e-3,
                raise_exception=True
            )
            print(f"   ✓ gradcheck PASSED (PyTorch)")
        except Exception as e:
            passed = False
            all_passed = False
            print(f"   ✗ gradcheck FAILED (PyTorch): {str(e)[:100]}")

        # Test CUDA if available
        if HAS_CUDA:
            def func_cuda(m_1, m_2, covars, lambda_view, lambda_time):
                m_cond, cov3D, opacity_scale = cuda_slice_v2(
                    m_1, m_2, query, covars,
                    use_view_dependent_pos=use_pos,
                    lambda_view=lambda_view,
                    lambda_time=lambda_time,
                    lambda_opc=0.35,
                    zero_view_time_cross_terms=zero_cross
                )
                return m_cond, cov3D, opacity_scale

            try:
                passed_cuda = gradcheck(
                    func_cuda,
                    (m_1, m_2, covars, lambda_view, lambda_time),
                    eps=1e-6,
                    atol=1e-4,
                    rtol=1e-3,
                    raise_exception=True
                )
                print(f"   ✓ gradcheck PASSED (CUDA)")
            except Exception as e:
                passed_cuda = False
                all_passed = False
                print(f"   ✗ gradcheck FAILED (CUDA): {str(e)[:100]}")

    print("\n" + "-" * 70)
    if all_passed:
        print("All gradcheck tests PASSED!")
    else:
        print("Some gradcheck tests FAILED!")

    return all_passed


def test_lambda_gradients():
    """Test that gradients flow through lambda parameters."""
    print("\n" + "=" * 70)
    print("TEST: Lambda Parameter Gradients")
    print("=" * 70)

    device = "cuda"
    N = 20
    C = 4

    m_1, m_2, query, covars, _, _ = create_test_inputs(N, C, device=device, requires_grad=True)

    # Lambda parameters with gradients
    lambda_view = torch.rand(N, device=device, dtype=torch.float64, requires_grad=True)
    lambda_time = torch.rand(N, device=device, dtype=torch.float64, requires_grad=True)

    all_passed = True

    # Test PyTorch
    print("\nPyTorch implementation:")
    m_cond, cov3D, opacity_scale = pytorch_slice_v2(
        m_1, m_2, query, covars,
        use_view_dependent_pos=True,
        lambda_view=lambda_view,
        lambda_time=lambda_time,
        lambda_opc=0.35
    )

    loss = m_cond.sum() + cov3D.sum() + opacity_scale.sum()
    loss.backward()

    has_grad_lambda_view = lambda_view.grad is not None and lambda_view.grad.abs().sum() > 0
    has_grad_lambda_time = lambda_time.grad is not None and lambda_time.grad.abs().sum() > 0

    status_view = "✓" if has_grad_lambda_view else "✗"
    status_time = "✓" if has_grad_lambda_time else "✗"

    print(f"   lambda_view has gradient: {has_grad_lambda_view} {status_view}")
    if has_grad_lambda_view:
        print(f"      norm: {lambda_view.grad.norm():.6f}")
    print(f"   lambda_time has gradient: {has_grad_lambda_time} {status_time}")
    if has_grad_lambda_time:
        print(f"      norm: {lambda_time.grad.norm():.6f}")

    all_passed = all_passed and has_grad_lambda_view and has_grad_lambda_time

    # Test CUDA if available
    if HAS_CUDA:
        print("\nCUDA implementation:")

        # Reset gradients
        m_1.grad = None
        m_2.grad = None
        covars.grad = None
        lambda_view.grad = None
        lambda_time.grad = None

        m_cond_cuda, cov3D_cuda, opacity_scale_cuda = cuda_slice_v2(
            m_1, m_2, query, covars,
            use_view_dependent_pos=True,
            lambda_view=lambda_view,
            lambda_time=lambda_time,
            lambda_opc=0.35
        )

        loss_cuda = m_cond_cuda.sum() + cov3D_cuda.sum() + opacity_scale_cuda.sum()
        loss_cuda.backward()

        has_grad_lambda_view_cuda = lambda_view.grad is not None and lambda_view.grad.abs().sum() > 0
        has_grad_lambda_time_cuda = lambda_time.grad is not None and lambda_time.grad.abs().sum() > 0

        status_view_cuda = "✓" if has_grad_lambda_view_cuda else "✗"
        status_time_cuda = "✓" if has_grad_lambda_time_cuda else "✗"

        print(f"   lambda_view has gradient: {has_grad_lambda_view_cuda} {status_view_cuda}")
        if has_grad_lambda_view_cuda:
            print(f"      norm: {lambda_view.grad.norm():.6f}")
        print(f"   lambda_time has gradient: {has_grad_lambda_time_cuda} {status_time_cuda}")
        if has_grad_lambda_time_cuda:
            print(f"      norm: {lambda_time.grad.norm():.6f}")

        all_passed = all_passed and has_grad_lambda_view_cuda and has_grad_lambda_time_cuda

    print("\n" + "-" * 70)
    if all_passed:
        print("Lambda gradient tests PASSED!")
    else:
        print("Lambda gradient tests FAILED!")

    return all_passed


# ==============================================================================
# Main Test Runner
# ==============================================================================

def run_all_tests():
    """Run all forward and backward tests."""
    print("\n" + "=" * 70)
    print(" slice_gaussian_ndgs_v2 Test Suite")
    print("=" * 70)
    print("Testing both PyTorch and CUDA implementations")
    if not HAS_CUDA:
        print("WARNING: CUDA implementation not available, only testing PyTorch")
    print()

    results = {}

    # Forward Tests
    print("\n" + "#" * 70)
    print("# FORWARD PASS TESTS")
    print("#" * 70)

    results["forward_shapes"] = test_forward_output_shapes()
    results["v2_features"] = test_v2_specific_features()
    results["cuda_vs_pytorch"] = test_cuda_vs_pytorch()

    # Backward Tests
    print("\n" + "#" * 70)
    print("# BACKWARD PASS TESTS")
    print("#" * 70)

    results["backward_gradcheck"] = test_backward_gradcheck()
    results["lambda_gradients"] = test_lambda_gradients()

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {test_name}: {status}")

    all_passed = all(results.values())
    print("\n" + "-" * 70)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
    else:
        num_failed = sum(1 for v in results.values() if not v)
        print(f"{num_failed} TEST(S) FAILED! ✗")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
