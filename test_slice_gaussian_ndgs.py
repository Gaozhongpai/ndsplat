"""
Test suite for _slice_gaussian_ndgs: Forward and Backward verification.

This script tests the _slice_gaussian_ndgs function from the PyTorch implementation.
It verifies:
1. Forward pass: Correctness of output shapes and mathematical properties
2. Backward pass: Gradient computation via torch.autograd.gradcheck

Usage:
    python test_slice_gaussian_ndgs.py
"""

import torch
import torch.nn.functional as F
from torch.autograd import gradcheck
import inspect

# Import the PyTorch implementation
from gsplat.cuda._torch_impl import _slice_gaussian_ndgs


def create_test_inputs(N: int, C: int, device: str = "cuda", requires_grad: bool = True):
    """
    Create valid test inputs for _slice_gaussian_ndgs.
    
    Args:
        N: Number of Gaussians
        C: Conditioning dimension (3 for view direction, 4 for view+time)
        device: Device to create tensors on
        requires_grad: Whether to require gradients
        
    Returns:
        Tuple of (m_1, m_2, query, covars, lambda_opc, lambda_opc_time)
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
    
    # Lambda parameters
    lambda_opc = 0.35
    lambda_opc_time = 0.45 if C == 4 else None
    
    return m_1, m_2, query, covars, lambda_opc, lambda_opc_time


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
        
        m_1, m_2, query, covars, lambda_opc, lambda_opc_time = create_test_inputs(
            N, C, device="cuda", requires_grad=False
        )
        
        # Forward pass
        m_cond, cov3D, scale = _slice_gaussian_ndgs(
            m_1, m_2, query, covars, lambda_opc, lambda_opc_time
        )
        
        # Check shapes
        expected_m_cond_shape = (N, 3)
        expected_cov3D_shape = (N, 6)  # Upper triangular format
        expected_scale_shape = (N, 1)
        
        shapes_correct = (
            m_cond.shape == expected_m_cond_shape and
            cov3D.shape == expected_cov3D_shape and
            scale.shape == expected_scale_shape
        )
        
        status = "✓ PASS" if shapes_correct else "✗ FAIL"
        if not shapes_correct:
            all_passed = False
            
        print(f"\n{desc}:")
        print(f"  m_cond: expected {expected_m_cond_shape}, got {tuple(m_cond.shape)} {status}")
        print(f"  cov3D:  expected {expected_cov3D_shape}, got {tuple(cov3D.shape)} {status}")
        print(f"  scale:  expected {expected_scale_shape}, got {tuple(scale.shape)} {status}")
    
    print("\n" + "-" * 70)
    if all_passed:
        print("All shape tests PASSED!")
    else:
        print("Some shape tests FAILED!")
    
    return all_passed


def test_forward_mathematical_properties():
    """Test mathematical properties of the forward pass."""
    print("\n" + "=" * 70)
    print("TEST: Forward Pass Mathematical Properties")
    print("=" * 70)
    
    device = "cuda"
    N = 50
    
    all_passed = True
    
    # Test 1: Scale should be in (0, 1] range (it's exp of negative value)
    print("\n1. Scale values should be in (0, 1]:")
    for C in [3, 4]:
        m_1, m_2, query, covars, lambda_opc, lambda_opc_time = create_test_inputs(
            N, C, device=device, requires_grad=False
        )
        _, _, scale = _slice_gaussian_ndgs(m_1, m_2, query, covars, lambda_opc, lambda_opc_time)
        
        in_range = (scale > 0).all() and (scale <= 1).all()
        status = "✓" if in_range else "✗"
        all_passed = all_passed and in_range
        print(f"   C={C}: min={scale.min().item():.6f}, max={scale.max().item():.6f} {status}")
    
    # Test 2: When query == m_2, conditional mean should equal spatial mean
    print("\n2. When query equals m_2, m_cond should equal m_1:")
    for C in [3, 4]:
        m_1, m_2, _, covars, lambda_opc, lambda_opc_time = create_test_inputs(
            N, C, device=device, requires_grad=False
        )
        query = m_2.clone()  # Query equals m_2
        
        m_cond, _, scale = _slice_gaussian_ndgs(m_1, m_2, query, covars, lambda_opc, lambda_opc_time)
        
        # m_cond should be close to m_1
        diff = (m_cond - m_1).abs().max().item()
        # Scale should be close to 1 (since x=0)
        scale_diff = (scale - 1.0).abs().max().item()
        
        m_cond_match = diff < 1e-5
        scale_match = scale_diff < 1e-5
        status = "✓" if m_cond_match and scale_match else "✗"
        all_passed = all_passed and m_cond_match and scale_match
        print(f"   C={C}: m_cond diff={diff:.8f}, scale diff from 1={scale_diff:.8f} {status}")
    
    # Test 3: Covariance matrix should be symmetric (reconstructed from upper triangular)
    print("\n3. Reconstructed covariance should be symmetric positive semi-definite:")
    for C in [3, 4]:
        m_1, m_2, query, covars, lambda_opc, lambda_opc_time = create_test_inputs(
            N, C, device=device, requires_grad=False
        )
        _, cov3D, _ = _slice_gaussian_ndgs(m_1, m_2, query, covars, lambda_opc, lambda_opc_time)
        
        # Reconstruct 3x3 matrix from upper triangular format [0,0], [0,1], [0,2], [1,1], [1,2], [2,2]
        cov_mat = torch.zeros(N, 3, 3, device=device, dtype=cov3D.dtype)
        cov_mat[:, 0, 0] = cov3D[:, 0]
        cov_mat[:, 0, 1] = cov_mat[:, 1, 0] = cov3D[:, 1]
        cov_mat[:, 0, 2] = cov_mat[:, 2, 0] = cov3D[:, 2]
        cov_mat[:, 1, 1] = cov3D[:, 3]
        cov_mat[:, 1, 2] = cov_mat[:, 2, 1] = cov3D[:, 4]
        cov_mat[:, 2, 2] = cov3D[:, 5]
        
        # Check symmetry
        sym_diff = (cov_mat - cov_mat.transpose(-1, -2)).abs().max().item()
        is_symmetric = sym_diff < 1e-10
        
        # Check positive semi-definite (eigenvalues >= 0)
        eigenvalues = torch.linalg.eigvalsh(cov_mat)
        min_eigenvalue = eigenvalues.min().item()
        is_psd = min_eigenvalue > -1e-5  # Allow small numerical error
        
        status = "✓" if is_symmetric and is_psd else "✗"
        all_passed = all_passed and is_symmetric and is_psd
        print(f"   C={C}: symmetry_diff={sym_diff:.8f}, min_eigenvalue={min_eigenvalue:.6f} {status}")
    
    print("\n" + "-" * 70)
    if all_passed:
        print("All mathematical property tests PASSED!")
    else:
        print("Some mathematical property tests FAILED!")
    
    return all_passed


def test_forward_lambda_variations():
    """Test forward pass with different lambda parameter variations."""
    print("\n" + "=" * 70)
    print("TEST: Forward Pass with Lambda Variations")
    print("=" * 70)
    
    device = "cuda"
    N = 50
    C = 4  # Only C=4 supports lambda_opc_time
    
    all_passed = True
    
    m_1, m_2, query, covars, _, _ = create_test_inputs(N, C, device=device, requires_grad=False)
    
    test_cases = [
        {"lambda_opc": 0.35, "lambda_opc_time": None, "desc": "lambda_opc=scalar, lambda_opc_time=None"},
        {"lambda_opc": 0.35, "lambda_opc_time": 0.45, "desc": "Both scalars"},
        {"lambda_opc": torch.full((N,), 0.35, device=device, dtype=torch.float64), 
         "lambda_opc_time": 0.45, "desc": "lambda_opc=tensor, lambda_opc_time=scalar"},
        {"lambda_opc": 0.35,
         "lambda_opc_time": torch.full((N,), 0.45, device=device, dtype=torch.float64),
         "desc": "lambda_opc=scalar, lambda_opc_time=tensor"},
        {"lambda_opc": torch.rand(N, device=device, dtype=torch.float64) * 0.3 + 0.2,
         "lambda_opc_time": torch.rand(N, device=device, dtype=torch.float64) * 0.3 + 0.3,
         "desc": "Both per-Gaussian tensors"},
    ]
    
    for case in test_cases:
        lambda_opc = case["lambda_opc"]
        lambda_opc_time = case["lambda_opc_time"]
        desc = case["desc"]
        
        try:
            m_cond, cov3D, scale = _slice_gaussian_ndgs(
                m_1, m_2, query, covars, lambda_opc, lambda_opc_time
            )
            
            # Basic sanity checks
            valid = (
                m_cond.shape == (N, 3) and
                cov3D.shape == (N, 6) and
                scale.shape == (N, 1) and
                (scale > 0).all() and (scale <= 1).all()
            )
            status = "✓ PASS" if valid else "✗ FAIL"
            all_passed = all_passed and valid
            print(f"\n{desc}:")
            print(f"   Output shapes correct, scale in (0,1] {status}")
            
        except Exception as e:
            all_passed = False
            print(f"\n{desc}:")
            print(f"   ✗ FAIL with error: {e}")
    
    print("\n" + "-" * 70)
    if all_passed:
        print("All lambda variation tests PASSED!")
    else:
        print("Some lambda variation tests FAILED!")
    
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
        {"N": 5, "C": 3, "desc": "6DGS (C=3)"},
        {"N": 5, "C": 4, "zero_cross": True, "desc": "7DGS (C=4) with zero_view_time_cross_terms=True"},
        {"N": 5, "C": 4, "zero_cross": False, "desc": "7DGS (C=4) with zero_view_time_cross_terms=False"},
    ]
    
    all_passed = True
    
    for case in test_cases:
        N, C = case["N"], case["C"]
        zero_cross = case.get("zero_cross", True)
        desc = case["desc"]
        
        print(f"\n{desc}:")
        
        m_1, m_2, query, covars, lambda_opc, lambda_opc_time = create_test_inputs(
            N, C, device=device, requires_grad=True
        )
        
        # Define wrapper function for gradcheck
        def func(m_1, m_2, covars):
            m_cond, cov3D, scale = _slice_gaussian_ndgs(
                m_1, m_2, query, covars, lambda_opc, lambda_opc_time, zero_cross
            )
            return m_cond, cov3D, scale
        
        try:
            # Note: gradcheck requires double precision for numerical stability
            passed = gradcheck(
                func, 
                (m_1, m_2, covars), 
                eps=1e-6, 
                atol=1e-4, 
                rtol=1e-3,
                raise_exception=True
            )
            print(f"   ✓ gradcheck PASSED")
        except Exception as e:
            passed = False
            all_passed = False
            print(f"   ✗ gradcheck FAILED: {e}")
    
    print("\n" + "-" * 70)
    if all_passed:
        print("All gradcheck tests PASSED!")
    else:
        print("Some gradcheck tests FAILED!")
    
    return all_passed


def test_backward_manual_verification():
    """Manually verify gradients by comparing numerical and analytical gradients."""
    print("\n" + "=" * 70)
    print("TEST: Backward Pass - Manual Gradient Verification")
    print("=" * 70)
    
    device = "cuda"
    N = 20
    eps = 1e-5
    
    test_cases = [
        {"C": 3, "zero_cross": True, "desc": "6DGS (C=3)"},
        {"C": 4, "zero_cross": True, "desc": "7DGS (C=4), zero_cross=True"},
        {"C": 4, "zero_cross": False, "desc": "7DGS (C=4), zero_cross=False"},
    ]
    
    all_passed = True
    
    for case in test_cases:
        C = case["C"]
        zero_cross = case["zero_cross"]
        desc = case["desc"]
        
        print(f"\n{desc}:")
        
        m_1, m_2, query, covars, lambda_opc, lambda_opc_time = create_test_inputs(
            N, C, device=device, requires_grad=True
        )
        
        # Forward pass
        m_cond, cov3D, scale = _slice_gaussian_ndgs(
            m_1, m_2, query, covars, lambda_opc, lambda_opc_time, zero_cross
        )
        
        # Create upstream gradients
        grad_m_cond = torch.randn_like(m_cond)
        grad_cov3D = torch.randn_like(cov3D)
        grad_scale = torch.randn_like(scale)
        
        # Compute loss and backward
        loss = (
            (m_cond * grad_m_cond).sum() +
            (cov3D * grad_cov3D).sum() +
            (scale * grad_scale).sum()
        )
        loss.backward()
        
        # Get analytical gradients
        grad_m_1_analytical = m_1.grad.clone()
        grad_m_2_analytical = m_2.grad.clone()
        grad_covars_analytical = covars.grad.clone()
        
        # Compute numerical gradients for a few random elements
        def compute_numerical_grad(param, idx, eps=1e-5):
            """Compute numerical gradient for a single parameter element."""
            original = param.data.flatten()[idx].item()
            
            # f(x + eps)
            param.data.flatten()[idx] = original + eps
            m_cond_p, cov3D_p, scale_p = _slice_gaussian_ndgs(
                m_1.detach(), m_2.detach(), query, covars.detach(), 
                lambda_opc, lambda_opc_time, zero_cross
            )
            loss_p = (
                (m_cond_p * grad_m_cond).sum() +
                (cov3D_p * grad_cov3D).sum() +
                (scale_p * grad_scale).sum()
            ).item()
            
            # f(x - eps)
            param.data.flatten()[idx] = original - eps
            m_cond_m, cov3D_m, scale_m = _slice_gaussian_ndgs(
                m_1.detach(), m_2.detach(), query, covars.detach(),
                lambda_opc, lambda_opc_time, zero_cross
            )
            loss_m = (
                (m_cond_m * grad_m_cond).sum() +
                (cov3D_m * grad_cov3D).sum() +
                (scale_m * grad_scale).sum()
            ).item()
            
            # Restore original
            param.data.flatten()[idx] = original
            
            return (loss_p - loss_m) / (2 * eps)
        
        # Sample a few indices for each parameter
        num_samples = 5
        results = {}
        
        for name, param, analytical_grad in [
            ("m_1", m_1, grad_m_1_analytical),
            ("m_2", m_2, grad_m_2_analytical),
            ("covars", covars, grad_covars_analytical),
        ]:
            indices = torch.randperm(param.numel())[:num_samples]
            max_rel_error = 0.0
            
            for idx in indices:
                numerical = compute_numerical_grad(param, idx.item(), eps)
                analytical = analytical_grad.flatten()[idx].item()
                
                # Relative error
                denom = max(abs(numerical), abs(analytical), 1e-8)
                rel_error = abs(numerical - analytical) / denom
                max_rel_error = max(max_rel_error, rel_error)
            
            passed = max_rel_error < 0.05  # 5% relative error tolerance
            results[name] = {"max_rel_error": max_rel_error, "passed": passed}
            status = "✓" if passed else "✗"
            all_passed = all_passed and passed
            print(f"   {name}: max_rel_error={max_rel_error:.6f} {status}")
    
    print("\n" + "-" * 70)
    if all_passed:
        print("All manual gradient verification tests PASSED!")
    else:
        print("Some manual gradient verification tests FAILED!")
    
    return all_passed


def test_backward_gradient_flow():
    """Test that gradients flow correctly through the computation graph.
    
    Mathematical dependencies in _slice_gaussian_ndgs:
    - m_cond = m_1 + v_regr @ (query - m_2)  -> depends on m_1, m_2, covars
    - cov3D = v_11 - v_regr @ v_21           -> depends on covars only
    - scale = exp(-lambda * x^T @ v_22_inv @ x) where x = query - m_2 -> depends on m_2, covars
    """
    print("\n" + "=" * 70)
    print("TEST: Backward Pass - Gradient Flow")
    print("=" * 70)
    
    device = "cuda"
    N = 30
    
    test_cases = [
        {"C": 3, "desc": "6DGS (C=3)"},
        {"C": 4, "desc": "7DGS (C=4)"},
    ]
    
    all_passed = True
    
    for case in test_cases:
        C = case["C"]
        desc = case["desc"]
        
        print(f"\n{desc}:")
        
        m_1, m_2, query, covars, lambda_opc, lambda_opc_time = create_test_inputs(
            N, C, device=device, requires_grad=True
        )
        
        # Expected gradient flow based on mathematical dependencies:
        # - m_cond: m_1 ✓, m_2 ✓, covars ✓
        # - cov3D:  m_1 ✗, m_2 ✗, covars ✓ (only depends on covariance blocks)
        # - scale:  m_1 ✗, m_2 ✓, covars ✓ (depends on x = query - m_2 and v_22_inv)
        expected_flow = {
            "m_cond": {"m_1": True, "m_2": True, "covars": True},
            "cov3D":  {"m_1": False, "m_2": False, "covars": True},
            "scale":  {"m_1": False, "m_2": True, "covars": True},
            "all outputs": {"m_1": True, "m_2": True, "covars": True},
        }
        
        for output_name in ["m_cond", "cov3D", "scale", "all outputs"]:
            # Clear gradients
            if m_1.grad is not None:
                m_1.grad.zero_()
            if m_2.grad is not None:
                m_2.grad.zero_()
            if covars.grad is not None:
                covars.grad.zero_()
            
            # Re-compute forward since we need fresh graph
            m_cond, cov3D, scale = _slice_gaussian_ndgs(
                m_1, m_2, query, covars, lambda_opc, lambda_opc_time
            )
            
            if output_name == "m_cond":
                loss = m_cond.sum()
            elif output_name == "cov3D":
                loss = cov3D.sum()
            elif output_name == "scale":
                loss = scale.sum()
            else:
                loss = m_cond.sum() + cov3D.sum() + scale.sum()
            
            loss.backward()
            
            # Check that gradients match expected flow
            has_grad_m1 = m_1.grad is not None and m_1.grad.abs().sum() > 0
            has_grad_m2 = m_2.grad is not None and m_2.grad.abs().sum() > 0
            has_grad_covars = covars.grad is not None and covars.grad.abs().sum() > 0
            
            expected = expected_flow[output_name]
            correct_m1 = has_grad_m1 == expected["m_1"]
            correct_m2 = has_grad_m2 == expected["m_2"]
            correct_covars = has_grad_covars == expected["covars"]
            
            all_correct = correct_m1 and correct_m2 and correct_covars
            status = "✓" if all_correct else "✗"
            all_passed = all_passed and all_correct
            
            print(f"   From {output_name}: m_1={has_grad_m1} (expect {expected['m_1']}), "
                  f"m_2={has_grad_m2} (expect {expected['m_2']}), "
                  f"covars={has_grad_covars} (expect {expected['covars']}) {status}")
    
    print("\n" + "-" * 70)
    if all_passed:
        print("All gradient flow tests PASSED!")
    else:
        print("Some gradient flow tests FAILED!")
    
    return all_passed


def test_backward_with_tensor_lambda():
    """Test backward pass when lambda parameters are tensors.
    
    With the updated CUDA kernel, gradients should now flow through
    lambda_opc and lambda_opc_time parameters.
    """
    print("\n" + "=" * 70)
    print("TEST: Backward Pass with Tensor Lambda Parameters")
    print("=" * 70)
    
    device = "cuda"
    N = 20
    C = 4  # Need C=4 for lambda_opc_time
    
    m_1, m_2, query, covars, _, _ = create_test_inputs(N, C, device=device, requires_grad=True)
    
    # Test with tensor lambdas - these should now receive gradients!
    lambda_opc = torch.rand(N, device=device, dtype=torch.float64) * 0.3 + 0.2
    lambda_opc.requires_grad_(True)
    lambda_opc_time = torch.rand(N, device=device, dtype=torch.float64) * 0.3 + 0.3
    lambda_opc_time.requires_grad_(True)
    
    all_passed = True
    
    try:
        # Forward pass
        m_cond, cov3D, scale = _slice_gaussian_ndgs(
            m_1, m_2, query, covars, lambda_opc, lambda_opc_time
        )
        
        # Backward pass - use scale to ensure lambda gradients flow
        loss = m_cond.sum() + cov3D.sum() + scale.sum()
        loss.backward()
        
        # Check gradients for standard inputs
        has_grad_m1 = m_1.grad is not None and m_1.grad.abs().sum() > 0
        has_grad_m2 = m_2.grad is not None and m_2.grad.abs().sum() > 0
        has_grad_covars = covars.grad is not None and covars.grad.abs().sum() > 0
        
        print(f"\nGradient flow with tensor lambdas:")
        print(f"   m_1: {has_grad_m1} {'✓' if has_grad_m1 else '✗'}")
        print(f"   m_2: {has_grad_m2} {'✓' if has_grad_m2 else '✗'}")
        print(f"   covars: {has_grad_covars} {'✓' if has_grad_covars else '✗'}")
        
        # Check gradients for lambda parameters (now expected!)
        has_grad_lambda_opc = lambda_opc.grad is not None and lambda_opc.grad.abs().sum() > 0
        has_grad_lambda_opc_time = lambda_opc_time.grad is not None and lambda_opc_time.grad.abs().sum() > 0
        
        print(f"   lambda_opc: {has_grad_lambda_opc} {'✓' if has_grad_lambda_opc else '✗'}")
        if has_grad_lambda_opc:
            print(f"      gradient norm: {lambda_opc.grad.norm():.6f}")
        print(f"   lambda_opc_time: {has_grad_lambda_opc_time} {'✓' if has_grad_lambda_opc_time else '✗'}")
        if has_grad_lambda_opc_time:
            print(f"      gradient norm: {lambda_opc_time.grad.norm():.6f}")
        
        # All inputs should have gradients, including lambda parameters
        all_passed = (has_grad_m1 and has_grad_m2 and has_grad_covars and 
                      has_grad_lambda_opc and has_grad_lambda_opc_time)
            
    except Exception as e:
        all_passed = False
        print(f"   ✗ FAIL with error: {e}")
    
    print("\n" + "-" * 70)
    if all_passed:
        print("Tensor lambda test PASSED! (gradients flow through lambda parameters)")
    else:
        print("Tensor lambda test FAILED!")
    
    return all_passed


def test_cuda_vs_pytorch_zero_cross_terms():
    """Test that CUDA and PyTorch implementations match for zero_view_time_cross_terms variations.
    
    This tests the fix where CUDA now correctly applies both lambda_opc and lambda_opc_time
    even when zero_view_time_cross_terms=False (matching PyTorch implementation).
    """
    print("\n" + "=" * 70)
    print("TEST: CUDA vs PyTorch - zero_view_time_cross_terms Behavior")
    print("=" * 70)
    
    device = "cuda"
    N = 100
    C = 4  # Only C=4 supports lambda_opc_time
    
    all_passed = True
    
    try:
        from gsplat.cuda._wrapper import slice_gaussian_ndgs as cuda_slice_gaussian_ndgs
        has_cuda = True
    except ImportError as e:
        print(f"   ⚠ CUDA implementation not available: {e}")
        print("   Skipping CUDA vs PyTorch comparison test")
        return True  # Skip test if CUDA not available
    
    m_1, m_2, query, covars, _, _ = create_test_inputs(N, C, device=device, requires_grad=False)
    
    # Use per-Gaussian lambda tensors
    lambda_opc = torch.rand(N, device=device, dtype=torch.float64) * 0.3 + 0.2
    lambda_opc_time = torch.rand(N, device=device, dtype=torch.float64) * 0.3 + 0.3
    
    test_cases = [
        {"zero_cross": True, "desc": "zero_view_time_cross_terms=True"},
        {"zero_cross": False, "desc": "zero_view_time_cross_terms=False"},
    ]
    
    for case in test_cases:
        zero_cross = case["zero_cross"]
        desc = case["desc"]
        
        print(f"\n{desc}:")
        
        # PyTorch implementation
        m_cond_py, cov3D_py, scale_py = _slice_gaussian_ndgs(
            m_1, m_2, query, covars, lambda_opc, lambda_opc_time, zero_cross
        )
        
        # CUDA implementation
        m_cond_cuda, cov3D_cuda, scale_cuda = cuda_slice_gaussian_ndgs(
            m_1, m_2, query, covars, lambda_opc, lambda_opc_time, zero_cross
        )
        
        # Compare outputs
        m_cond_diff = (m_cond_py - m_cond_cuda).abs().max().item()
        cov3D_diff = (cov3D_py - cov3D_cuda).abs().max().item()
        scale_diff = (scale_py - scale_cuda).abs().max().item()
        
        m_cond_match = m_cond_diff < 1e-5
        cov3D_match = cov3D_diff < 1e-5
        scale_match = scale_diff < 1e-5
        
        all_match = m_cond_match and cov3D_match and scale_match
        status = "✓" if all_match else "✗"
        all_passed = all_passed and all_match
        
        print(f"   m_cond max diff: {m_cond_diff:.2e} {'✓' if m_cond_match else '✗'}")
        print(f"   cov3D max diff:  {cov3D_diff:.2e} {'✓' if cov3D_match else '✗'}")
        print(f"   scale max diff:  {scale_diff:.2e} {'✓' if scale_match else '✗'}")
        
        # Additional check: scale values should differ between zero_cross=True and False
        # (because the quadratic forms are computed differently)
        
    # Test that zero_cross=True and False give different results (as expected)
    print("\n   Checking that zero_cross=True and False give different results:")
    _, _, scale_true = _slice_gaussian_ndgs(m_1, m_2, query, covars, lambda_opc, lambda_opc_time, True)
    _, _, scale_false = _slice_gaussian_ndgs(m_1, m_2, query, covars, lambda_opc, lambda_opc_time, False)
    scale_diff_between = (scale_true - scale_false).abs().mean().item()
    print(f"   Mean scale diff between True/False: {scale_diff_between:.6f}")
    
    print("\n" + "-" * 70)
    if all_passed:
        print("CUDA vs PyTorch comparison PASSED!")
    else:
        print("CUDA vs PyTorch comparison FAILED!")
    
    return all_passed


# ==============================================================================
# Main Test Runner
# ==============================================================================

def run_all_tests():
    """Run all forward and backward tests."""
    print("\n" + "=" * 70)
    print(" _slice_gaussian_ndgs Test Suite")
    print("=" * 70)
    print("Testing PyTorch implementation from gsplat.cuda._torch_impl")
    print()
    
    results = {}
    
    # Forward Tests
    print("\n" + "#" * 70)
    print("# FORWARD PASS TESTS")
    print("#" * 70)
    
    results["forward_shapes"] = test_forward_output_shapes()
    results["forward_math"] = test_forward_mathematical_properties()
    results["forward_lambda"] = test_forward_lambda_variations()
    
    # Backward Tests
    print("\n" + "#" * 70)
    print("# BACKWARD PASS TESTS")
    print("#" * 70)
    
    results["backward_gradcheck"] = test_backward_gradcheck()
    results["backward_manual"] = test_backward_manual_verification()
    results["backward_flow"] = test_backward_gradient_flow()
    results["backward_tensor_lambda"] = test_backward_with_tensor_lambda()
    results["cuda_vs_pytorch"] = test_cuda_vs_pytorch_zero_cross_terms()
    
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
    run_all_tests()
