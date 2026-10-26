"""Gradient correctness of the differentiable fused kernel (`upsample`)."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_upsample.kernels.bilinear import upsample
from mlx_upsample.variants import VARIANTS

SCALES = [2, 4, 1.5, 3, 0.5, 0.25]
# (B, H, W, C, scale_factor): covers each axis without the full product.
CASES = [(1, 16, 16, 4, (s, s)) for s in SCALES] + [
    (2, 7, 5, 1, (2, 2)),
    (1, 7, 5, 33, (1.5, 1.5)),
    (2, 7, 5, 4, (2, 3)),  # unequal height / width scale
]


def make(B, H, W, C, dtype=mx.float32):
    return mx.random.normal((B, H, W, C)).astype(dtype)


def stock_grad(x, scale_factor, align_corners, cot):
    return mx.vjp(lambda a: VARIANTS["gather"](a, scale_factor, align_corners), [x], [cot])[1][0]


@pytest.mark.parametrize("align_corners", [True, False])
@pytest.mark.parametrize("B,H,W,C,scale_factor", CASES)
def test_gradient_matches_gather_float32(B, H, W, C, scale_factor, align_corners):
    x = make(B, H, W, C)
    ref_out = VARIANTS["gather"](x, scale_factor, align_corners)
    cot = mx.random.normal(ref_out.shape)
    ref = stock_grad(x, scale_factor, align_corners, cot)
    out = mx.vjp(lambda a: upsample(a, scale_factor, align_corners), [x], [cot])[1][0]
    err = mx.abs(out - ref).max().item() / mx.abs(ref).max().item()
    assert err <= 1e-5


@pytest.mark.parametrize("align_corners", [True, False])
def test_gradient_float16_close_to_float32_reference(align_corners):
    x = make(2, 7, 5, 4, mx.float16)
    sf = (2, 2)
    cot = mx.random.normal((2, 14, 10, 4)).astype(mx.float16)
    ref = stock_grad(x.astype(mx.float32), sf, align_corners, cot.astype(mx.float32))
    grad = mx.vjp(lambda a: upsample(a, sf, align_corners), [x], [cot])[1][0]
    assert grad.dtype == mx.float16
    err = mx.abs(grad.astype(mx.float32) - ref).max().item()
    assert err <= 1e-3 * mx.abs(ref).max().item()


@pytest.mark.parametrize("scale_factor", [(2, 2), (0.5, 0.5), (1.5, 1.5), (2, 3)])
@pytest.mark.parametrize("align_corners", [True, False])
def test_adjoint(scale_factor, align_corners):
    # <upsample(x), g> must equal <x, vjp(g)> for any linear map: this holds
    # independent of any other implementation, so it checks the backward
    # kernel's math directly against the forward kernel it is paired with.
    x = make(2, 7, 5, 4)
    out = upsample(x, scale_factor, align_corners)
    g = mx.random.normal(out.shape)
    _, (grad,) = mx.vjp(lambda a: upsample(a, scale_factor, align_corners), [x], [g])
    lhs = (out * g).sum().item()
    rhs = (x * grad).sum().item()
    scale = max(abs(lhs), abs(rhs), 1.0)
    assert abs(lhs - rhs) <= 1e-5 * scale


def test_pspnet_1x1_scale60():
    x = make(1, 1, 1, 512)
    sf = (60, 60)
    cot = mx.random.normal((1, 60, 60, 512))
    ref = stock_grad(x, sf, True, cot)
    out = mx.vjp(lambda a: upsample(a, sf, True), [x], [cot])[1][0]
    err = mx.abs(out - ref).max().item() / mx.abs(ref).max().item()
    assert err <= 1e-5


@pytest.mark.parametrize("align_corners", [True, False])
@pytest.mark.parametrize("scale_factor", [(2, 2), (0.5, 0.5), (1.5, 1.5)])
def test_gradient_matches_finite_difference(scale_factor, align_corners):
    # Checks mx.grad against central differences, independent of stock MLX:
    # upsampling is linear, so the central difference is exact up to float32
    # rounding. Shape is tiny so looping over every input element is cheap.
    mx.random.seed(0)
    B, H, W, C = 1, 5, 4, 2
    x = make(B, H, W, C)
    out_shape = upsample(x, scale_factor, align_corners).shape
    w = mx.random.normal(out_shape)

    def loss(a):
        return (upsample(a, scale_factor, align_corners) * w).sum().item()

    grad = mx.grad(lambda a: (upsample(a, scale_factor, align_corners) * w).sum())(x)

    eps = 1e-2
    x_flat = np.array(x).reshape(-1)
    fd = np.zeros_like(x_flat)
    for i in range(x_flat.size):
        plus, minus = x_flat.copy(), x_flat.copy()
        plus[i] += eps
        minus[i] -= eps
        fd[i] = (loss(mx.array(plus.reshape(B, H, W, C))) -
                 loss(mx.array(minus.reshape(B, H, W, C)))) / (2 * eps)

    fd = fd.reshape(B, H, W, C)
    assert mx.abs(grad - mx.array(fd.astype(np.float32))).max().item() <= 1e-3


def test_grad_through_loss():
    x = make(2, 7, 5, 4)
    sf = (2, 2)
    w = mx.random.normal((2, 14, 10, 4))

    def loss(a):
        return (upsample(a, sf, False) * w).sum()

    value, grad = mx.value_and_grad(loss)(x)
    assert value.shape == ()
    assert grad.shape == x.shape
