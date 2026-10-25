"""Correctness of the fused Metal kernel against stock mlx.nn.Upsample."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_upsample.kernels.bilinear import _tables_cached, fused, fused_uncached

FUSED_FNS = [fused, fused_uncached]
SCALES = [2, 4, 1.5, 0.5, 0.25]
# (B, H, W, C, scale_factor): covers each dimension without the full product.
CASES = [(1, 16, 16, 4, (s, s)) for s in SCALES] + [
    (2, 7, 5, 1, (2, 2)),
    (2, 7, 5, 32, (1.5, 1.5)),
    (1, 7, 5, 4, (0.5, 0.5)),
    (2, 16, 16, 32, (4, 4)),
    (1, 16, 16, 1, (0.25, 0.25)),
    (1, 7, 5, 4, (2, 3)),  # unequal height / width scale
    (1, 7, 5, 33, (2, 2)),
]


def make(B, H, W, C, dtype):
    return mx.random.normal((B, H, W, C)).astype(dtype)


def reference(x, scale_factor, align_corners):
    return nn.Upsample(scale_factor=scale_factor, mode="linear", align_corners=align_corners)(x)


@pytest.mark.parametrize("fn", FUSED_FNS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("align_corners", [True, False])
@pytest.mark.parametrize(
    "dtype,atol", [(mx.float32, 1e-5), (mx.float16, 1e-2), (mx.bfloat16, 1e-2)]
)
@pytest.mark.parametrize("B,H,W,C,scale_factor", CASES)
def test_matches_stock_upsample(fn, dtype, atol, align_corners, B, H, W, C, scale_factor):
    x = make(B, H, W, C, dtype)
    ref = reference(x, scale_factor, align_corners)
    out = fn(x, scale_factor, align_corners)
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    assert mx.abs(out - ref).max().item() <= atol


@pytest.mark.parametrize("fn", FUSED_FNS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("align_corners", [True, False])
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
def test_odd_size(fn, dtype, align_corners):
    x = make(2, 7, 5, 4, dtype)
    ref = reference(x, (1.5, 1.5), align_corners)
    out = fn(x, (1.5, 1.5), align_corners)
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    atol = 1e-5 if dtype == mx.float32 else 1e-2
    assert mx.abs(out - ref).max().item() <= atol


@pytest.mark.parametrize("fn", FUSED_FNS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
def test_issue_shape(fn, dtype):
    x = make(1, 192, 320, 4, dtype)
    ref = reference(x, (4, 4), False)
    out = fn(x, (4, 4), False)
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    atol = 1e-5 if dtype == mx.float32 else 1e-2
    assert mx.abs(out - ref).max().item() <= atol


def test_fused_cached_hits_cache():
    x = make(1, 16, 16, 4, mx.float32)
    _tables_cached.cache_clear()
    out1 = fused(x, (2, 2), False)
    hits_before = _tables_cached.cache_info().hits
    out2 = fused(x, (2, 2), False)
    assert _tables_cached.cache_info().hits == hits_before + 1
    assert mx.array_equal(out1, out2).item()
