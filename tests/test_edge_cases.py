"""Edge cases for bilinear upsampling, probing shape/dtype/layout extremes.

Compares two implementations against `variants.gather` (float32 reference):
- the project's own fused kernel, `mlx_upsample.kernels.bilinear.upsample`
  (always run; failures here are reported, not swallowed).
- `mx.fast.upsample_bilinear`, only when the installed MLX build has it
  (skipped with a clear reason otherwise).
"""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_upsample.kernels.bilinear import upsample as fused
from mlx_upsample.variants import gather

HAS_FAST = hasattr(mx.fast, "upsample_bilinear")
IMPLS = [fused] + ([mx.fast.upsample_bilinear] if HAS_FAST else [])
IMPL_IDS = [f.__name__ if hasattr(f, "__name__") else "mx.fast.upsample_bilinear" for f in IMPLS]


def _id(fn):
    return getattr(fn, "__name__", "mx.fast.upsample_bilinear")


def R(*shape, dtype=mx.float32):
    return mx.random.normal(shape, key=mx.random.key(0)).astype(dtype)


def fwd_tol(dtype):
    return 1e-5 if dtype == mx.float32 else 2e-2


def check_forward(fn, x, scale_factor, align_corners=False):
    xf = x.astype(mx.float32)
    out = fn(x, scale_factor, align_corners)
    ref = gather(xf, scale_factor, align_corners)
    mx.eval(out, ref)
    assert out.shape == ref.shape
    if ref.size == 0:
        return out, ref
    err = mx.abs(out.astype(mx.float32) - ref).max().item()
    assert err <= fwd_tol(x.dtype), f"{_id(fn)} forward err {err:.3e}"
    return out, ref


def check_grad(fn, x, scale_factor, align_corners=False):
    xf = x.astype(mx.float32)
    ref = gather(xf, scale_factor, align_corners)
    if ref.size == 0:
        return
    cot = mx.random.normal(ref.shape, key=mx.random.key(1))
    g = mx.vjp(lambda a: fn(a, scale_factor, align_corners), [xf], [cot])[1][0]
    gr = mx.vjp(lambda a: gather(a, scale_factor, align_corners), [xf], [cot])[1][0]
    rel = (mx.abs(g - gr).max() / (mx.abs(gr).max() + 1e-12)).item()
    assert rel <= 1e-5, f"{_id(fn)} grad rel err {rel:.3e}"


# ---------------------------------------------------------------------------
# Forward + gradient cases: (name, array, scale_factor, align_corners)
# ---------------------------------------------------------------------------
FWD_GRAD_CASES = [
    ("C=1", R(2, 9, 7, 1), (2, 2), False),
    ("C=3", R(1, 9, 7, 3), (2, 2), False),
    ("C=33 (tail)", R(1, 9, 7, 33), (1.5, 1.5), False),
    ("C=1000", R(1, 4, 4, 1000), (2, 2), False),
    ("B=5", R(5, 6, 6, 4), (3, 3), False),
    ("H=1 W>1", R(1, 1, 17, 4), (4, 4), False),
    ("H=1 W>1 ac", R(1, 1, 17, 4), (4, 4), True),
    ("out size 1 ac", R(1, 5, 5, 4), (0.3, 0.3), True),
    ("out size 1", R(1, 5, 5, 4), (0.3, 0.3), False),
    ("mixed (0.5,3)", R(1, 16, 9, 4), (0.5, 3), False),
    ("mixed (3,0.5) ac", R(1, 9, 16, 4), (3, 0.5), True),
    ("scale 1.0 identity", R(1, 8, 8, 4), (1, 1), False),
    ("scale 1.0 ac", R(1, 8, 8, 4), (1, 1), True),
    ("odd trunc 7x11", R(1, 7, 11, 4), (1.7, 1.3), False),
    ("tiny scale 1/64", R(1, 256, 256, 2), (1 / 64, 1 / 64), False),
    ("huge W=8191 /3", R(1, 2, 8191, 2), (1, 1 / 3), False),
    ("huge W=8191 x2.7", R(1, 2, 8191, 2), (1, 2.7), False),
    ("W=65537 x1.1", R(1, 1, 65537, 1), (1, 1.1), False),
    ("W=65537 x1.1 ac", R(1, 1, 65537, 1), (1, 1.1), True),
    ("1x1 x255", R(1, 1, 1, 8), (255, 255), False),
]


@pytest.mark.parametrize("fn", IMPLS, ids=IMPL_IDS)
@pytest.mark.parametrize("name,x,sf,ac", FWD_GRAD_CASES, ids=[c[0] for c in FWD_GRAD_CASES])
def test_forward(fn, name, x, sf, ac):
    check_forward(fn, x, sf, ac)


@pytest.mark.parametrize("fn", IMPLS, ids=IMPL_IDS)
@pytest.mark.parametrize("name,x,sf,ac", FWD_GRAD_CASES, ids=[c[0] for c in FWD_GRAD_CASES])
def test_gradient(fn, name, x, sf, ac):
    check_grad(fn, x, sf, ac)


# ---------------------------------------------------------------------------
# Non-standard layouts: transposed, strided-sliced, broadcast.
# ---------------------------------------------------------------------------
_layout_base = R(1, 10, 12, 6)
LAYOUT_CASES = [
    ("transposed", _layout_base.transpose(0, 2, 1, 3), (2, 2)),
    ("strided slice", R(1, 20, 24, 8)[:, ::2, 1::3, ::2], (2, 2)),
    ("broadcast", mx.broadcast_to(R(1, 1, 1, 4), (2, 5, 5, 4)), (2, 2)),
]


@pytest.mark.parametrize("fn", IMPLS, ids=IMPL_IDS)
@pytest.mark.parametrize("name,x,sf", LAYOUT_CASES, ids=[c[0] for c in LAYOUT_CASES])
def test_forward_layout(fn, name, x, sf):
    check_forward(fn, x, sf)


@pytest.mark.parametrize("fn", IMPLS, ids=IMPL_IDS)
@pytest.mark.parametrize("name,x,sf", LAYOUT_CASES, ids=[c[0] for c in LAYOUT_CASES])
def test_gradient_layout(fn, name, x, sf):
    check_grad(fn, x, sf)


# ---------------------------------------------------------------------------
# dtypes: bf16, fp16 (forward + grad), uint8/bool (forward only, no grad).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fn", IMPLS, ids=IMPL_IDS)
def test_forward_bf16(fn):
    check_forward(fn, R(1, 9, 7, 4, dtype=mx.bfloat16), (2, 2))


@pytest.mark.parametrize("fn", IMPLS, ids=IMPL_IDS)
def test_gradient_bf16(fn):
    check_grad(fn, R(1, 9, 7, 4, dtype=mx.bfloat16), (2, 2))


@pytest.mark.parametrize("fn", IMPLS, ids=IMPL_IDS)
def test_forward_fp16_big_channels(fn):
    check_forward(fn, R(1, 8, 8, 512, dtype=mx.float16), (2, 2))


@pytest.mark.parametrize("fn", IMPLS, ids=IMPL_IDS)
def test_gradient_fp16_big_channels(fn):
    check_grad(fn, R(1, 8, 8, 512, dtype=mx.float16), (2, 2))


@pytest.mark.parametrize("fn", IMPLS, ids=IMPL_IDS)
def test_forward_uint8(fn):
    x = (mx.random.uniform(shape=(1, 6, 6, 3)) * 255).astype(mx.uint8)
    check_forward(fn, x, (2, 2))


@pytest.mark.parametrize("fn", IMPLS, ids=IMPL_IDS)
def test_forward_bool(fn):
    x = mx.random.uniform(shape=(1, 6, 6, 3)) > 0.5
    check_forward(fn, x, (2, 2))


# ---------------------------------------------------------------------------
# NaN/inf passthrough: bilinear blending spreads a NaN/inf source pixel to
# every output pixel that reads it, so compare exactly (NaN positions must
# match, finite values must match exactly) rather than with a tolerance.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fn", IMPLS, ids=IMPL_IDS)
def test_nan_inf_passthrough(fn):
    x = mx.array([[[[1.0], [float("inf")]], [[float("nan")], [2.0]]]])
    out = fn(x, (2, 2), False)
    ref = gather(x, (2, 2), False)
    mx.eval(out, ref)
    assert out.shape == ref.shape
    assert mx.array_equal(out, ref, equal_nan=True).item()


# ---------------------------------------------------------------------------
# Behaviour checks specific to mx.fast.upsample_bilinear (compile, vmap,
# second-order grad, explicit CPU stream, nn.Upsample with other ranks).
# ---------------------------------------------------------------------------
pytestmark_fast = pytest.mark.skipif(
    not HAS_FAST, reason="mx.fast.upsample_bilinear not available in this MLX build"
)


@pytestmark_fast
def test_fast_compile():
    up = mx.fast.upsample_bilinear
    x = R(1, 10, 12, 6)
    out = mx.compile(lambda a: up(a, (2, 2), False))(x)
    ref = up(x, (2, 2), False)
    assert mx.allclose(out, ref).item()


@pytestmark_fast
def test_fast_vmap_over_batch():
    up = mx.fast.upsample_bilinear
    x = R(1, 10, 12, 6)
    out = mx.vmap(lambda a: up(a[None], (2, 2), False)[0])(x)
    ref = up(x, (2, 2), False)
    assert mx.allclose(out, ref, atol=1e-5).item()


@pytestmark_fast
def test_fast_grad_of_grad_runs():
    up = mx.fast.upsample_bilinear
    x = R(1, 10, 12, 6)
    g = mx.grad(lambda a: mx.grad(lambda b: (up(b, (2, 2), False) ** 2).sum())(a).sum())(x)
    mx.eval(g)
    assert g.shape == x.shape


@pytestmark_fast
def test_fast_cpu_stream_matches_gpu():
    up = mx.fast.upsample_bilinear
    x = R(1, 10, 12, 6)
    out_cpu = up(x, (2, 2), False, stream=mx.cpu)
    out_gpu = up(x, (2, 2), False)
    assert mx.allclose(out_cpu, out_gpu, atol=1e-5).item()


@pytestmark_fast
@pytest.mark.parametrize("ndim,shape", [(3, (1, 9, 4)), (5, (1, 4, 4, 4, 2))])
def test_fast_nn_upsample_other_ranks_unchanged(ndim, shape):
    x = R(*shape)
    out = nn.Upsample(2.0, mode="linear")(x)
    mx.eval(out)
    assert out.ndim == ndim
