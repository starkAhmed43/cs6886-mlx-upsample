import mlx.core as mx
import pytest

from mlx_upsample.baselines import pr4418
from mlx_upsample.variants import VARIANTS

OTHERS = [name for name in VARIANTS if name != "gather"]
SCALES = [2, 4, 1.5, 0.5, 0.25]
# (B, H, W, C, scale): covers each dimension without the full product.
CASES = [(1, 16, 16, 4, s) for s in SCALES] + [
    (2, 7, 5, 1, 2),
    (2, 7, 5, 32, 1.5),
    (1, 7, 5, 4, 0.5),
    (2, 16, 16, 32, 4),
    (1, 16, 16, 1, 0.25),
]


def make(B, H, W, C, dtype=mx.float32):
    return mx.random.normal((B, H, W, C)).astype(dtype)


@pytest.mark.parametrize("name", OTHERS)
@pytest.mark.parametrize("align_corners", [True, False])
@pytest.mark.parametrize("dtype,atol", [(mx.float32, 1e-5), (mx.float16, 1e-2)])
@pytest.mark.parametrize("B,H,W,C,scale", CASES)
def test_matches_gather(name, align_corners, dtype, atol, B, H, W, C, scale):
    x = make(B, H, W, C, dtype)
    sf = (scale, scale)
    ref = VARIANTS["gather"](x, sf, align_corners)
    out = VARIANTS[name](x, sf, align_corners)
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    assert mx.abs(out - ref).max().item() <= atol


@pytest.mark.parametrize("name", OTHERS)
@pytest.mark.parametrize("align_corners", [True, False])
def test_gradient_matches_gather(name, align_corners):
    x = make(2, 7, 5, 4)
    sf = (2, 2)
    cot = mx.random.normal(VARIANTS["gather"](x, sf, align_corners).shape)
    grads = {
        n: mx.vjp(lambda a, f=VARIANTS[n]: f(a, sf, align_corners), [x], [cot])[1][0]
        for n in ("gather", name)
    }
    assert mx.abs(grads[name] - grads["gather"]).max().item() <= 1e-4


@pytest.mark.parametrize("name", list(VARIANTS))
@pytest.mark.parametrize("align_corners", [True, False])
def test_gradient_float16_close_to_float32_reference(name, align_corners):
    x = make(2, 7, 5, 4, mx.float16)
    sf = (2, 2)
    cot = mx.random.normal((2, 14, 10, 4)).astype(mx.float16)
    ref = mx.vjp(
        lambda a: VARIANTS["gather"](a, sf, align_corners),
        [x.astype(mx.float32)],
        [cot.astype(mx.float32)],
    )[1][0]
    grad = mx.vjp(lambda a: VARIANTS[name](a, sf, align_corners), [x], [cot])[1][0]
    err = mx.abs(grad.astype(mx.float32) - ref).max().item()
    assert err <= 5e-2 * mx.abs(ref).max().item()


@pytest.mark.parametrize(
    "size,scale,expected",
    [(128, 2, "_interpolate_matmul"), (129, 2, "_interpolate"), (64, 0.5, "_interpolate")],
)
def test_pr4418_follows_gate(monkeypatch, size, scale, expected):
    calls = []
    for fn in ("_interpolate_matmul", "_interpolate"):
        orig = getattr(pr4418, fn)

        def wrapper(*args, _orig=orig, _name=fn, **kwargs):
            calls.append(_name)
            return _orig(*args, **kwargs)

        monkeypatch.setattr(pr4418, fn, wrapper)
    VARIANTS["pr4418"](make(1, size, size, 1), (scale, scale), False)
    assert calls == [expected]
