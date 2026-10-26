"""Bilinear upsampling variants. Each is f(x, scale_factor, align_corners)."""

from collections.abc import Callable
from functools import lru_cache

import mlx.core as mx

from mlx_upsample.baselines import pr4418
from mlx_upsample.kernels.bilinear import upsample as fused

Variant = Callable[[mx.array, tuple[float, ...], bool], mx.array]


def gather(x, scale_factor, align_corners):
    return pr4418._interpolate(x, scale_factor, pr4418._linear_indices, align_corners)


@lru_cache
def _compiled(scale_factor, align_corners):
    return mx.compile(lambda x: gather(x, scale_factor, align_corners))


def gather_compiled(x, scale_factor, align_corners):
    return _compiled(tuple(scale_factor), align_corners)(x)


def matmul(x, scale_factor, align_corners):
    return pr4418._interpolate_matmul(
        x, scale_factor, pr4418._linear_indices, align_corners
    )


def pr4418_shipped(x, scale_factor, align_corners):
    return pr4418.upsample_linear(x, scale_factor, align_corners)


VARIANTS: dict[str, Variant] = {
    "gather": gather,
    "gather_compiled": gather_compiled,
    "matmul": matmul,
    "pr4418": pr4418_shipped,
    "fused": fused,
}
