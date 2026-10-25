"""Fused bilinear upsampling for MLX, written as one custom Metal GPU kernel.

Run `python bilinear.py` to compare it with `nn.Upsample` on four test cases.
"""
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.upsample import _linear_indices

# The kernel body lives in bilinear.metal, next to this file.
source = (Path(__file__).parent / "bilinear.metal").read_text()

# Compile the Metal source into a kernel that Python can call.
# input_names and output_names become the buffer names used in the source.
# By default MLX makes inputs row-contiguous, so the flat-index math is valid.
kernel = mx.fast.metal_kernel(
    name="bilinear_upsample",
    input_names=["x", "y_l", "y_r", "y_w", "x_l", "x_r", "x_w"],
    output_names=["out"],
    source=source,
)

def axis_tables(n, s, align_corners):
    # Build lookup tables for one axis (height or width).
    # We reuse MLX's own _linear_indices (ndims=1). It returns
    # ((left_idx, 1 - t), (right_idx, t)).
    # Reusing it makes align_corners and half-pixel rules match stock MLX exactly.
    # We drop "1 - t" because the kernel computes it itself.
    (l, wl), (r, w) = _linear_indices(n, s, align_corners, 0, 1)
    return l.reshape(-1), r.reshape(-1), w.reshape(-1)

def _launch(x: mx.array, tables: tuple) -> mx.array:
    """Run the compiled kernel given a ready (yl, yr, yw, xl, xr, xw) table set."""
    B, H, W, C = x.shape
    yl, yr, yw, xl, xr, xw = tables
    Ho, Wo = yl.size, xl.size
    # Stock nn.Upsample always computes in float32, so a float16 input still
    # comes out float32. Match that here: T is the input dtype (used to read
    # x), Tout is the output dtype the kernel casts and writes to.
    out_dtype = mx.result_type(x.dtype, mx.float32)
    return kernel(
        inputs=[x, yl, yr, yw, xl, xr, xw],
        # template values become compile-time constants in the Metal code
        # (T = input dtype, Tout = output dtype, the rest are sizes).
        # MLX compiles and caches one kernel per new combination.
        # So the first call with a new shape or dtype is slower.
        # Warm up before timing.
        template=[("T", x.dtype), ("Tout", out_dtype), ("C", C), ("H_in", H), ("W_in", W),
                  ("H_out", Ho), ("W_out", Wo)],
        # grid = TOTAL threads per axis (not the number of teams).
        # One thread per output value: (channels, columns, batch * rows).
        grid=(C, Wo, B * Ho),
        # threadgroup = team size: up to 32 channels x 8 columns = 256 threads.
        # Threads in a team can share fast threadgroup memory.
        # That is the tiling idea for later. This kernel does not use it yet.
        threadgroup=(min(C, 32), 8, 1),
        # Tell MLX what to allocate for the output.
        output_shapes=[(B, Ho, Wo, C)],
        output_dtypes=[out_dtype],
    )[0]  # the kernel returns a list of outputs; we have one.


def fused_uncached(x: mx.array, scale_factor: tuple[float, float], align_corners: bool = False) -> mx.array:
    """Bilinear upsampling, same signature as the variants in variants.py.

    x has shape (B, H, W, C): batch, height, width, channels.
    scale_factor is (height_scale, width_scale): each axis gets its own table.
    The kernel indexes with 32-bit integers, so the output must stay well
    under 2**32 elements. The workloads here cap output at 512 MB.

    This rebuilds the index/weight tables on every call. See `fused` for a
    version that reuses them across calls with the same shape.
    """
    B, H, W, C = x.shape
    h_scale, w_scale = scale_factor
    yl, yr, yw = axis_tables(H, h_scale, align_corners)
    xl, xr, xw = axis_tables(W, w_scale, align_corners)
    return _launch(x, (yl, yr, yw, xl, xr, xw))


@lru_cache(maxsize=128)
def _tables_cached(H: int, W: int, scale_factor: tuple[float, float], align_corners: bool) -> tuple:
    # Same tables as `fused_uncached`, but built once per (H, W, scale,
    # align_corners) and cached. mx.eval forces them to exist on the GPU
    # before the kernel call, so the cost of building them is paid once, not
    # every call. lru_cache default maxsize (128) is enough here: we never
    # need more than a handful of distinct shapes at once.
    h_scale, w_scale = scale_factor
    yl, yr, yw = axis_tables(H, h_scale, align_corners)
    xl, xr, xw = axis_tables(W, w_scale, align_corners)
    mx.eval(yl, yr, yw, xl, xr, xw)
    return yl, yr, yw, xl, xr, xw


def fused(x: mx.array, scale_factor: tuple[float, float], align_corners: bool = False) -> mx.array:
    """Bilinear upsampling, same signature as the variants in variants.py.

    Same kernel as `fused_uncached`, but the index/weight tables come from a
    cache keyed on (H, W, scale_factor, align_corners) instead of being
    rebuilt per call. This is the default: see `fused_uncached` for the
    version without the cache.
    """
    B, H, W, C = x.shape
    tables = _tables_cached(H, W, tuple(scale_factor), align_corners)
    return _launch(x, tables)

if __name__ == "__main__":
    # Compare with nn.Upsample on four cases: x2, x4, x1.5 with align_corners=True,
    # and x0.5 (downscale). Differences near 1e-7 are float32 rounding.
    for shape, s, ac in [((2, 5, 7, 3), 2, False), ((1, 192, 320, 4), 4, False),
                         ((1, 9, 6, 64), 1.5, True), ((1, 32, 32, 4), 0.5, False)]:
        x = mx.random.normal(shape)
        ref = nn.Upsample(scale_factor=s, mode="linear", align_corners=ac)(x)
        print(shape, s, ac, "max diff:", mx.abs(fused(x, (s, s), ac) - ref).max().item())
