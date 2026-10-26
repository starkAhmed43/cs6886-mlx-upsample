"""Time forward+backward ("fwd_bwd") for gather/matmul/pr4418/fused, upscale and downscale.

Unlike 03_tiling (forward only, scale > 1), the fused kernel's hand-written VJP has no
input-reuse restriction, so this sweep covers every scale in grid() and real().
`mlx_upsample.bench.check` already records grad_max_diff and grad_rel_err against a
float32 `gather` reference for every variant, so float16 gradient error of stock gather
and of fused are both recorded here.
"""

import argparse
import random
import sys
from pathlib import Path

import mlx.core as mx

from mlx_upsample.bench import ResultStore, run_sweep, write_meta
from mlx_upsample.kernels.bilinear import upsample
from mlx_upsample.variants import gather, matmul
from mlx_upsample.variants import pr4418_shipped as pr4418
from mlx_upsample.workloads import grid, out_bytes, real

RESULTS = Path(__file__).parent / "results"
SEED = 0
VARIANTS = {"gather": gather, "matmul": matmul, "pr4418": pr4418, "fused": upsample}
SMOKE_GRID = [(64, 0.5, 4, "float32"), (128, 2, 32, "float16"), (192, 4, 4, "float32")]
SMOKE_REAL = {"rife.up2", "rife.up4"}


def _apply_cap(ws: list, max_mb: int) -> list:
    capped = [w for w in ws if out_bytes(w) <= max_mb * 2**20]
    print(f"--max-mb={max_mb}: dropped {len(ws) - len(capped)} of {len(ws)} workloads")
    return capped


def pick_workloads(smoke: bool, max_mb: int) -> list:
    ws = grid() + real()
    if smoke:
        ws = [w for w in grid() if (w.H, w.scale, w.C, w.dtype) in SMOKE_GRID] + [
            w for w in real() if w.name in SMOKE_REAL and w.B == 1
        ]
    ws = _apply_cap(ws, max_mb)
    random.Random(SEED).shuffle(ws)
    return ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--max-mb", type=int, default=256)
    args = ap.parse_args()

    # MLX keeps freed buffers for reuse; capping the cache returns memory to the OS while still
    # letting a 256 MB output reuse its buffer between timing iterations.
    mx.set_cache_limit(1 << 30)

    out = args.out or RESULTS / ("scratch/sweep.jsonl" if args.smoke else "sweep.jsonl")
    store = ResultStore(out)
    ws = pick_workloads(args.smoke, args.max_mb)
    write_meta(out.parent / "meta.json", vars(args), list(VARIANTS), SEED)

    run_sweep(store, ws, VARIANTS, ["fwd_bwd"])


if __name__ == "__main__":
    sys.exit(main())
