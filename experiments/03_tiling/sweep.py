"""Time gather/fused/fused_tiled (forward only) on upscale workloads.

Tiling only has input reuse to exploit when scale > 1 (see README), so this
sweep only covers upscale workloads from the grid and real catalogs.
"""

import argparse
import random
import sys
from pathlib import Path

import mlx.core as mx

from mlx_upsample.bench import ResultStore, run_sweep, write_meta
from mlx_upsample.kernels.bilinear import fused, fused_tiled
from mlx_upsample.variants import gather
from mlx_upsample.workloads import grid, out_bytes, real

RESULTS = Path(__file__).parent / "results"
SEED = 0
VARIANTS = {"gather": gather, "fused": fused, "fused_tiled": fused_tiled}
SMOKE_GRID = [(64, 2, 4, "float32"), (128, 2, 32, "float16"), (192, 4, 4, "float32")]
SMOKE_REAL = {"rife.up2", "rife.up4"}


def _apply_cap(ws: list, max_mb: int) -> list:
    capped = [w for w in ws if out_bytes(w) <= max_mb * 2**20]
    print(f"--max-mb={max_mb}: dropped {len(ws) - len(capped)} of {len(ws)} workloads")
    return capped


def pick_workloads(smoke: bool, max_mb: int) -> list:
    ws = [w for w in grid() + real() if w.scale > 1]
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

    run_sweep(store, ws, VARIANTS, ["fwd"])


if __name__ == "__main__":
    sys.exit(main())
