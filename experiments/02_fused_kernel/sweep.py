"""Time gather/matmul/pr4418/fused (forward only) on the grid and real workloads.

Appends rows to results/sweep.jsonl. The fused kernel has no backward pass yet,
so only the "fwd" pass is timed; see 01_crossover for fwd_bwd timing of the
other variants.
"""

import argparse
import random
import sys
from pathlib import Path

from mlx_upsample.bench import ResultStore, run_sweep, write_meta
from mlx_upsample.kernels.bilinear import fused_uncached
from mlx_upsample.variants import VARIANTS as STOCK_VARIANTS
from mlx_upsample.workloads import grid, real

RESULTS = Path(__file__).parent / "results"
SEED = 0
# Recorded results used the uncached kernel under the name "fused"; keep that
# name here so results/sweep.jsonl stays reproducible.
VARIANTS = {**STOCK_VARIANTS, "fused": fused_uncached}
SMOKE_GRID = [(64, 2, 4, "float32"), (128, 2, 32, "float16"), (192, 4, 4, "float32"), (64, 0.5, 32, "float32"), (256, 0.25, 4, "float16")]
SMOKE_REAL = {"pspnet.bin6", "pspnet.bin2", "upernet.fpn16"}


def pick_workloads(smoke: bool) -> list:
    ws = grid() + real()
    if smoke:
        ws = [w for w in grid() if (w.H, w.scale, w.C, w.dtype) in SMOKE_GRID] + [
            w for w in real() if w.name in SMOKE_REAL and w.B == 1
        ]
    random.Random(SEED).shuffle(ws)
    return ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    out = args.out or RESULTS / ("scratch/sweep.jsonl" if args.smoke else "sweep.jsonl")
    store = ResultStore(out)
    ws = pick_workloads(args.smoke)
    write_meta(out.parent / "meta.json", vars(args), list(VARIANTS), SEED)

    run_sweep(store, ws, VARIANTS, ["fwd"])


if __name__ == "__main__":
    sys.exit(main())
