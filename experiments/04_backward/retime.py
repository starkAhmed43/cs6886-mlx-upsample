"""Re-time the sweep's two losses in a fresh process, without the reference gather buffer.

The sweep times `fused` and `matmul` together with `gather` and `pr4418`, and builds a float32
`gather` reference for correctness checks. For the largest workload (grid.h1536.s0.25.c256,
2.4 GB in + 2.4 GB grad) that extra memory pressure changes the result: this script re-times
only `fused` and `matmul`, with no reference built, to check whether the sweep's loss there was
a memory artifact.
"""

import json
import subprocess
from pathlib import Path

import mlx.core as mx

from mlx_upsample.bench import make_fwd_bwd, time_interleaved
from mlx_upsample.variants import VARIANTS
from mlx_upsample.workloads import grid, make_input, real

RESULTS = Path(__file__).parent / "results" / "retime.jsonl"
PICK = {"fused": VARIANTS["fused"], "matmul": VARIANTS["matmul"]}


def _swap() -> dict:
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True, check=True).stdout.strip()
    return {"vm.swapusage": out.split(": ", 1)[1] if ": " in out else out}


def _workload(all_ws: list, name: str, B: int, dtype: str):
    for w in all_ws:
        if w.name == name and w.B == B and w.dtype == dtype:
            return w
    raise KeyError((name, B, dtype))


def retime(w) -> dict:
    sf = (w.scale, w.scale)
    x = make_input(w)
    before = _swap()
    fns = {name: make_fwd_bwd(f, x, sf, w.align_corners) for name, f in PICK.items()}
    times = time_interleaved(fns, repeats=5)
    after = _swap()
    x = fns = None
    mx.clear_cache()
    row = {
        "name": w.name, "B": w.B, "dtype": w.dtype,
        "swap_before": before["vm.swapusage"], "swap_after": after["vm.swapusage"],
        **{f"{v}_ms_median": sorted(ms)[len(ms) // 2] for v, ms in times.items()},
        **{f"{v}_ms_repeats": ms for v, ms in times.items()},
    }
    print(f"{w.name} B={w.B} {w.dtype}: " + ", ".join(f"{v} {row[f'{v}_ms_median']:.3f} ms" for v in PICK))
    return row


def main():
    mx.set_cache_limit(1 << 30)
    workloads = [
        _workload(grid(capped=False), "grid.h1536.s0.25.c256", 1, "float32"),
        _workload(real(capped=False), "pspnet.bin1", 1, "float16"),
    ]
    rows = [retime(w) for w in workloads]
    RESULTS.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
