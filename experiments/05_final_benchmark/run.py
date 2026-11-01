"""Time the user-facing nn.Upsample(mode="linear") path on the real-model catalog.

Three implementations, two conda envs:
  - stock:     nn.Upsample in the pip-mlx env (gather/matmul dispatch).
  - pr4418:    mlx_upsample.variants.pr4418_shipped, same env as stock.
  - fused_cpp: nn.Upsample in the from-source mlx env (dispatches to mx.fast.upsample_bilinear).

Each (env, workload, run) is timed in its own fresh subprocess, using the right conda env's
python, so GPU/driver state never leaks between runs (see experiments/01_crossover/confirm.py).
Rows are appended to results/runs.jsonl as they complete, so a crash only loses the in-flight row.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from mlx_upsample.bench import ResultStore, row_key
from mlx_upsample.workloads import Workload, out_bytes, real

RESULTS = Path(__file__).parent / "results"
PASSES = ("fwd", "fwd_bwd")
MLX_REPO = Path.home() / "GitHub/mlx"

STOCK_PY = "/Users/adhilahmed/miniconda3/envs/cs6886-mlx-upsample/bin/python"
DEV_PY = "/Users/adhilahmed/miniconda3/envs/cs6886-mlx-dev/bin/python"
ENVS = {
    "stock": {"python": STOCK_PY, "variants": ["stock", "pr4418"]},
    "fused_cpp": {"python": DEV_PY, "variants": ["fused_cpp"]},
}
SMOKE_NAMES = {"pspnet.bin1", "segformer_b0.mlp16", "upernet.fpn16"}


class RunStore(ResultStore):
    """Like ResultStore, but the resumability key also includes `run` (one row per run)."""

    def __init__(self, path):
        super().__init__(path)
        self.keys = set()
        for line in self.path.read_text().splitlines() if self.path.exists() else []:
            try:
                r = json.loads(line)
                self.keys.add((*row_key(r), r["run"]))
            except (json.JSONDecodeError, KeyError):
                pass  # truncated last line from a crash

    def append(self, row: dict) -> None:
        super().append(row)
        self.keys.add((*row_key(row), row["run"]))


def make_adapter(variant: str, scale_factor: tuple, align_corners: bool):
    """f(x, scale_factor, align_corners), built once per workload. Not timed."""
    if variant in ("stock", "fused_cpp"):
        import mlx.nn as nn

        layer = nn.Upsample(scale_factor=scale_factor, mode="linear", align_corners=align_corners)
        return lambda x, _sf, _ac: layer(x)
    if variant == "pr4418":
        from mlx_upsample.variants import pr4418_shipped

        return pr4418_shipped
    raise ValueError(variant)


def one(spec: dict) -> None:
    """Worker: time every variant of spec['env'] on one workload, for both passes, in this process."""
    import mlx.core as mx

    from mlx_upsample.bench import check, make_fwd, make_fwd_bwd, time_interleaved, vjp_grad
    from mlx_upsample.variants import gather
    from mlx_upsample.workloads import make_input

    w = Workload(**spec["w"])
    sf, ac = (w.scale, w.scale), w.align_corners
    x = make_input(w)
    variants = ENVS[spec["env"]]["variants"]
    adapters = {v: make_adapter(v, sf, ac) for v in variants}

    y = gather(x, sf, ac)
    ref_y = y.astype(mx.float32)
    mx.eval(ref_y)

    for pas in PASSES:
        ref = {"y": ref_y}
        if pas == "fwd_bwd":
            cot = mx.random.normal(y.shape, key=mx.random.key(0)).astype(mx.float32)
            g = vjp_grad(gather, x.astype(mx.float32), sf, ac, cot)
            mx.eval(cot, g)
            ref = {**ref, "cot": cot, "g": g, "gmax": mx.abs(g).max().item()}

        rows, fns = {}, {}
        for v in variants:
            try:
                rows[v] = check(w, x, adapters[v], pas, ref, sf)
                rows[v]["out_dtype"] = str(adapters[v](x, sf, ac).dtype)
                fns[v] = (make_fwd if pas == "fwd" else make_fwd_bwd)(adapters[v], x, sf, ac)
            except Exception as e:
                rows[v] = {"out_shape": None, "error": repr(e), "valid": False}
        if fns:
            try:
                times = time_interleaved(fns, seed=spec["run"])
            except Exception as e:
                times = {}
                for v in fns:
                    rows[v] |= {"error": f"timing: {e!r}", "valid": False}
            for v, ms in times.items():
                rows[v] |= {"ms_median": sorted(ms)[len(ms) // 2], "ms_repeats": ms}

        for v in variants:
            row = {**asdict(w), "variant": v, "pass": pas, "run": spec["run"], "env": spec["env"], **rows[v]}
            print(json.dumps(row), flush=True)

    x = ref_y = y = None
    mx.clear_cache()


def _apply_cap(ws: list, max_mb: int) -> list:
    capped = [w for w in ws if out_bytes(w) <= max_mb * 2**20]
    print(f"--max-mb={max_mb}: dropped {len(ws) - len(capped)} of {len(ws)} workloads")
    return capped


def pick_workloads(smoke: bool, max_mb: int) -> list:
    ws = real()
    if smoke:
        ws = [w for w in ws if w.name in SMOKE_NAMES and w.B == 1]
    return _apply_cap(ws, max_mb)


def row_keys(w: Workload, env: str, run: int) -> list:
    return [(*row_key({**asdict(w), "variant": v, "pass": p}), run) for v in ENVS[env]["variants"] for p in PASSES]


def env_order(i: int, run: int) -> list:
    names = list(ENVS)
    return names if (i + run) % 2 == 0 else list(reversed(names))


def write_meta(path: Path) -> None:
    meta = {}
    for name, cfg in ENVS.items():
        out = subprocess.run(
            [cfg["python"], "-c", "import json, mlx_upsample.bench as b; print(json.dumps(b.env_metadata()))"],
            capture_output=True, text=True, check=True,
        )
        meta[name] = json.loads(out.stdout.strip().splitlines()[-1])
    commit = subprocess.run(["git", "-C", str(MLX_REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(MLX_REPO), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    path.write_text(json.dumps({"envs": meta, "mlx_repo_commit": commit or None, "mlx_repo_dirty": dirty}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-mb", type=int, default=256)
    ap.add_argument("--one", help="internal: JSON spec of one (env, workload, run) to time")
    args = ap.parse_args()
    if args.one:
        return one(json.loads(args.one))

    runs = 2 if args.smoke else args.runs
    out = RESULTS / ("scratch/runs.jsonl" if args.smoke else "runs.jsonl")
    store = RunStore(out)
    ws = pick_workloads(args.smoke, args.max_mb)
    write_meta(out.parent / "meta.json")

    tasks = runs * len(ws) * len(ENVS)
    n = 0
    for run in range(runs):
        for i, w in enumerate(ws):
            for env in env_order(i, run):
                n += 1
                if all(store.has(k) for k in row_keys(w, env, run)):
                    continue
                spec = json.dumps({"env": env, "w": asdict(w), "run": run})
                res = subprocess.run([ENVS[env]["python"], __file__, "--one", spec], capture_output=True, text=True)
                for line in res.stdout.splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    store.append(row)
                    print(f"{n}/{tasks} {w.name} B={w.B} {w.dtype} {env} {row['variant']} {row['pass']} run {run} "
                          f"{row.get('ms_median', float('nan')):.3f} ms{'' if row.get('valid', True) else ' INVALID'}", flush=True)
                if res.returncode:
                    print(f"FAILED {w.name} {env} run {run}: {res.stderr[-400:]}")
                    continue


if __name__ == "__main__":
    sys.exit(main())
