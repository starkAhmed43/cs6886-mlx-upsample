"""Timing, resumable JSONL result store, environment metadata, and the sweep loop."""

import json
import math
import platform
import random
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

from mlx_upsample.workloads import Workload

KEY_FIELDS = tuple(f.name for f in fields(Workload)) + ("variant", "pass")


def time_fn(fn: Callable[[], object], warmup: int = 3, repeats: int = 5, iters: int = 20) -> list[float]:
    """Mean ms per call for each repeat. `fn` must evaluate its own result."""
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(repeats):
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        out.append((time.perf_counter() - t) * 1e3 / iters)
    return out


def time_interleaved(
    fns: dict[str, Callable[[], object]],
    repeats: int = 7,
    min_ms: float = 50.0,
    warmup: int = 3,
    seed: int = 0,
) -> dict[str, list[float]]:
    """Mean ms per call per repeat; shuffled interleaving makes all fns see the same machine state."""
    iters = {}
    for name, fn in fns.items():
        for _ in range(warmup):
            fn()
        t = time.perf_counter()
        fn()
        one_ms = max((time.perf_counter() - t) * 1e3, 1e-6)
        iters[name] = min(1000, max(1, math.ceil(min_ms / one_ms)))
    out = {name: [] for name in fns}
    for r in range(repeats):
        names = list(fns)
        random.Random(seed + r).shuffle(names)
        for name in names:
            t = time.perf_counter()
            for _ in range(iters[name]):
                fns[name]()
            out[name].append((time.perf_counter() - t) * 1e3 / iters[name])
    return out


def make_fwd(f: Callable, x: mx.array, scale_factor: tuple, align_corners: bool) -> Callable[[], mx.array]:
    def run() -> mx.array:
        y = f(x, scale_factor, align_corners)
        mx.eval(y)
        return y

    return run


def make_fwd_bwd(f: Callable, x: mx.array, scale_factor: tuple, align_corners: bool, seed: int = 0) -> Callable[[], tuple]:
    y = f(x, scale_factor, align_corners)
    cot = mx.random.normal(y.shape, key=mx.random.key(seed)).astype(y.dtype)
    mx.eval(cot)

    def run() -> tuple:
        out, grads = mx.vjp(lambda a: f(a, scale_factor, align_corners), [x], [cot])
        mx.eval(out, grads)
        return out, grads

    return run


def row_key(row: dict) -> tuple:
    return tuple(row[k] for k in KEY_FIELDS)


class ResultStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keys: set[tuple] = set()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    self.keys.add(row_key(json.loads(line)))
                except (json.JSONDecodeError, KeyError):
                    pass  # truncated last line from a crash

    def has(self, key: tuple) -> bool:
        return key in self.keys

    def append(self, row: dict) -> None:
        with self.path.open("a") as f:
            if f.tell() and not self._ends_with_newline():
                f.write("\n")
            f.write(json.dumps(row) + "\n")
            f.flush()
        self.keys.add(row_key(row))

    def _ends_with_newline(self) -> bool:
        with self.path.open("rb") as f:
            f.seek(-1, 2)
            return f.read(1) == b"\n"


def vjp_grad(f: Callable, x: mx.array, sf: tuple, ac: bool, cot: mx.array) -> mx.array:
    return mx.vjp(lambda a: f(a, sf, ac), [x], [cot.astype(f(x, sf, ac).dtype)])[1][0]


def maxdiff(a: mx.array, b: mx.array) -> float:
    return mx.abs(a.astype(mx.float32) - b).max().item()


def check(w: Workload, x: mx.array, f: Callable, pas: str, ref: dict, sf: tuple) -> dict:
    """Shape, dtype and (for fwd_bwd) gradient check of variant `f` against `ref`."""
    y = f(x, sf, w.align_corners)
    diff = maxdiff(y, ref["y"])
    valid = diff <= (1e-5 if w.dtype == "float32" else 1e-2)
    row = {"out_shape": list(y.shape), "max_diff": diff}
    if pas == "fwd_bwd":
        gdiff = maxdiff(vjp_grad(f, x, sf, w.align_corners, ref["cot"]), ref["g"])
        row |= {"grad_max_diff": gdiff, "grad_rel_err": gdiff / ref["gmax"] if ref["gmax"] else 0.0}
        valid &= w.dtype != "float32" or row["grad_rel_err"] <= 1e-5  # float16 gradient error is recorded, not gated
    return row | {"valid": bool(valid)}


def write_meta(path: Path, args: dict, variant_names: list, seed: int) -> None:
    path.write_text(json.dumps(
        {**env_metadata(), "args": {k: str(v) for k, v in args.items()}, "variants": list(variant_names), "seed": seed},
        indent=2,
    ))


def run_sweep(
    store: ResultStore,
    workloads: list[Workload],
    variants: dict[str, Callable],
    passes: list[str],
    print_progress: bool = True,
) -> None:
    """Time and check every (workload, pass, variant) not already in `store`; append rows as they finish."""
    from mlx_upsample.workloads import make_input

    total, n = len(workloads) * len(passes) * len(variants), 0
    for i, w in enumerate(workloads):
        sf = (w.scale, w.scale)
        x = make_input(w)
        for pas in passes:
            todo = [v for v in variants if not store.has(row_key({**asdict(w), "variant": v, "pass": pas}))]
            n += len(variants) - len(todo)
            if not todo:
                continue
            rows, ref = {}, {}
            try:
                y = variants["gather"](x, sf, w.align_corners)
                ref["y"] = y.astype(mx.float32)
                if pas == "fwd_bwd":
                    ref["cot"] = mx.random.normal(y.shape, key=mx.random.key(0)).astype(mx.float32)
                    ref["g"] = vjp_grad(variants["gather"], x.astype(mx.float32), sf, w.align_corners, ref["cot"])
                    ref["gmax"] = mx.abs(ref["g"]).max().item()
                mx.eval(ref)
            except Exception as e:  # reference failure: every variant gets an error row
                ref = {"error": f"reference: {e!r}"}
            fns = {}
            for v in todo:
                try:
                    if "error" in ref:
                        raise RuntimeError(ref["error"])
                    rows[v] = check(w, x, variants[v], pas, ref, sf)
                    fns[v] = (make_fwd if pas == "fwd" else make_fwd_bwd)(variants[v], x, sf, w.align_corners)
                except Exception as e:
                    rows[v] = {"out_shape": None, "error": repr(e), "valid": False}
            if fns:
                try:
                    times = time_interleaved(fns, seed=i)
                except Exception as e:
                    times = {}
                    for v in fns:
                        rows[v] |= {"error": f"timing: {e!r}", "valid": False}
                for v, ms in times.items():
                    rows[v] |= {"ms_median": sorted(ms)[len(ms) // 2], "ms_repeats": ms}
            for v in todo:
                n += 1
                row = {**asdict(w), "variant": v, "pass": pas, **rows[v]}
                store.append(row)
                if print_progress:
                    print(f"{n}/{total} {w.name} B={w.B} {w.dtype} {v} {pas} "
                          f"{row.get('ms_median', float('nan')):.3f} ms{'' if row['valid'] else ' INVALID'}", flush=True)
        x = ref = y = fns = rows = None
        mx.clear_cache()


def _run(*cmd: str) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def env_metadata() -> dict:
    return {
        "chip": _run("sysctl", "-n", "machdep.cpu.brand_string"),
        "macos": platform.mac_ver()[0],
        "mlx": mx.__version__,
        "python": platform.python_version(),
        "git_commit": _run("git", "rev-parse", "HEAD"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
