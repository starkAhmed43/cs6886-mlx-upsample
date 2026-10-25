"""Timing, resumable JSONL result store, and environment metadata."""

import json
import math
import platform
import random
import subprocess
import time
from collections.abc import Callable
from dataclasses import fields
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
