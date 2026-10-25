import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import mlx.core as mx
import pandas as pd

from mlx_upsample.variants import VARIANTS
from mlx_upsample.workloads import Workload

EXP = Path(__file__).parents[1] / "experiments/01_crossover"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sweep, confirm = load("sweep"), load("confirm")
SF = (2.0, 2.0)


def setup(dtype):
    w = Workload("t", "test", "grid", 1, 6, 6, 3, 2.0, False, dtype)
    x = mx.random.normal((1, 6, 6, 3)).astype(getattr(mx, dtype))
    y = VARIANTS["gather"](x, SF, False)
    cot = mx.random.normal(y.shape).astype(mx.float32)
    g = sweep.vjp_grad(VARIANTS["gather"], x.astype(mx.float32), SF, False, cot)
    ref = {"y": y.astype(mx.float32), "cot": cot, "g": g, "gmax": mx.abs(g).max().item()}
    return w, x, ref


def run(dtype, pas="fwd_bwd", dy=0.0, dg=0.0, gmax=None):
    w, x, ref = setup(dtype)
    ref["y"] = ref["y"] + dy
    ref["g"] = ref["g"] + dg
    ref["gmax"] = gmax or ref["gmax"]
    return sweep.check(w, x, "matmul", pas, ref, SF)


def test_forward_gate():
    assert run("float32", "fwd")["valid"]
    assert not run("float32", "fwd", dy=1e-3)["valid"]
    assert run("float16", "fwd", dy=1e-3)["valid"]
    assert not run("float16", "fwd", dy=1e-1)["valid"]


def test_float32_gradient_gate_is_relative():
    assert run("float32")["valid"]
    assert not run("float32", dg=1e-3)["valid"]  # relative error about 1e-3
    assert run("float32", dg=1e-3, gmax=1e6)["valid"]  # absolute 1e-3, relative 1e-9


def test_float16_gradient_recorded_not_gated():
    row = run("float16", dg=1.0)
    assert row["valid"] and row["grad_rel_err"] > 0.01 and row["grad_max_diff"] > 0.5


def test_grad_fields_only_for_fwd_bwd():
    assert "grad_max_diff" not in run("float32", "fwd") and "grad_rel_err" not in run("float32", "fwd")
    assert {"grad_max_diff", "grad_rel_err"} <= run("float32").keys()


def test_confirm_targets(tmp_path, monkeypatch):
    def co(first, last, scale, C, status="partial"):
        return {"pass": "fwd", "scale": scale, "C": C, "dtype": "float32", "first_win_H": first, "last_win_H": last, "status": status}

    pd.DataFrame([
        co(64, 512, 2, 4),  # last 512 and next 768; first 64 and below 32
        co(32, 128, 2, 32),  # first is the smallest size: only last and next
        co(None, None, 2, 256, "never"),  # skipped
        co(64, 512, 3, 4),  # next size up (768) errored in the sweep
    ]).to_csv(tmp_path / "crossover.csv", index=False)
    bad = {"name": "grid.h768.s3.c4", "dtype": "float32", "B": 1, "pass": "fwd", "valid": False, "error": "OOM"}
    (tmp_path / "sweep.jsonl").write_text(json.dumps(bad) + "\n")
    monkeypatch.setattr(confirm, "RESULTS", tmp_path)
    got = {(w.name, p) for w, p in confirm.targets(Namespace(smoke=False, names=None)) if w.set == "grid"}
    assert got == {
        *((f"grid.h{h}.s2.c4", "fwd") for h in (32, 64, 512, 768)),
        *((f"grid.h{h}.s2.c32", "fwd") for h in (128, 192)),
        *((f"grid.h{h}.s3.c4", "fwd") for h in (32, 64, 512)),
    }
