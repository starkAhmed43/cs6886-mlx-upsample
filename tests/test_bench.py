import time

import mlx.core as mx

from mlx_upsample.bench import (
    ResultStore,
    check,
    env_metadata,
    make_fwd,
    make_fwd_bwd,
    row_key,
    run_sweep,
    time_fn,
    time_interleaved,
    vjp_grad,
)
from mlx_upsample.variants import VARIANTS
from mlx_upsample.workloads import Workload, grid


def _row(variant="gather", pass_="fwd"):
    return {**grid()[0].__dict__, "variant": variant, "pass": pass_, "ms_median": 1.0}


def test_store_skips_existing_and_survives_reopen(tmp_path):
    path = tmp_path / "a" / "r.jsonl"
    store = ResultStore(path)
    row = _row()
    assert not store.has(row_key(row))
    store.append(row)
    assert store.has(row_key(row))
    reopened = ResultStore(path)
    assert reopened.has(row_key(row))
    assert not reopened.has(row_key(_row("matmul")))


def test_store_tolerates_truncated_last_line(tmp_path):
    path = tmp_path / "r.jsonl"
    store = ResultStore(path)
    store.append(_row("gather"))
    with path.open("a") as f:
        f.write('{"name": "trunc')
    store = ResultStore(path)
    assert store.has(row_key(_row("gather")))
    store.append(_row("matmul"))
    assert len(ResultStore(path).keys) == 2


def test_time_fn_returns_repeats():
    t = time_fn(lambda: mx.eval(mx.ones((8,)) + 1), warmup=1, repeats=4, iters=3)
    assert len(t) == 4 and all(isinstance(v, float) and v > 0 for v in t)


def test_make_fwd_and_fwd_bwd():
    def f(x, s, ac):
        return x * s[0]

    x = mx.ones((1, 2, 2, 1))
    assert make_fwd(f, x, (2.0, 2.0), False)().shape == x.shape
    out, grads = make_fwd_bwd(f, x, (2.0, 2.0), False)()
    assert out[0].shape == x.shape and grads[0].shape == x.shape


def test_env_metadata():
    assert {"chip", "macos", "mlx", "python", "git_commit", "timestamp"} <= env_metadata().keys()


def test_time_interleaved():
    fast = lambda: mx.eval(mx.ones((8,)) + 1)
    res = time_interleaved({"a": fast, "b": fast}, repeats=3, min_ms=2, warmup=1)
    assert res.keys() == {"a", "b"}
    assert all(len(v) == 3 and all(x > 0 for x in v) for v in res.values())
    assert len(time_interleaved({"a": fast}, repeats=2, min_ms=2)["a"]) == 2


def test_time_interleaved_calibrates_iters():
    calls = []

    def slow():
        calls.append(1)
        time.sleep(0.001)

    time_interleaved({"s": slow}, repeats=1, min_ms=5, warmup=1)
    assert len(calls) - 2 >= 4  # minus warmup and calibration call


SF = (2.0, 2.0)


def setup_check(dtype):
    w = Workload("t", "test", "grid", 1, 6, 6, 3, 2.0, False, dtype)
    x = mx.random.normal((1, 6, 6, 3)).astype(getattr(mx, dtype))
    y = VARIANTS["gather"](x, SF, False)
    cot = mx.random.normal(y.shape).astype(mx.float32)
    g = vjp_grad(VARIANTS["gather"], x.astype(mx.float32), SF, False, cot)
    ref = {"y": y.astype(mx.float32), "cot": cot, "g": g, "gmax": mx.abs(g).max().item()}
    return w, x, ref


def run_check(dtype, pas="fwd_bwd", dy=0.0, dg=0.0, gmax=None):
    w, x, ref = setup_check(dtype)
    ref["y"] = ref["y"] + dy
    ref["g"] = ref["g"] + dg
    ref["gmax"] = gmax or ref["gmax"]
    return check(w, x, VARIANTS["matmul"], pas, ref, SF)


def test_forward_gate():
    assert run_check("float32", "fwd")["valid"]
    assert not run_check("float32", "fwd", dy=1e-3)["valid"]
    assert run_check("float16", "fwd", dy=1e-3)["valid"]
    assert not run_check("float16", "fwd", dy=1e-1)["valid"]


def test_float32_gradient_gate_is_relative():
    assert run_check("float32")["valid"]
    assert not run_check("float32", dg=1e-3)["valid"]  # relative error about 1e-3
    assert run_check("float32", dg=1e-3, gmax=1e6)["valid"]  # absolute 1e-3, relative 1e-9


def test_float16_gradient_recorded_not_gated():
    row = run_check("float16", dg=1.0)
    assert row["valid"] and row["grad_rel_err"] > 0.01 and row["grad_max_diff"] > 0.5


def test_grad_fields_only_for_fwd_bwd():
    assert "grad_max_diff" not in run_check("float32", "fwd") and "grad_rel_err" not in run_check("float32", "fwd")
    assert {"grad_max_diff", "grad_rel_err"} <= run_check("float32").keys()


def test_run_sweep_writes_rows_and_is_resumable(tmp_path, monkeypatch):
    monkeypatch.setattr("mlx_upsample.bench.time_interleaved", lambda fns, seed: {v: [1.0] for v in fns})
    w = grid()[0]
    variants = {"gather": VARIANTS["gather"], "matmul": VARIANTS["matmul"]}
    store = ResultStore(tmp_path / "sweep.jsonl")

    run_sweep(store, [w], variants, ["fwd"], print_progress=False)
    assert len(store.keys) == 2

    store2 = ResultStore(tmp_path / "sweep.jsonl")
    before = len(store2.keys)
    run_sweep(store2, [w], variants, ["fwd"], print_progress=False)
    assert len(store2.keys) == before
