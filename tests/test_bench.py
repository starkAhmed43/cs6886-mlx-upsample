import time

import mlx.core as mx

from mlx_upsample.bench import ResultStore, env_metadata, make_fwd, make_fwd_bwd, row_key, time_fn, time_interleaved
from mlx_upsample.workloads import grid


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
