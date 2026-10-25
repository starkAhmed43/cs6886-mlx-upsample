import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

EXP = Path(__file__).parents[1] / "experiments/01_crossover"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


confirm = load("confirm")


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
