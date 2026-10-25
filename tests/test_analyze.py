import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from mlx_upsample.baselines import pr4418

PATH = Path(__file__).parents[1] / "experiments/01_crossover/analyze.py"
spec = importlib.util.spec_from_file_location("analyze", PATH)
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


def grid_rows(times, scale=2.0, C=4, dtype="float32", p="fwd"):
    """times: {H: (gather_ms, matmul_ms)}"""
    return pd.DataFrame([
        {"pass": p, "scale": scale, "C": C, "dtype": dtype, "H": h, "W": h, "gather": g, "matmul": m,
         "gather_compiled": g, "name": f"g{h}", "set": "grid", "B": 1, "align_corners": False}
        for h, (g, m) in times.items()
    ])


def one(times):
    return analyze.crossover(grid_rows(times)).iloc[0]


def test_margin_win_tie_and_gather_win():
    r = one({32: (10, 5), 64: (10, 9.5), 128: (10, 9.0), 256: (5, 10)})
    # 5 < 9 wins; 9.5 and 9.0 are ties (9.0 is not below 0.9 x 10); gather wins at 256
    assert (r.n_wins, r.n_ties, r.n_sizes, r.max_H) == (1, 2, 4, 256)
    assert (r.first_win_H, r.last_win_H, r.status, r.contiguous) == (32, 32, "partial", True)


def test_always_never():
    a = one({32: (10, 1), 64: (10, 2)})
    assert (a.status, a.max_H) == ("always", 64)
    n = one({32: (10, 9.5), 64: (1, 10)})
    assert n.status == "never" and pd.isna(n.first_win_H)


def test_gaps_when_gather_wins_between_matmul_wins():
    r = one({32: (10, 1), 64: (1, 10), 128: (10, 1)})
    assert (r.first_win_H, r.last_win_H, r.contiguous) == (32, 128, False)


@pytest.mark.parametrize("scale,H,expected", [(0.5, 64, False), (2.0, 128, True), (2.0, 129, False), (1.0, 32, True)])
def test_gate_matches_pr_rule(scale, H, expected):
    w = grid_rows({H: (10, 10)}, scale=scale)
    gr = analyze.gate_regret(w).iloc[0]
    assert gr.gate_matmul == expected == pr4418._use_matmul((H, H), (scale, scale), 128)


def test_gate_regret_uses_chosen_method_and_tolerance():
    # gate picks gather (H=256): 10 vs matmul 5 -> wrong; 10 vs 9.5 -> within 10 percent
    gr = analyze.gate_regret(grid_rows({256: (10, 5)})).iloc[0]
    assert (gr.chosen, gr.regret_ms, gr.gate_wrong) == (10, 5, True)
    gr = analyze.gate_regret(grid_rows({256: (10, 9.5)})).iloc[0]
    assert not gr.gate_wrong


def test_duplicate_rows_fail():
    row = {k: 1 for k in analyze.UNIQUE}
    analyze.check_unique(pd.DataFrame([row]))
    with pytest.raises(ValueError):
        analyze.check_unique(pd.DataFrame([row, row]))


def test_confirm_winner_agreement(tmp_path):
    def rows(g, m, **kw):
        return [{"name": "a", "B": 1, "dtype": "float32", "pass": "fwd", "variant": v, "ms_median": ms, **kw}
                for v, ms in (("gather", g), ("matmul", m))]

    sweep = pd.DataFrame(rows(10, 5))  # matmul wins (r=2)
    p = tmp_path / "confirm.jsonl"
    # cold start: both variants 40% slower, same ratio; the last line is truncated
    p.write_text("\n".join(json.dumps(r) for r in rows(14, 7, run=0)) + '\n{"name": "a"')
    t = analyze.confirm_vs_sweep(sweep, analyze.read_confirm(p))
    w = analyze.winner_check(t).iloc[0]
    assert (w.winner_fresh, w.winner_sweep, w.agree, w.opposite) == ("matmul", "matmul", True, False)
    assert w.ratio_gap_pct == pytest.approx(0)
    opp = analyze.winner_check(analyze.confirm_vs_sweep(sweep, pd.DataFrame(rows(5, 10, run=0)))).iloc[0]
    assert opp.opposite and not opp.agree
    tie = analyze.winner_check(analyze.confirm_vs_sweep(sweep, pd.DataFrame(rows(10, 10, run=0)))).iloc[0]
    assert tie.winner_fresh == "tie" and not tie.agree and not tie.opposite
