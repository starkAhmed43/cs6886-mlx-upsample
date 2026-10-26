"""Does tiling only help once one input row no longer fits the GPU cache?

Hypothesis (see 03_tiling/README.md): `fused` only re-reads input from DRAM when
one input row (W_in * C * dtype_bytes) is bigger than the GPU's cache. Below
that size, repeat reads of a row should mostly hit the cache, so `fused` and
`fused_tiled` should both already be close to the bandwidth ceiling. Above
that size, repeat reads should miss the cache and go to DRAM, so tiling
(which avoids most of those repeat reads) should pull ahead.

To test this in isolation from total working-set size, every config here
holds input and output size fixed (32 MiB in / 128 MiB out at scale 2, 8 MiB
in / 128 MiB out at scale 4) and only reshapes the input: fewer, wider rows
vs. more, narrower rows. Only `row_bytes` (one row's size) changes.

`copy` is a minimal custom kernel that reads every input element exactly once
from DRAM (each output thread reads out[i] = x[i / R], so adjacent threads
share the same input element and the input is streamed through once, not
walked repeatedly) and writes the full output: the practical bandwidth
ceiling for this output size, independent of any upsampling math.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import pandas as pd

from mlx_upsample.bench import env_metadata, time_interleaved
from mlx_upsample.kernels.bilinear import fused, fused_tiled

RESULTS = Path(__file__).parent / "results"
C, B = 64, 1
MIB = 2**20
OUT_BYTES = 128 * MIB  # held fixed across every config

# row_bytes (bytes per input row = W_in * C * 4) to sweep, per scale.
ROW_BYTES = {
    2: [16 * 2**10, 64 * 2**10, 256 * 2**10, 1 * MIB, 4 * MIB, 16 * MIB],
    4: [16 * 2**10, 64 * 2**10, 256 * 2**10, 1 * MIB, 4 * MIB],
}

# A minimal custom kernel: one thread per output element, out[i] = x[i / R],
# where R = N_out / N_in is an exact integer (R copies of each input element,
# as many as N_out - N_in in config() guarantees). Dividing by R means
# adjacent output threads (i, i+1, ..., i+R-1) read the SAME input element,
# so each of the N_in input elements is read from DRAM exactly once overall
# (R-1 of every R reads hit the same cache line / register, not DRAM).
# `out[i % N_in]` would instead stride across the whole input once per
# output pass; at this input size (8-32 MiB) that walk is far bigger than
# the GPU cache and each of the s^2 passes is far apart in time, so the
# input gets re-read from DRAM s^2 times -- the opposite of a cache hit.
# No upsampling math at all -- this is the plain copy/broadcast bandwidth a
# kernel could ever reach for this much input + output traffic.
_COPY_SOURCE = """
    // thread_position_in_grid.x is this thread's flat output index (we
    // launch a 1D grid, one thread per output element).
    uint i = thread_position_in_grid.x;
    if (i >= N_out) return;
    // Integer-divide by R (not modulo!) so adjacent threads read the same
    // input element: the whole input is streamed through DRAM exactly once.
    // T and Tout are both float here, so this is a plain copy, not a cast.
    out[i] = static_cast<Tout>(x[i / R]);
"""
_copy_kernel = mx.fast.metal_kernel(name="copy_bandwidth", input_names=["x"], output_names=["out"], source=_COPY_SOURCE)


def make_copy(out_shape: tuple) -> callable:
    n_out = 1
    for d in out_shape:
        n_out *= d

    def run(x: mx.array) -> mx.array:
        n_in = x.size
        assert n_out % n_in == 0, f"N_out ({n_out}) must be an exact multiple of N_in ({n_in})"
        r = n_out // n_in
        return _copy_kernel(
            inputs=[x],
            template=[("T", x.dtype), ("Tout", mx.float32), ("N_in", n_in), ("N_out", n_out), ("R", r)],
            grid=(n_out, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[out_shape],
            output_dtypes=[mx.float32],
        )[0]

    return run


def configs() -> list[dict]:
    out = []
    for scale, row_bytes_list in ROW_BYTES.items():
        mib_per_row = (32 if scale == 2 else 8) * MIB  # scale**2 * H * row_bytes == OUT_BYTES
        for row_bytes in row_bytes_list:
            H = mib_per_row // row_bytes
            W = row_bytes // (C * 4)
            out.append({"scale": scale, "H": H, "W": W, "C": C, "row_bytes": row_bytes})
    return out


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["scale"], r["row_bytes"], r["variant"]))
            except (json.JSONDecodeError, KeyError):
                pass  # truncated last line from a crash
    return done


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def run_config(cfg: dict, path: Path, done: set) -> None:
    scale, H, W = cfg["scale"], cfg["H"], cfg["W"]
    todo = [v for v in ("fused", "fused_tiled", "copy") if (scale, cfg["row_bytes"], v) not in done]
    if not todo:
        return
    sf = (scale, scale)
    x = mx.random.normal((B, H, W, C))
    mx.eval(x)
    out_shape = (B, int(scale * H), int(scale * W), C)
    min_bytes = x.nbytes + OUT_BYTES

    y_fused = fused(x, sf, False)
    y_tiled = fused_tiled(x, sf, False)
    mx.eval(y_fused, y_tiled)
    max_diff = mx.abs(y_fused - y_tiled).max().item()

    fns = {}
    if "fused" in todo:
        fns["fused"] = lambda: mx.eval(fused(x, sf, False))
    if "fused_tiled" in todo:
        fns["fused_tiled"] = lambda: mx.eval(fused_tiled(x, sf, False))
    if "copy" in todo:
        copy_fn = make_copy(out_shape)
        fns["copy"] = lambda: mx.eval(copy_fn(x))

    times = time_interleaved(fns, seed=cfg["row_bytes"])
    for v, ms in times.items():
        ms_median = sorted(ms)[len(ms) // 2]
        row = {
            "scale": scale, "H": H, "W": W, "C": C, "row_bytes": cfg["row_bytes"],
            "min_bytes": min_bytes, "variant": v, "ms_median": ms_median, "ms_repeats": ms,
            "gbs": min_bytes / ms_median / 1e6, "max_diff": max_diff,
        }
        append_row(path, row)
        print(f"scale={scale} row_bytes={cfg['row_bytes']} {v} {ms_median:.4f} ms {row['gbs']:.1f} GB/s", flush=True)
    x = y_fused = y_tiled = None
    mx.clear_cache()


def analyze(path: Path) -> None:
    out_dir = path.parent
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_json(path, lines=True, dtype=False)
    wide = df.pivot_table(index=["scale", "row_bytes", "H", "W"], columns="variant", values=["ms_median", "gbs"]).reset_index()
    wide.columns = ["_".join(c).rstrip("_") if c[1] else c[0] for c in wide.columns]
    wide = wide.sort_values(["scale", "row_bytes"])

    lines = ["# Row-size summary\n", f"Source: `{path.name}`\n",
              "Tests whether `fused` only re-reads input from DRAM once one input row "
              "(W_in * C * 4 bytes) exceeds the GPU cache; `fused_tiled` avoids most of "
              "those re-reads, and `copy` is the practical bandwidth ceiling for this "
              "input/output traffic.\n"]
    for scale, g in wide.groupby("scale"):
        g = g.sort_values("row_bytes")
        lines.append(f"\n## scale={scale}\n")
        lines.append("| row_bytes | H | W | fused ms | fused_tiled ms | copy ms | "
                      "fused GB/s | fused_tiled GB/s | copy GB/s | fused/copy | fused/fused_tiled |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for _, r in g.iterrows():
            frac = r.get("gbs_fused", float("nan")) / r.get("gbs_copy", float("nan"))
            ratio = r.get("ms_median_fused", float("nan")) / r.get("ms_median_fused_tiled", float("nan"))
            lines.append(
                f"| {int(r.row_bytes):,} | {int(r.H)} | {int(r.W)} | "
                f"{r.get('ms_median_fused', float('nan')):.4f} | {r.get('ms_median_fused_tiled', float('nan')):.4f} | "
                f"{r.get('ms_median_copy', float('nan')):.4f} | "
                f"{r.get('gbs_fused', float('nan')):.1f} | {r.get('gbs_fused_tiled', float('nan')):.1f} | "
                f"{r.get('gbs_copy', float('nan')):.1f} | {frac:.3f} | {ratio:.3f} |"
            )
    maxdiff = df["max_diff"].max() if "max_diff" in df else float("nan")
    lines.append(f"\nMax abs diff, fused vs fused_tiled, across all configs: {maxdiff:.2e}\n")
    (out_dir / "rowsize_summary.md").write_text("\n".join(lines) + "\n")

    scales = sorted(wide.scale.unique())
    fig, axes = plt.subplots(1, len(scales), figsize=(6 * len(scales), 4.5), squeeze=False)
    for ax, scale in zip(axes[0], scales):
        g = wide[wide.scale == scale].sort_values("row_bytes")
        for v in ("fused", "fused_tiled", "copy"):
            col = f"gbs_{v}"
            if col in g:
                ax.plot(g.row_bytes, g[col], marker="o", label=v)
        ax.set_xscale("log")
        ax.set_xlabel("row_bytes (W_in * C * 4)")
        ax.set_ylabel("GB/s")
        ax.set_title(f"scale={scale}")
        ax.legend()
    fig.suptitle("Effective bandwidth vs. input row size (output fixed at 128 MiB)")
    fig.savefig(fig_dir / "rowsize.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir / 'rowsize_summary.md'} and {fig_dir / 'rowsize.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()

    mx.set_cache_limit(1 << 30)

    out = RESULTS / ("scratch/rowsize.jsonl" if args.smoke else "rowsize.jsonl")
    if not args.analyze:
        cfgs = configs()
        if args.smoke:
            cfgs = [cfgs[0], cfgs[len(ROW_BYTES[2])]]  # one scale-2, one scale-4 config
        done = load_done(out)
        for cfg in cfgs:
            run_config(cfg, out, done)
            done = load_done(out)
        if not args.smoke:
            (RESULTS / "rowsize_meta.json").write_text(json.dumps(
                {**env_metadata(), "variants": ["fused", "fused_tiled", "copy"]}, indent=2))
    analyze(out)


if __name__ == "__main__":
    sys.exit(main())
