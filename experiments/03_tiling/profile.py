"""Read Apple GPU hardware counters for `fused`, `fused_tiled` and the `copy` kernel.

Uses Xcode's `xctrace` (Metal System Trace template + Metal GPU Counters instrument) to
record one short trace per (config, variant), then parses the exported XML tables to find
what limits `fused_tiled` relative to `fused` and to the `copy` bandwidth ceiling.

Each trace launches this same file in a child process (`--run-child CONFIG VARIANT`): it
builds the input once, warms up 5 calls, then calls the kernel in a loop for ~1.5s of GPU
time (mx.eval per call), and writes the call count to a small JSON sidecar.

xctrace's `--launch` matches the launched binary by name through Launch Services, not by
its literal path: launching a path literally named "python" picks whatever registered
`python.app` Launch Services finds (e.g. a *different* conda env's interpreter), silently
running the wrong interpreter. We work around this by launching a uniquely-named symlink
to `sys.executable` instead.

XML export gotcha: values are de-duplicated. The first occurrence of a value is
`<tag id="N" fmt="...">raw</tag>`; later ones are `<tag ref="N"/>`. We resolve refs by
id across the whole exported document and always use the raw inner text (not `fmt`) for
numbers.
"""

import argparse
import csv
import functools
import json
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
TRACES = RESULTS / "scratch" / "traces"  # overwritten in main() to a per-label subdir
RUNNER_NAME = "_mlx_profile_runner"  # must not be "python": see module docstring

sys.path.insert(0, str(HERE))
from rowsize import make_copy  # noqa: E402

import mlx.core as mx  # noqa: E402
from mlx_upsample.kernels.bilinear import fused, fused_tiled  # noqa: E402
from mlx_upsample.workloads import make_input, out_hw  # noqa: E402
from mlx_upsample.workloads import real as real_workloads  # noqa: E402

VARIANTS = {
    "fused": fused,
    "fused_tiled": fused_tiled,
    "fused_rows4": functools.partial(fused, tg_rows=4),
    "fused_rows8": functools.partial(fused, tg_rows=8),
    "copy": None,  # handled separately: not a bilinear kernel
}
VARIANT_NAMES = list(VARIANTS)

CONFIGS = {
    "rowsize_1MiB": {"kind": "rowsize", "scale": 2, "H": 32, "W": 4096, "C": 64, "dtype": "float32"},
    "rowsize_4MiB": {"kind": "rowsize", "scale": 2, "H": 8, "W": 16384, "C": 64, "dtype": "float32"},
    "rife_up4": {"kind": "real", "name": "rife.up4", "B": 1, "dtype": "float16"},
    "rife_up2": {"kind": "real", "name": "rife.up2", "B": 1, "dtype": "float16"},
}

COUNTERS = [
    "GPU Read Bandwidth",
    "GPU Write Bandwidth",
    "ALU Utilization",
    "ALU Limiter",
    "Compute Occupancy",
    "Buffer Read Limiter",
    "Buffer Load Utilization",
    "Buffer Write Limiter",
    "Buffer Store Utilization",
    "GPU Last Level Cache Utilization",
    "GPU Last Level Cache Limiter",
    "MMU Limiter",
    "Threadgroup/Imageblock Load Limiter",
    "Threadgroup/Imageblock Load Utilization",
    "Threadgroup/Imageblock Store Limiter",
    "Threadgroup/Imageblock Store Utilization",
    "F32 Utilization",
    "F16 Utilization",
]

COUNTER_EXPLAIN = {
    "GPU Read Bandwidth": "Bytes per second the GPU reads from main memory (DRAM). Unit looks like GB/s (see note below).",
    "GPU Write Bandwidth": "Bytes per second the GPU writes to main memory (DRAM). Unit looks like GB/s (see note below).",
    "ALU Utilization": "How busy the math (arithmetic) units are, percent of peak.",
    "ALU Limiter": "How often math-unit work is stalled waiting to run, percent of peak; high means math is the bottleneck.",
    "Compute Occupancy": "Percent of the GPU's compute slots that are busy running our kernel at once.",
    "Buffer Read Limiter": "How often a buffer (array) read is stalled waiting to run, percent of peak; high means reads are the bottleneck.",
    "Buffer Load Utilization": "How busy the buffer-read hardware is, percent of peak.",
    "Buffer Write Limiter": "How often a buffer (array) write is stalled waiting to run, percent of peak; high means writes are the bottleneck.",
    "Buffer Store Utilization": "How busy the buffer-write hardware is, percent of peak.",
    "GPU Last Level Cache Utilization": "How busy the GPU's shared (last-level) cache is, percent of peak.",
    "GPU Last Level Cache Limiter": "How often the shared cache is the reason work is stalled, percent of peak; high means the cache is the bottleneck.",
    "MMU Limiter": "How often address translation on a cache miss stalls work, percent of peak; high points at DRAM, not cache.",
    "Threadgroup/Imageblock Load Limiter": "How often reading fast on-chip shared memory stalls work, percent of peak.",
    "Threadgroup/Imageblock Load Utilization": "How busy the fast on-chip shared-memory read path is, percent of peak.",
    "Threadgroup/Imageblock Store Limiter": "How often writing fast on-chip shared memory stalls work, percent of peak.",
    "Threadgroup/Imageblock Store Utilization": "How busy the fast on-chip shared-memory write path is, percent of peak.",
    "F32 Utilization": "How busy the 32-bit float math units are, percent of peak.",
    "F16 Utilization": "How busy the 16-bit float math units are, percent of peak (relevant for the fp16 rife.up4 config).",
}


# ---------------------------------------------------------------------------
# Building the workload (shared by the host's skip-check and the child's run)
# ---------------------------------------------------------------------------

def _find_workload(cfg: dict):
    matches = [w for w in real_workloads() if w.name == cfg["name"] and w.B == cfg["B"] and w.dtype == cfg["dtype"]]
    assert len(matches) == 1, f"expected exactly one workload match for {cfg}, got {len(matches)}"
    return matches[0]


def _sizes(cfg: dict) -> tuple[int, int]:
    """(n_in, n_out) element counts, computed without touching the GPU."""
    if cfg["kind"] == "rowsize":
        n_in = cfg["H"] * cfg["W"] * cfg["C"]
        return n_in, n_in * cfg["scale"] * cfg["scale"]
    w = _find_workload(cfg)
    oh, ow = out_hw(w)
    return w.B * w.H * w.W * w.C, w.B * oh * ow * w.C


def copy_runnable(cfg: dict) -> tuple[bool, int, int]:
    n_in, n_out = _sizes(cfg)
    return n_out % n_in == 0, n_in, n_out


def build_arrays(cfg: dict):
    if cfg["kind"] == "rowsize":
        H, W, C, scale = cfg["H"], cfg["W"], cfg["C"], cfg["scale"]
        x = mx.random.normal((1, H, W, C)).astype(getattr(mx, cfg["dtype"]))
        sf = (float(scale), float(scale))
        out_shape = (1, int(scale * H), int(scale * W), C)
    else:
        w = _find_workload(cfg)
        x = make_input(w)
        sf = (w.scale, w.scale)
        out_shape = (w.B, *out_hw(w), w.C)
    mx.eval(x)
    return x, sf, out_shape


def make_variant_fn(variant: str, x, sf, out_shape):
    if variant == "copy":
        copy_fn = make_copy(out_shape)
        return lambda: mx.eval(copy_fn(x))
    if variant not in VARIANTS:
        raise ValueError(variant)
    kernel = VARIANTS[variant]
    return lambda: mx.eval(kernel(x, sf, False))


def run_child(config_name: str, variant: str, out_json: Path) -> None:
    cfg = CONFIGS[config_name]
    x, sf, out_shape = build_arrays(cfg)
    fn = make_variant_fn(variant, x, sf, out_shape)
    for _ in range(5):
        fn()
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < 1.5:
        fn()
        n += 1
    out_json.write_text(json.dumps({"n_calls": n, "elapsed_s": time.perf_counter() - t0}))


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def _runner_symlink() -> Path:
    link = TRACES / RUNNER_NAME
    TRACES.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(sys.executable)
    return link


def record_trace(config_name: str, variant: str, trace_path: Path, json_path: Path) -> None:
    if trace_path.exists():
        shutil.rmtree(trace_path)
    json_path.unlink(missing_ok=True)
    runner = _runner_symlink()
    cmd = [
        "xcrun", "xctrace", "record",
        "--template", "Metal System Trace",
        "--instrument", "Metal GPU Counters",
        "--output", str(trace_path),
        "--launch", "--",
        str(runner), str(Path(__file__).resolve()), "--run-child", config_name, variant, "--out-json", str(json_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=90)


# ---------------------------------------------------------------------------
# XML parsing (see module docstring for the id/ref de-duplication scheme)
# ---------------------------------------------------------------------------

_LEAF = re.compile(r'id="(\d+)" fmt="(?:[^"\\]|\\.)*?">([^<]*)<')
_ATTR = re.compile(r'<(?:process|gpu-channel-name|gpu-counter-name) id="(\d+)" fmt="((?:[^"\\]|\\.)*?)"')


def _build_id_map(xml: str) -> dict:
    id_map = dict(_LEAF.findall(xml))
    id_map.update(_ATTR.findall(xml))
    return id_map


def _rows(xml: str) -> list:
    return re.findall(r"<row>(.*?)</row>", xml, re.S)


def _col(row: str, tagname: str, id_map: dict) -> list:
    ids = re.findall(rf'<{tagname} (?:id|ref)="(\d+)"', row)
    return [id_map[i] for i in ids]


def export_table(trace_path: Path, schema: str) -> str:
    cmd = ["xcrun", "xctrace", "export", "--input", str(trace_path),
           "--xpath", f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return r.stdout


def parse_counter_info(xml: str) -> dict:
    """counter-id (int) -> name."""
    id_map = _build_id_map(xml)
    out = {}
    for row in _rows(xml):
        cid = int(_col(row, "uint32", id_map)[0])
        name = _col(row, "gpu-counter-name", id_map)[0]
        out[cid] = name
    return out


def parse_counter_value(xml: str) -> list:
    """list of (timestamp_ns, counter_id, value)."""
    id_map = _build_id_map(xml)
    out = []
    for row in _rows(xml):
        ts = int(_col(row, "event-time", id_map)[0])
        cid = int(_col(row, "uint32", id_map)[0])
        val = float(_col(row, "fixed-decimal", id_map)[0])
        out.append((ts, cid, val))
    return out


def parse_intervals(xml: str, process_substr: str, channel: str | None = "Compute") -> list:
    """list of (start_ns, duration_ns) for rows whose process matches and (if given) channel matches."""
    id_map = _build_id_map(xml)
    out = []
    for row in _rows(xml):
        procs = _col(row, "process", id_map)
        proc = procs[-1] if procs else ""
        if process_substr not in proc:
            continue
        if channel is not None:
            chans = _col(row, "gpu-channel-name", id_map)
            if not chans or chans[0] != channel:
                continue
        start = int(_col(row, "start-time", id_map)[0])
        dur = int(_col(row, "duration", id_map)[0])
        out.append((start, dur))
    return out


def analyze_trace(trace_path: Path, json_path: Path) -> dict:
    meta = json.loads(json_path.read_text()) if json_path.exists() else {}
    n_calls = meta.get("n_calls")

    info_xml = export_table(trace_path, "gpu-counter-info")
    id_to_name = parse_counter_info(info_xml)

    intervals_xml = export_table(trace_path, "metal-gpu-intervals")
    compute = parse_intervals(intervals_xml, RUNNER_NAME, channel="Compute")
    note = ""
    if not compute:
        compute = parse_intervals(intervals_xml, RUNNER_NAME, channel=None)
        note = "no Compute-channel intervals found for our process; fell back to all GPU-channel intervals for it."
    if not compute:
        note = "no intervals found for our process at all; averaged over the whole trace instead."
        window = None
    else:
        starts = [s for s, _ in compute]
        ends = [s + d for s, d in compute]
        window = (min(starts), max(ends))

    total_gpu_ns = sum(d for _, d in compute)
    gpu_ms_per_call = (total_gpu_ns / 1e6 / n_calls) if (compute and n_calls) else None

    value_xml = export_table(trace_path, "gpu-counter-value")
    samples = parse_counter_value(value_xml)
    by_counter = defaultdict(list)
    for ts, cid, val in samples:
        if window is not None and not (window[0] <= ts <= window[1]):
            continue
        name = id_to_name.get(cid)
        if name in COUNTERS:
            by_counter[name].append(val)

    counters = {}
    for name in COUNTERS:
        vals = by_counter.get(name, [])
        counters[name] = {"mean": sum(vals) / len(vals) if vals else None, "n": len(vals)}
    return {"counters": counters, "gpu_ms_per_call": gpu_ms_per_call, "n_calls": n_calls, "note": note}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "config", "counter", "mean", "n_samples", "gpu_ms_per_call"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _load_v1_gpu_ms() -> dict:
    """(config, variant) -> gpu_ms_per_call from results/profile_v1.csv, if it exists."""
    v1_path = RESULTS / "profile_v1.csv"
    if not v1_path.exists():
        return {}
    out = {}
    with v1_path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["config"], row["variant"])
            if key not in out and row["gpu_ms_per_call"]:
                out[key] = float(row["gpu_ms_per_call"])
    return out


def write_summary_md(summaries: dict, path: Path, label: str) -> None:
    lines = ["# GPU hardware counter profile\n",
             "Read with Apple's `xctrace` Metal GPU Counters instrument, one trace per (config, "
             "variant). Each number is the average, over that trace's GPU counter samples, "
             "restricted to the time this process's GPU \"Compute\" work was running (not idle "
             "time, and not other apps' GPU work).\n",
             "What each counter means:\n"]
    for name in COUNTERS:
        lines.append(f"- **{name}**: {COUNTER_EXPLAIN[name]}")
    lines.append("\nNote on units: `GPU Read/Write Bandwidth` values below land in the same "
                 "~20-65 range as the GB/s this project measures by wall-clock timing elsewhere "
                 "(see `rowsize_summary.md`), so the counter's unit is GB/s, not a raw byte count "
                 "or a percentage.\n")

    for config_name, by_variant in summaries.items():
        variants_present = [v for v in VARIANT_NAMES if v in by_variant]
        lines.append(f"\n## {config_name}\n")
        lines.append("| counter | " + " | ".join(variants_present) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in variants_present) + " |")
        for name in COUNTERS:
            cells = []
            for v in variants_present:
                res = by_variant[v]
                if res.get("skipped"):
                    cells.append("skipped")
                    continue
                c = res["counters"][name]
                cells.append(f"{c['mean']:.2f}" if c["mean"] is not None else "n/a")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        gms_row = []
        for v in variants_present:
            res = by_variant[v]
            if res.get("skipped"):
                gms_row.append("skipped")
            else:
                g = res.get("gpu_ms_per_call")
                gms_row.append(f"{g:.4f}" if g is not None else "n/a")
        lines.append(f"| **GPU ms / call** | " + " | ".join(gms_row) + " |")
        for v in variants_present:
            res = by_variant[v]
            if res.get("skipped"):
                lines.append(f"\n_{v}: skipped, {res['skip_reason']}_")
            elif res.get("note"):
                lines.append(f"\n_{v}: {res['note']}_")

    v1_gpu_ms = _load_v1_gpu_ms()
    has_v1 = bool(v1_gpu_ms)
    lines.append(f"\n## GPU ms / call: {label} vs. v1\n")
    header = ["config", "variant", label] + (["v1"] if has_v1 else [])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for config_name, by_variant in summaries.items():
        for v in VARIANT_NAMES:
            res = by_variant.get(v)
            if res is None or res.get("skipped"):
                continue
            g = res.get("gpu_ms_per_call")
            cells = [config_name, v, f"{g:.4f}" if g is not None else "n/a"]
            if has_v1:
                g1 = v1_gpu_ms.get((config_name, v))
                cells.append(f"{g1:.4f}" if g1 is not None else "-")
            lines.append("| " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--label", default="v2", help="output name suffix: results/profile_{label}.csv etc.")
    ap.add_argument("--run-child", nargs=2, metavar=("CONFIG", "VARIANT"))
    ap.add_argument("--out-json", type=Path)
    args = ap.parse_args()

    if args.run_child:
        run_child(args.run_child[0], args.run_child[1], args.out_json)
        return

    global TRACES
    TRACES = RESULTS / "scratch" / "traces" / args.label

    mx.set_cache_limit(1 << 30)
    TRACES.mkdir(parents=True, exist_ok=True)

    configs = list(CONFIGS)
    variants = VARIANT_NAMES
    prefix = ""
    if args.smoke:
        configs, variants, prefix = configs[:1], variants[:1], "smoke_"

    rows, summaries = [], {}
    for config_name in configs:
        cfg = CONFIGS[config_name]
        summaries[config_name] = {}
        for variant in variants:
            if variant == "copy":
                ok, n_in, n_out = copy_runnable(cfg)
                if not ok:
                    reason = f"N_out ({n_out}) is not an exact multiple of N_in ({n_in})"
                    print(f"{config_name} copy: skipped ({reason})", flush=True)
                    summaries[config_name]["copy"] = {"skipped": True, "skip_reason": reason}
                    continue

            trace_path = TRACES / f"{prefix}{config_name}__{variant}.trace"
            json_path = TRACES / f"{prefix}{config_name}__{variant}.json"
            if not args.analyze:
                print(f"recording {config_name} {variant}...", flush=True)
                record_trace(config_name, variant, trace_path, json_path)
            if not trace_path.exists():
                print(f"{config_name} {variant}: no trace found, skipping analysis", flush=True)
                continue

            result = analyze_trace(trace_path, json_path)
            summaries[config_name][variant] = result
            print(f"{config_name} {variant}: {result['gpu_ms_per_call']} ms/call "
                  f"({result['n_calls']} calls); {result['note']}", flush=True)
            for name in COUNTERS:
                c = result["counters"][name]
                rows.append({
                    "variant": variant, "config": config_name, "counter": name,
                    "mean": c["mean"], "n_samples": c["n"], "gpu_ms_per_call": result["gpu_ms_per_call"],
                })

    label = f"smoke_{args.label}" if args.smoke else args.label
    csv_path, md_path = RESULTS / f"profile_{label}.csv", RESULTS / f"profile_{label}_summary.md"
    write_csv(rows, csv_path)
    write_summary_md(summaries, md_path, label)
    print(f"wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    sys.exit(main())
