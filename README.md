# mlx-upsample

Study of bilinear upsampling in MLX. We compare stock gather, matmul, PR #4418 and a fused Metal kernel. This work relates to ml-explore/mlx issue #4435.

```
git clone https://github.com/starkAhmed43/cs6886-mlx-upsample.git
```

## Report

The course report is [`report/report.pdf`](report/report.pdf).

Run `python src/report/make_figures.py` to regenerate its three figures from the experiment results into
`report/figures/`.

## Layout

- `src/mlx_upsample/`: reusable code (variants, baselines, kernels, workloads, bench).
- `experiments/NN_name/`: one folder per experiment, with its scripts and results.
- `tests/`: pytest tests.
- [starkAhmed43/mlx, branch `upsample-fused`](https://github.com/starkAhmed43/mlx/tree/upsample-fused)
  (separate repo): the fused kernel as an MLX C++ primitive, `mx.fast.upsample_bilinear`.
  Experiment 05 uses it. The exact diff is
  `experiments/05_final_benchmark/results/mlx_uncommitted.patch`.
  Upstream pull request: https://github.com/ml-explore/mlx/pull/XXXX (closes #4435).

## Setup

```
conda env create -f environment.yml
conda activate cs6886-mlx-upsample
pip install -e .
```

## Tests

```
pytest -q
```

`tests/test_edge_cases.py` also runs against `mx.fast.upsample_bilinear` when the installed MLX
has it (the `cs6886-mlx-dev` env).

## MLX fork (needed for experiment 05)

`mx.fast.upsample_bilinear` is not in stock MLX. It lives on a fork branch. You need it to run
experiment 05's `fused_cpp` variant and the `mx.fast.upsample_bilinear` tests in
`tests/test_edge_cases.py`.

### Prerequisites

- Apple silicon Mac.
- Xcode command line tools: `xcode-select --install`.
- The Metal toolchain component (recent Xcode ships it separately; the build fails with a
  "missing Metal Toolchain" error without it): `xcodebuild -downloadComponent MetalToolchain`.
- `cmake` >= 3.25 (`brew install cmake` or `pip install cmake`).

### Clone the fork and check out the branch

```
git clone https://github.com/starkAhmed43/mlx.git ~/GitHub/mlx
cd ~/GitHub/mlx
git checkout upsample-fused
```

### Create the dev env and build MLX from source

```
conda create -n cs6886-mlx-dev python=3.12
conda activate cs6886-mlx-dev
pip install nanobind setuptools
cd ~/GitHub/mlx
CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -e . --no-build-isolation
```

The build takes about 10-20 minutes. Then install this project into the same env, since
`run.py` and the tests import `mlx_upsample` there too:

```
cd /path/to/cs6886-mlx-upsample
pip install -e .
```

### Verify

```
python -c "import mlx.core as mx; print(mx.__version__, hasattr(mx.fast, 'upsample_bilinear'))"
```

On the `upsample-fused` branch, this prints a `0.32.5.dev...` version and `True` (the branch is
rebased past the `v0.32.4` tag; MLX's `version.h` there says `0.32.5`). On the exact-reproduction
route below (tag `v0.32.4` plus the patch), it prints `0.32.4.dev...` and `True`.

### Run experiment 05 and the edge-case tests

`experiments/05_final_benchmark/run.py` finds the `cs6886-mlx-upsample` and `cs6886-mlx-dev`
conda envs by name and calls out to `cs6886-mlx-dev`'s python itself for the `fused_cpp` rows. If
your envs are not conda envs with those names, set the `STOCK_PY`, `DEV_PY`, and `MLX_REPO`
environment variables to override them. Run from the repo root with the `cs6886-mlx-upsample`
env's python, as in [`05_final_benchmark`](experiments/05_final_benchmark):

```
python experiments/05_final_benchmark/run.py --smoke
python experiments/05_final_benchmark/analyze.py --smoke
```

For `tests/test_edge_cases.py`, run `pytest` from the `cs6886-mlx-dev` env so
`hasattr(mx.fast, "upsample_bilinear")` is true and those tests run instead of skipping:

```
conda activate cs6886-mlx-dev
pytest tests/test_edge_cases.py -q
```

### Exact reproduction of experiment 05's `fused_cpp` numbers

The published numbers come from MLX tag `v0.32.4` (commit `36969c094`) plus the uncommitted
working-tree diff in `experiments/05_final_benchmark/results/mlx_uncommitted.patch`, built from
source. The `upsample-fused` branch has the same code, rebased onto a newer upstream `main` (only
one comment was re-wrapped by `clang-format`). For day-to-day use, prefer the branch above. To
reproduce the exact numbers:

```
cd ~/GitHub/mlx
git checkout v0.32.4
git apply /path/to/cs6886-mlx-upsample/experiments/05_final_benchmark/results/mlx_uncommitted.patch
CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -e . --no-build-isolation
```

GPU timing must not run next to other GPU work. Measured on Apple M1, 16 GB.

## Experiments

- [`01_crossover`](experiments/01_crossover): finds where gather or matmul is faster, and checks
  whether PR #4418's 128-pixel rule picks the faster one.
- [`02_fused_kernel`](experiments/02_fused_kernel): builds a fused forward Metal kernel, checks it
  for parity, and times it against gather, matmul, and PR #4418.
- [`03_tiling`](experiments/03_tiling): adds threadgroup tiling to the fused kernel and profiles it
  on the GPU.
- [`04_backward`](experiments/04_backward): writes a backward (VJP) kernel for the fused kernel.
- [`05_final_benchmark`](experiments/05_final_benchmark): runs the MLX C++ primitive through
  `nn.Upsample` on 53 real-model workloads, with a geometric-mean speedup of 16.6x forward and
  13.9x forward+backward over stock MLX. The full run (`run.py`, no `--smoke`) takes about 35
  minutes on an Apple M1, based on `results/meta.json`'s start timestamps and
  `results/runs.jsonl`'s last-modified time.

Each experiment's README has its question, method, how to run, and findings.
