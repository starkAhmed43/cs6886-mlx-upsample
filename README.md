# mlx-upsample

Study of bilinear upsampling in MLX. We compare stock gather, matmul, PR #4418 and a fused Metal kernel. This work relates to ml-explore/mlx issue #4435.

## Layout

- `src/mlx_upsample/`: reusable code (variants, baselines, kernels, workloads, bench).
- `experiments/NN_name/`: one folder per experiment, with its scripts and results.
- `tests/`: pytest tests.
- `~/GitHub/mlx`, branch `upsample-fused` (separate repo): the fused kernel as an MLX C++
  primitive, `mx.fast.upsample_bilinear`. Experiment 05 uses it. The exact diff is
  `experiments/05_final_benchmark/results/mlx_uncommitted.patch`.

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
  13.9x forward+backward over stock MLX.

Each experiment's README has its question, method, how to run, and findings.
