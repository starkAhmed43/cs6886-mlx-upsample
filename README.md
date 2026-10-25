# mlx-upsample

Study of bilinear upsampling in MLX. We compare stock gather, matmul, PR #4418 and a fused Metal kernel. This work relates to ml-explore/mlx issue #4435.

## Layout

- `src/mlx_upsample/`: reusable code (variants, baselines, kernels, workloads, bench).
- `experiments/NN_name/`: one folder per experiment, with its scripts and results.
- `tests/`: pytest tests.
- `PLAN.md`: goals and schedule.

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

## Experiments

- `01_crossover` (in progress): gather against matmul over sizes and scales.

```
python experiments/01_crossover/sweep.py --smoke
```
