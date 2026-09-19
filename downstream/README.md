# Downstream evaluation — BEACON function and engineering tasks

Fine-tuning and evaluation code for the BEACON results reported in the paper, covering the
function and engineering tasks: SPL, APA, NcRNA, Modif, MRL, PRS, CRI-On and CRI-Off.

## Layout

| File | Purpose |
|---|---|
| `benchmark_beacon.py` | task runner: fine-tuning and evaluation |
| `task_config.py` | per-task settings (metric, max length, epochs) and the encoder entry |
| `metrics.py` | metric implementations |

## Data

Obtain the BEACON task data from the benchmark's own release and place it so that each task
directory sits under one root, e.g. `beacon_data/MeanRibosomeLoading/{train,val,test}.csv`.
The splits are used exactly as released; this repository does not re-partition them.

## Running

```bash
python benchmark_beacon.py \
    --task mrl \
    --model RNAElectra \
    --data_dir ./beacon_data \
    --output_dir ./benchmark_results \
    --seed 42 \
    --lr 1e-4 \
    --batch_size 32
```

`--model RNAElectra` loads the released encoder from the Hugging Face repository. To evaluate a
different checkpoint, add `--model_path /path/to/checkpoint`, which accepts either a local
directory or a Hugging Face identifier.

Each run writes `config.json` (the full resolved configuration) and `results.csv` (one row per
evaluation) into `<output_dir>/<task>/<model>_<date>_seed<seed>_lr<lr>/`.

The reported metric for each task is named in `task_config.py` and repeated in the
`primary_metric` column of `results.csv`.

## Reported settings

Results in the paper are the mean over seeds 42, 30 and 2026. Learning rates were selected per
task on the validation split; the grids are given in the manuscript's Methods.

Regression tasks report `r2` as the coefficient of determination. The squared Pearson
correlation used in the additional MRL comparison is computed separately and is not this column.

## Dependencies

`torch`, `transformers`, `scikit-learn`, `scipy`, `pandas` and `numpy`. The RNAElectra tokenizer
is imported from `model/` at the repository root.

Baseline values in the paper's BEACON table are the benchmark's own published numbers and were
not produced with this code.
