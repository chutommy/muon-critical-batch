# Critical Batch Size under Muon and the Limits of the Gradient Noise Scale

A CS-439 project investigating whether a critical batch size exists for the Muon optimiser, how momentum shifts it, and whether McCandlish's gradient noise scale predicts it under Muon's non-linear Newton–Schulz update.

Based on [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) by Keller Jordan.

Thanks to [RCI CTU](https://rci.cvut.cz/) for providing the compute infrastructure used to run these experiments.

```
requirements.txt
run.py                    reproduces figures (default) or full pipeline (--full)
run.sbatch                runs slurm job (single config or array)
src/
    train.py              trains modded-nanogpt (track 3)
configs/
    generate_sweep.py     spits out 124-config sweep
    sweep/                per-run JSON config
data/
    cached_fineweb10B.py  fineweb downloader
results/                  per-run JSON result
logs/                     per-run TEXT log and SLURM stdout
analysis/
    parse.py              turns results JSON into analysis PARQUET
    plots/                per-figure plot script
```

## Reproduce figures

All logs, outputs, and parsed results are shipped in this repo, so the figures are ez to re-generate without burning down forests.

```
pip install -r requirements.txt
python run.py
```

This rebuilds `analysis/tables/*.parquet` from `results/*.json` and writes figures to `analysis/figures`.

## Reproduce results

Single-GPU setup:

```
python run.py --full
```

This downloads data, generates 124 sweep configs, trains them all, then parses and plots.

Alternatively, run multi-GPU setup:

```bash
python data/cached_fineweb10B.py 40
python configs/generate_sweep.py
sbatch --array=0-123%4 run.sbatch
python run.py
```

This downloads data, generates 124 sweep configs, trains them all, then parses and plots :)
