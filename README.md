# Muon's critical batch size, and how momentum moves it

A CS-439 nano-project on how Muon's per-step efficiency depends on batch size and momentum, and whether McCandlish's gradient noise scale predicts the critical batch size for Muon's non-linear update.

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
