"""
Convert results/*.json into four long-format Parquet tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "analysis" / "tables"

RUN_SCALAR_FIELDS = (
    "world_size",
    "train_steps",
    "val_step_freq",
    "wallclock_seconds",
    "final_loss",
)
CONFIG_FIELDS = (
    "batch_size",
    "mu",
    "lr_muon",
    "total_tokens",
    "seed",
    "mbs",
    "noise_target_samples",
    "b_small_seqs",
)
VAL_FIELDS = ("step", "tokens", "val_loss", "training_time_s")
NOISE_FIELDS = (
    "step",
    "tokens",
    "dot",
    "diff_sq",
    "tr_sigma",
    "B_noise_tokens",
    "measure_time_s",
)
TRAIN_FIELDS = (
    "step",
    "tokens",
    "examples",
    "train_loss",
    "lr_muon",
    "training_time_s",
)


def parse_one(path: Path):
    with path.open() as f:
        d = json.load(f)

    run_id = d["run_id"]
    cfg = d["config"]
    assert cfg["run_id"] == run_id

    run_row = {"run_id": run_id, "config_path": cfg["config"]}
    run_row.update({k: cfg[k] for k in CONFIG_FIELDS})
    run_row.update({k: d[k] for k in RUN_SCALAR_FIELDS})

    def with_run_id(trace, fields):
        return [{"run_id": run_id, **{k: r[k] for k in fields}} for r in trace]

    return (
        run_row,
        with_run_id(d["val_loss_trace"], VAL_FIELDS),
        with_run_id(d["noise_scale_trace"], NOISE_FIELDS),
        with_run_id(d["train_step_trace"], TRAIN_FIELDS),
    )


def main():
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files in {RESULTS_DIR}")

    runs, vals, noises, trains = [], [], [], []
    for p in files:
        run_row, v, n, t = parse_one(p)
        runs.append(run_row)
        vals.extend(v)
        noises.extend(n)
        trains.extend(t)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        "runs": pd.DataFrame(runs),
        "val_loss": pd.DataFrame(vals),
        "noise_scale": pd.DataFrame(noises),
        "train_step": pd.DataFrame(trains),
    }
    for name, df in tables.items():
        out = OUT_DIR / f"{name}.parquet"
        df.to_parquet(out, index=False)
        print(f"{name:12s} {len(df):>7d} rows  ->  {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
