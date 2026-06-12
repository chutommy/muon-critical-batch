"""Reproduce the figures (default) or the full pipeline (--full)."""

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sh(*cmd):
    subprocess.run(cmd, cwd=ROOT, check=True)


def train_all():
    import torch

    nproc = torch.cuda.device_count() or 1
    for cfg in (ROOT / "configs/sweep/_manifest.txt").read_text().splitlines():
        if (ROOT / "results" / f"{Path(cfg).stem}.json").exists():
            continue
        sh("torchrun", "--standalone", f"--nproc_per_node={nproc}", "src/train.py", "--config", cfg)


def plot_all():
    sys.path.insert(0, str(ROOT / "analysis/plots"))
    for path in sorted((ROOT / "analysis/plots").glob("plot_*.py")):
        importlib.import_module(path.stem)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    if args.full:
        sh(sys.executable, "data/cached_fineweb10B.py", "40")
        sh(sys.executable, "configs/generate_sweep.py")
        train_all()
    sh(sys.executable, "analysis/parse.py")
    plot_all()
