"""
Generates a sweep of config files for training runs.
"""

import json
import math
from pathlib import Path

OUT_DIR = "configs/sweep"

SEED = 0
MBS = 8
NOISE_TARGET_SAMPLES = 100
B_SMALL_SEQS = 128

TOTAL_TOKENS = 1_000_000_000
TOTAL_TOKENS_LARGE_B = 2_000_000_000
TOTAL_TOKENS_THRESHOLD = 2_097_152

LR_BASELINE = 0.035
B_BASELINE = 524_288

BATCH_SIZES = [
    32_768, 65_536, 131_072, 262_144, 524_288,
    1_048_576, 2_097_152, 4_194_304, 8_388_608,
]
MUS = [0.85, 0.95, 0.99]
LR_SCALES = [0.5, 0.75, 1.0, 1.5]

TOP_UPS = [
    (32_768,    0.95, [0.25, 0.35]),
    (32_768,    0.99, [0.25, 0.35]),
    (131_072,   0.85, [2.0, 2.5]),
    (262_144,   0.85, [2.0, 2.5]),
    (524_288,   0.85, [2.0, 2.5]),
    (524_288,   0.99, [0.25, 0.35]),
    (1_048_576, 0.99, [0.25, 0.35]),
    (2_097_152, 0.99, [0.25, 0.35]),
]


def _write_config(out_dir, B, mu, lr_muon):
    total_tokens = TOTAL_TOKENS_LARGE_B if B >= TOTAL_TOKENS_THRESHOLD else TOTAL_TOKENS
    run_id = f"B{B}_mu{mu:.2f}_lr{lr_muon:.5f}_seed{SEED}"
    cfg = {
        "run_id": run_id,
        "batch_size": B,
        "mu": mu,
        "lr_muon": lr_muon,
        "total_tokens": total_tokens,
        "seed": SEED,
        "mbs": MBS,
        "noise_target_samples": NOISE_TARGET_SAMPLES,
        "b_small_seqs": B_SMALL_SEQS,
    }
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return str(out_path)


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "_manifest.txt"

    manifest = []
    for B in BATCH_SIZES:
        lr_anchor = LR_BASELINE * math.sqrt(B / B_BASELINE)
        for mu in MUS:
            for scale in LR_SCALES:
                lr_muon = round(scale * lr_anchor, 6)
                manifest.append(_write_config(out_dir, B, mu, lr_muon))

    for B, mu, scales in TOP_UPS:
        lr_anchor = LR_BASELINE * math.sqrt(B / B_BASELINE)
        for scale in scales:
            lr_muon = round(scale * lr_anchor, 6)
            manifest.append(_write_config(out_dir, B, mu, lr_muon))

    with open(manifest_path, "w") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"wrote {len(manifest)} configs to {out_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
