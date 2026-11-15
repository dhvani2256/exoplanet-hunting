"""Create small synthetic exoTrain.csv / exoTest.csv with the same format as the Kaggle files,
so the pipeline can be smoke-tested without downloading the real data.

Usage:  python scripts/make_synthetic.py data_synth   then   EXO_DATA_DIR=data_synth python src/train_rf.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_PTS = 3197
rng = np.random.default_rng(1)


def star(planet: bool) -> np.ndarray:
    t = np.arange(N_PTS)
    base = rng.uniform(-500, 500)
    trend = rng.normal(0, 60) * np.sin(2 * np.pi * t / rng.uniform(800, 4000) + rng.uniform(0, 6))
    noise = rng.normal(0, rng.uniform(5, 30), N_PTS)
    f = base + trend + noise
    if planet:
        period, width = rng.integers(150, 700), rng.integers(4, 12)
        depth = rng.uniform(3, 8) * noise.std()
        phase = (t - rng.integers(period)) % period
        f[phase < width] -= depth
    return f


def make(n, n_pos):
    labels = np.r_[np.full(n_pos, 2), np.ones(n - n_pos, int)]
    rows = np.array([star(l == 2) for l in labels])
    df = pd.DataFrame(rows, columns=[f"FLUX.{i+1}" for i in range(N_PTS)])
    df.insert(0, "LABEL", labels)
    return df


out = Path(sys.argv[1] if len(sys.argv) > 1 else "data_synth")
out.mkdir(exist_ok=True)
make(1200, 37).to_csv(out / "exoTrain.csv", index=False)
make(300, 5).to_csv(out / "exoTest.csv", index=False)
print(f"Wrote synthetic data to {out}/")
