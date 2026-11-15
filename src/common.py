"""Shared data loading, preprocessing and evaluation for the Kepler exoplanet project.

Kaggle dataset "Exoplanet Hunting in Deep Space" (keplersmachines/kepler-labelled-time-series-data):
  exoTrain.csv : 5087 stars x (LABEL + 3197 flux points), 37 confirmed exoplanet hosts
  exoTest.csv  :  570 stars x (LABEL + 3197 flux points),  5 confirmed exoplanet hosts
  LABEL: 2 = exoplanet host star, 1 = no exoplanet
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.stats import kurtosis, skew
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("EXO_DATA_DIR", ROOT / "data"))
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- loading
def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load exoTrain / exoTest. Returns flux (n, 3197) float32 and labels y (n,) in {0, 1}."""
    df = pd.read_csv(DATA_DIR / f"exo{name.capitalize()}.csv")
    y = (df["LABEL"].values == 2).astype(np.int8)
    X = df.drop(columns=["LABEL"]).values.astype(np.float32)
    return X, y


# --------------------------------------------------------------------------- preprocessing
def clean_flux(X: np.ndarray, window: int = 101, clip_sigma: float = 3.0) -> np.ndarray:
    """Per-star cleaning of raw flux:
    1. Detrend: subtract a running median (removes slow stellar variability / instrument drift
       while keeping short transit dips).
    2. Clip upward outliers (flares, cosmic rays). Transits are dips, so only the top is clipped.
    3. Robust normalisation: divide by the median absolute deviation.
    """
    out = np.empty_like(X, dtype=np.float32)
    for i, f in enumerate(X):
        f = f - median_filter(f, size=window, mode="reflect")
        mad = np.median(np.abs(f - np.median(f))) * 1.4826 + 1e-8
        f = np.minimum(f, clip_sigma * mad)  # remove upward spikes only
        out[i] = (f - np.median(f)) / mad
    return out


def fft_features(Xc: np.ndarray, n_bins: int = 400) -> np.ndarray:
    """Magnitude spectrum of the cleaned flux, log-scaled and binned.
    Periodic transits show up as a comb of harmonics in frequency space."""
    spec = np.abs(np.fft.rfft(Xc, axis=1))[:, 1:]  # drop DC
    spec = np.log1p(spec)
    # average neighbouring bins to n_bins features
    usable = (spec.shape[1] // n_bins) * n_bins
    return spec[:, :usable].reshape(len(Xc), n_bins, -1).mean(axis=2)


def stat_features(Xc: np.ndarray) -> np.ndarray:
    """Hand-crafted summary statistics that respond to transit dips."""
    pct = np.percentile(Xc, [0.5, 1, 5, 25, 75, 95], axis=1).T
    n_dips = (Xc < -3).sum(axis=1, keepdims=True)  # points >3 MAD below median
    feats = np.column_stack([
        Xc.min(axis=1), Xc.std(axis=1), skew(Xc, axis=1), kurtosis(Xc, axis=1),
        pct, n_dips,
    ])
    return np.nan_to_num(feats).astype(np.float32)


def rf_features(Xc: np.ndarray) -> np.ndarray:
    return np.hstack([fft_features(Xc), stat_features(Xc)])


# --------------------------------------------------------------------------- evaluation
def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Pick the decision threshold that maximises F1 (tune on validation, never on test)."""
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[np.argmax(f1[:-1])]) if len(t) else 0.5


def report(name: str, y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ap = average_precision_score(y_true, scores)
    print(f"\n=== {name} (threshold={threshold:.3f}) ===")
    print(f"PR-AUC (average precision): {ap:.3f}")
    print("Confusion matrix [[TN FP] [FN TP]]:\n", cm)
    print(classification_report(y_true, y_pred, target_names=["no planet", "planet"],
                                digits=3, zero_division=0))
    return {"cm": cm, "ap": ap, "threshold": threshold}


def save_eval_plots(prefix: str, y_true: np.ndarray, scores: np.ndarray, cm: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    p, r, _ = precision_recall_curve(y_true, scores)
    ax[0].plot(r, p)
    ax[0].set(xlabel="Recall", ylabel="Precision",
              title=f"PR curve (AP={average_precision_score(y_true, scores):.3f})")
    ax[1].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax[1].text(j, i, str(v), ha="center", va="center", fontsize=14)
    ax[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no planet", "planet"],
              yticklabels=["no planet", "planet"], xlabel="Predicted", ylabel="True",
              title="Confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{prefix}_eval.png", dpi=130)
    plt.close(fig)
