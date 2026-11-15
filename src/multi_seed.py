"""Repeat RF and CNN training over several random seeds and report mean ± std,
plus an ensemble of the CNNs (average of their predicted probabilities).

Usage:  python src/multi_seed.py --seeds 0 1 2 3 4 --epochs 40
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from tensorflow import keras

from common import OUT_DIR, best_f1_threshold, clean_flux, load_split, rf_features
from train_cnn import balanced_batches, build_model


def metrics(y, scores, thr):
    pred = (scores >= thr).astype(int)
    return {
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, scores),
        "TP": int(((pred == 1) & (y == 1)).sum()),
        "FP": int(((pred == 1) & (y == 0)).sum()),
    }


def run_rf(Ftr, ytr, Fte, yte, seed):
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                                class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = cross_val_predict(rf, Ftr, ytr, cv=cv, method="predict_proba")[:, 1]
    thr = best_f1_threshold(ytr, oof)
    rf.fit(Ftr, ytr)
    test = rf.predict_proba(Fte)[:, 1]
    return metrics(ytr, oof, thr), metrics(yte, test, thr)


def run_cnn(Xa, ya, Xv, yv, Xte, seed, epochs, steps):
    tf.keras.utils.set_random_seed(seed)
    model = build_model(Xa.shape[1])
    cbs = [keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=10,
                                         restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5, patience=4)]
    model.fit(balanced_batches(Xa, ya, rng=np.random.default_rng(seed)), steps_per_epoch=steps,
              epochs=epochs, validation_data=(Xv[..., None], yv), callbacks=cbs, verbose=0)
    val = model.predict(Xv[..., None], verbose=0).ravel()
    test = model.predict(Xte[..., None], verbose=0).ravel()
    return val, test


def summarise(rows, title):
    df = pd.DataFrame(rows)
    cols = ["recall", "precision", "f1", "pr_auc"]
    print(f"\n=== {title} ===")
    print(df.round(3).to_string(index=False))
    print("mean ± std: " + " | ".join(f"{c} {df[c].mean():.3f} ± {df[c].std():.3f}" for c in cols))
    return df


def main(seeds, epochs, steps):
    Xtr_raw, ytr = load_split("train")
    Xte_raw, yte = load_split("test")
    Xtr, Xte = clean_flux(Xtr_raw), clean_flux(Xte_raw)
    results = []

    # ---------------- Random Forest
    Ftr, Fte = rf_features(Xtr), rf_features(Xte)
    rf_cv, rf_test = [], []
    for s in seeds:
        cv_m, te_m = run_rf(Ftr, ytr, Fte, yte, s)
        rf_cv.append({"seed": s, **cv_m}); rf_test.append({"seed": s, **te_m})
        print(f"RF seed {s}: CV recall {cv_m['recall']:.2f} | test {te_m['TP']}/{yte.sum()} found, {te_m['FP']} FP")
    results.append(summarise(rf_cv, "Random Forest - 5-fold CV on train").assign(model="RF", split="cv"))
    results.append(summarise(rf_test, "Random Forest - test").assign(model="RF", split="test"))

    # ---------------- CNN (same train/val split for every seed, so the ensemble is fair)
    Xtr_c, Xte_c = np.clip(Xtr, -20, 5), np.clip(Xte, -20, 5)
    Xa, Xv, ya, yv = train_test_split(Xtr_c, ytr, test_size=0.2, stratify=ytr, random_state=42)
    cnn_val, cnn_test, val_scores, test_scores = [], [], [], []
    for s in seeds:
        v, t = run_cnn(Xa, ya, Xv, yv, Xte_c, s, epochs, steps)
        thr = best_f1_threshold(yv, v)
        cnn_val.append({"seed": s, **metrics(yv, v, thr)})
        cnn_test.append({"seed": s, **metrics(yte, t, thr)})
        val_scores.append(v); test_scores.append(t)
        print(f"CNN seed {s}: val PR-AUC {cnn_val[-1]['pr_auc']:.2f} | "
              f"test {cnn_test[-1]['TP']}/{yte.sum()} found, {cnn_test[-1]['FP']} FP")
    results.append(summarise(cnn_val, "CNN - validation").assign(model="CNN", split="val"))
    results.append(summarise(cnn_test, "CNN - test").assign(model="CNN", split="test"))

    # ---------------- CNN ensemble: average probabilities of all seeds
    ens_v, ens_t = np.mean(val_scores, axis=0), np.mean(test_scores, axis=0)
    thr = best_f1_threshold(yv, ens_v)
    ens = [{"seed": "ensemble", **metrics(yv, ens_v, thr)} | {"split": "val"},
           {"seed": "ensemble", **metrics(yte, ens_t, thr)} | {"split": "test"}]
    print("\n=== CNN ensemble (average of all seeds) ===")
    print(pd.DataFrame(ens).round(3).to_string(index=False))
    results.append(pd.DataFrame(ens).assign(model="CNN-ensemble"))

    out = OUT_DIR / "multi_seed_results.csv"
    pd.concat(results, ignore_index=True).to_csv(out, index=False)
    print(f"\nSaved all numbers to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=60)
    a = ap.parse_args()
    main(a.seeds, a.epochs, a.steps)
