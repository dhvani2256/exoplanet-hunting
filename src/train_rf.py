"""Random Forest baseline on FFT + statistical features.

Usage:  python src/train_rf.py
"""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from common import (OUT_DIR, best_f1_threshold, clean_flux, load_split, report,
                    rf_features, save_eval_plots)

SEED = 42


def main():
    Xtr_raw, ytr = load_split("train")
    Xte_raw, yte = load_split("test")
    print(f"train: {Xtr_raw.shape}, planets={ytr.sum()} | test: {Xte_raw.shape}, planets={yte.sum()}")

    print("Cleaning light curves and extracting features...")
    Ftr = rf_features(clean_flux(Xtr_raw))
    Fte = rf_features(clean_flux(Xte_raw))

    rf = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",  # ~1:136 imbalance in the training set
        n_jobs=-1,
        random_state=SEED,
    )

    # Out-of-fold scores on TRAIN only -> choose threshold without touching the test set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(rf, Ftr, ytr, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    thr = best_f1_threshold(ytr, oof)
    report("Random Forest - 5-fold CV on train", ytr, oof, thr)

    rf.fit(Ftr, ytr)
    scores = rf.predict_proba(Fte)[:, 1]
    res = report("Random Forest - held-out test", yte, scores, thr)
    save_eval_plots("rf", yte, scores, res["cm"])

    imp = rf.feature_importances_
    n_fft = Ftr.shape[1] - 11
    print(f"Importance share: FFT bins={imp[:n_fft].sum():.2f}, stats={imp[n_fft:].sum():.2f}")
    joblib.dump({"model": rf, "threshold": thr}, OUT_DIR / "rf_model.joblib")
    print(f"Saved model and plots to {OUT_DIR}")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
