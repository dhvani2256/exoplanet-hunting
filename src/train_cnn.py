"""1D Convolutional Neural Network on the cleaned light curves.

Handles the extreme class imbalance (37 positives in 5087) with
  * balanced mini-batches (positives oversampled),
  * augmentation of positives: random circular time shift + time reversal
    (a transit is still a transit if shifted or mirrored),
  * a validation split carved out of TRAIN for early stopping and threshold choice.

Usage:  python src/train_cnn.py [--epochs 40]
"""
import argparse

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from common import OUT_DIR, best_f1_threshold, clean_flux, load_split, report, save_eval_plots

SEED = 42
tf.keras.utils.set_random_seed(SEED)


def build_model(n_points: int) -> keras.Model:
    inp = keras.Input(shape=(n_points, 1))
    x = inp
    for filters, k in [(16, 11), (32, 11), (64, 7), (64, 7), (128, 5)]:
        x = layers.Conv1D(filters, k, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(pool_size=4 if filters < 128 else 2)(x)
    # global max pool: "is there a transit-like pattern anywhere in the series?"
    x = layers.Concatenate()([layers.GlobalMaxPooling1D()(x), layers.GlobalAveragePooling1D()(x)])
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc"),
                 keras.metrics.Recall(name="recall"), keras.metrics.Precision(name="precision")],
    )
    return model


def balanced_batches(X, y, batch_size=64, pos_frac=0.5, rng=None):
    """Infinite generator of batches with ~pos_frac positives, augmenting positives."""
    rng = rng or np.random.default_rng(SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_pos = int(batch_size * pos_frac)
    n = X.shape[1]
    while True:
        idx_p = rng.choice(pos, n_pos, replace=True)
        idx_n = rng.choice(neg, batch_size - n_pos, replace=False)
        xb_p = X[idx_p].copy()
        for i in range(len(xb_p)):
            xb_p[i] = np.roll(xb_p[i], rng.integers(n))
            if rng.random() < 0.5:
                xb_p[i] = xb_p[i][::-1]
        xb = np.concatenate([xb_p, X[idx_n]])[..., None]
        yb = np.concatenate([np.ones(n_pos), np.zeros(batch_size - n_pos)]).astype(np.float32)
        perm = rng.permutation(batch_size)
        yield xb[perm], yb[perm]


def main(epochs: int, steps: int):
    Xtr_raw, ytr = load_split("train")
    Xte_raw, yte = load_split("test")
    Xtr, Xte = clean_flux(Xtr_raw), clean_flux(Xte_raw)
    # tame extreme values so they don't dominate the convs
    Xtr, Xte = np.clip(Xtr, -20, 5), np.clip(Xte, -20, 5)

    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=SEED)
    print(f"train {Xa.shape} ({ya.sum()} planets) | val {Xv.shape} ({yv.sum()}) | "
          f"test {Xte.shape} ({yte.sum()})")

    model = build_model(Xtr.shape[1])
    model.summary()
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=10,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5, patience=4),
    ]
    hist = model.fit(balanced_batches(Xa, ya), steps_per_epoch=steps, epochs=epochs,
                     validation_data=(Xv[..., None], yv), callbacks=cbs, verbose=2)

    val_scores = model.predict(Xv[..., None], verbose=0).ravel()
    thr = best_f1_threshold(yv, val_scores)
    report("CNN - validation", yv, val_scores, thr)

    scores = model.predict(Xte[..., None], verbose=0).ravel()
    res = report("CNN - held-out test", yte, scores, thr)
    save_eval_plots("cnn", yte, scores, res["cm"])
    save_history(hist)
    model.save(OUT_DIR / "cnn_model.keras")
    print(f"Saved model and plots to {OUT_DIR}")


def save_history(hist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = hist.history
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(h["loss"], label="train"); ax[0].plot(h["val_loss"], label="val")
    ax[0].set(title="Loss", xlabel="epoch"); ax[0].legend()
    ax[1].plot(h["pr_auc"], label="train"); ax[1].plot(h["val_pr_auc"], label="val")
    ax[1].set(title="PR-AUC", xlabel="epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT_DIR / "cnn_history.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=60, help="batches per epoch")
    a = ap.parse_args()
    main(a.epochs, a.steps)
