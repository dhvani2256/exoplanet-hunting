"""Plot raw vs cleaned light curves for a few planet / non-planet stars (EDA figure for the README)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import OUT_DIR, clean_flux, load_split

X, y = load_split("train")
rng = np.random.default_rng(0)
idx = list(rng.choice(np.where(y == 1)[0], 3, replace=False)) + \
      list(rng.choice(np.where(y == 0)[0], 3, replace=False))
Xc = clean_flux(X[idx])

fig, ax = plt.subplots(len(idx), 2, figsize=(13, 2.2 * len(idx)), sharex=True)
for r, i in enumerate(idx):
    lab = "PLANET" if y[i] else "no planet"
    ax[r, 0].plot(X[i], lw=0.6); ax[r, 0].set_ylabel(f"#{i}\n{lab}")
    ax[r, 1].plot(Xc[r], lw=0.6, color="k")
ax[0, 0].set_title("Raw flux"); ax[0, 1].set_title("Detrended + normalised (MAD units)")
ax[-1, 0].set_xlabel("time index (~29.4 min cadence)"); ax[-1, 1].set_xlabel("time index")
fig.tight_layout(); fig.savefig(OUT_DIR / "examples.png", dpi=120)
print(f"Saved {OUT_DIR / 'examples.png'}")
