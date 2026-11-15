"""Real-data exploration with Lightkurve: download a Kepler light curve from MAST,
clean it, find the planet's period with a Box Least Squares periodogram and fold it.

Usage:  python src/lightkurve_explore.py --target "Kepler-10" [--quarters 1 2 3 4 5]
Other good targets: "Kepler-7", "Kepler-8", "Kepler-12", "KIC 8462852" (Tabby's star, no planet)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import lightkurve as lk

from common import OUT_DIR


def main(target: str, quarters: list[int] | None, lc=None):
    if lc is None:
        print(f"Searching MAST for {target} ...")
        search = lk.search_lightcurve(target, mission="Kepler", author="Kepler", cadence="long",
                                      quarter=quarters)
        print(search)
        lc = search.download_all().stitch()           # normalise each quarter and join
    lc = lc.remove_nans().remove_outliers(sigma_upper=4, sigma_lower=20)
    flat = lc.flatten(window_length=901)              # remove stellar variability

    # Search periods 0.5-30 d on a grid uniform in frequency (fine enough for short periods
    # like Kepler-10b at 0.84 d). Larger frequency_factor = coarser grid = faster.
    bls = flat.to_periodogram(method="bls", minimum_period=0.5, maximum_period=30,
                              duration=[0.05, 0.08, 0.12, 0.2, 0.3], frequency_factor=100)
    P = bls.period_at_max_power                 # astropy Quantity (days)
    t0 = bls.transit_time_at_max_power          # astropy Time (BKJD)
    dur = bls.duration_at_max_power             # astropy Quantity (days)
    depth_ppm = float(np.asarray(getattr(bls.depth_at_max_power, "value", bls.depth_at_max_power))) * 1e6
    print(f"\nBest period: {P.value:.4f} d | epoch: {t0.value:.4f} BKJD | "
          f"duration: {dur.value * 24:.2f} h | depth ≈ {depth_ppm:.0f} ppm")

    folded = flat.fold(period=P, epoch_time=t0)
    fig, ax = plt.subplots(4, 1, figsize=(10, 13))
    lc.plot(ax=ax[0], title=f"{target}: stitched PDCSAP flux")
    flat.plot(ax=ax[1], title="Flattened")
    bls.plot(ax=ax[2], title="BLS periodogram"); ax[2].axvline(P.value, color="r", alpha=0.3, lw=6)
    folded.scatter(ax=ax[3], s=1, alpha=0.3, label="folded")
    folded.bin(time_bin_size=0.01).plot(ax=ax[3], color="r", lw=2, label="binned")
    ax[3].set_xlim(-3 * dur.value, 3 * dur.value)
    ax[3].set_title(f"Folded on P = {P.value:.4f} d")
    fig.tight_layout()
    out = OUT_DIR / f"lightkurve_{target.replace(' ', '_')}.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Kepler-10")
    ap.add_argument("--quarters", type=int, nargs="*", default=None)
    a = ap.parse_args()
    main(a.target, a.quarters)
