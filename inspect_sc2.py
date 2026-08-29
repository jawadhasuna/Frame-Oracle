"""Inspect the SC2 link-prediction pickles before writing a loader.

~18 million frames across 911 radio links, saved as PyTorch tensors grouped
per link. This establishes shapes, label format, class balance, and whether
the features arrive pre-normalised -- all of which change how the loader and
the train/test split have to work.

Run:  uv run inspect_sc2.py
"""

import pickle
from pathlib import Path

import numpy as np
import torch

SPECTRUM = Path("..")
FILES = {
    "scrimmage4": SPECTRUM / "scrimmage4_link_dataset.pickle",
    "scrimmage5": SPECTRUM / "scrimmage5_link_dataset.pickle",
}

# Feature layout from the dataset README: SNR, MCS, centre frequency,
# bandwidth, then 16 PSD bins.
FEATURE_NAMES = (["snr", "mcs", "center_freq", "bandwidth"]
                 + [f"psd_{i}" for i in range(16)])

for tag, path in FILES.items():
    if not path.exists():
        print(f"{tag}: NOT FOUND at {path}")
        continue

    print(f"\n{'=' * 66}")
    print(f"{tag}  ({path.stat().st_size / 1e6:.0f} MB)")
    print("=" * 66)

    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    X0, y0 = data[0]
    print(f"links            : {len(data)}")
    print(f"per link         : tuple({type(X0).__name__}, {type(y0).__name__})")
    print(f"features shape   : {tuple(X0.shape)}  {X0.dtype}")
    print(f"labels shape     : {tuple(y0.shape)}  {y0.dtype}")

    n_feat = X0.shape[1] if X0.ndim > 1 else 1
    print(f"feature count    : {n_feat}"
          f"{'  (matches the documented 20)' if n_feat == 20 else '  <-- NOT 20'}")

    # --- labels ---------------------------------------------------------------
    yv = torch.unique(y0)
    print(f"label values     : {yv.tolist()[:10]}")

    # Class balance over a sample of links. Frame-error data is skewed, and by
    # how much decides whether accuracy is a usable metric at all.
    sample = data[:: max(1, len(data) // 60)]
    pos = sum(float(y.sum()) for _, y in sample)
    tot = sum(int(y.numel()) for _, y in sample)
    print(f"class balance    : {pos / tot * 100:.2f}% positive "
          f"over {tot:,} frames from {len(sample)} links")

    # Per-link balance matters separately: if some links never fail and others
    # always do, a model can score well by identifying the LINK rather than
    # learning anything about frames.
    rates = np.array([float(y.float().mean()) for _, y in sample])
    print(f"per-link rate    : min {rates.min():.3f}, "
          f"median {np.median(rates):.3f}, max {rates.max():.3f}")
    print(f"                   {(rates > 0.99).sum()} links >99% success, "
          f"{(rates < 0.01).sum()} links <1%")

    # --- features -------------------------------------------------------------
    # Pooled across a few links to see whether values are already standardised.
    pool = torch.cat([X for X, _ in data[:40]]).float()
    mu, sd = pool.mean(0), pool.std(0)
    lo, hi = pool.min(0).values, pool.max(0).values

    print(f"\n{'feature':<14} {'mean':>9} {'std':>8} {'min':>9} {'max':>9}")
    print("-" * 52)
    names = FEATURE_NAMES if n_feat == len(FEATURE_NAMES) else \
        [f"f{i}" for i in range(n_feat)]
    for i, nm in enumerate(names):
        print(f"{nm:<14} {mu[i]:>9.3f} {sd[i]:>8.3f} "
              f"{lo[i]:>9.3f} {hi[i]:>9.3f}")

    standardised = bool((mu.abs() < 0.5).all() and ((sd - 1).abs() < 0.5).all())
    print(f"\nlooks pre-standardised: {standardised}")
    if standardised:
        print("If so, the normalisation was fitted on the full dataset before")
        print("splitting -- test-set statistics leaked into training. Not fatal")
        print("for a reproduction, but it must be stated, and any comparison")
        print("against a properly-split baseline is not like-for-like.")

    del data

print("\n\nThe split question: records are grouped per link, so a random split")
print("across FRAMES would put frames from the same link in both train and")
print("test. Frames on one link share a channel, a distance, an interference")
print("environment -- so the model can memorise the link instead of learning")
print("what makes a frame survive. The paper distinguishes 'train and test on")
print("different links' from 'a pilot phase for each link' precisely because")
print("these are different problems with different achievable accuracies.")
print("\nSplitting BY LINK is the honest default.")
