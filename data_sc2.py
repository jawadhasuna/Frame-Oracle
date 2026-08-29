"""Loading the DARPA SC2 link-prediction data, and splitting it honestly.

Real measurements from the Colosseum testbed during SC2 scrimmages 4 and 5,
collected by Jameel, Mohamed, Zhang and El Gamal (arXiv 2005.01446). Each
record is one transmitted frame: 20 features describing the link and the
spectrum at that moment, and a binary label saying whether the frame arrived.

    features : SNR, MCS, centre frequency, bandwidth, 16 PSD bins
    label    : 1 = received, 0 = frame error

Those 16 PSD bins are the same quantity the SCATTER PHY paper describes its
FPGA computing and reporting upward -- averaged power spectral density, the
input an AI layer uses to choose a channel.

THE SPLIT IS THE WHOLE DESIGN
-----------------------------
Frames are grouped by radio link, and per-link success rates run from 0.26 to
1.00. Split frames at random and the same link appears in train and test, so a
model can score well by recognising WHICH LINK it is looking at -- easily done
from centre frequency and bandwidth -- rather than learning what makes a frame
survive. The score is high and means nothing.

Two honest scenarios, matching the paper:

  by_link   Train on some links, test on entirely unseen ones. Answers "will
            this work on a link I have never observed?" -- the deployment
            question, and the harder number.

  by_frame  Every link contributes frames to both sets. Answers "given a
            pilot period on this link, can I predict its later frames?" -- a
            legitimate and easier problem, not a leak, PROVIDED it is
            reported as such and never compared against by_link as though
            they measured the same thing.

Also worth stating: the features arrive already standardised, and that
standardisation was fitted over the whole dataset before any split. Test-set
statistics are therefore in the training features. It is a small leak and
unavoidable without the raw data, but it means these numbers are not
like-for-like against a pipeline that normalises after splitting.
"""

import pickle
from pathlib import Path

import numpy as np

SPECTRUM = Path("..")
FILES = {
    "scrimmage4": SPECTRUM / "scrimmage4_link_dataset.pickle",
    "scrimmage5": SPECTRUM / "scrimmage5_link_dataset.pickle",
}

FEATURE_NAMES = (["snr", "mcs", "center_freq", "bandwidth"]
                 + [f"psd_{i}" for i in range(16)])


def load_links(which="scrimmage4", max_links=None, max_frames_per_link=None):
    """Load per-link (features, labels) as NumPy arrays.

    Args:
        which: "scrimmage4" or "scrimmage5".
        max_links: keep only the first N links, for quick iteration.
        max_frames_per_link: subsample frames within each link. Frames are
            time-ordered, so this takes an evenly spaced slice rather than
            the first N -- taking a prefix would sample one period of the
            scrimmage rather than the whole run.

    Returns:
        list of (X float32 (n, 20), y int64 (n,))
    """
    path = FILES[which]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the SC2 link-prediction pickles "
            f"from the links in github.com/amahdeej/sc2-frame-error"
        )

    with open(path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    if max_links:
        raw = raw[:max_links]

    out = []
    for X, y in raw:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if max_frames_per_link and len(y) > max_frames_per_link:
            idx = np.linspace(0, len(y) - 1, max_frames_per_link).astype(int)
            X, y = X[idx], y[idx]
        out.append((X, y))
    return out


def split_by_link(links, fractions=(0.7, 0.15, 0.15), seed=0):
    """Assign whole links to train/val/test. The deployment-realistic split.

    Returns:
        (train, val, test) as lists of link indices.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(links))
    a = int(len(links) * fractions[0])
    b = a + int(len(links) * fractions[1])
    return order[:a], order[a:b], order[b:]


def assemble(links, link_idx, clip_sigma=None):
    """Concatenate the chosen links into flat arrays.

    Args:
        clip_sigma: if given, clip every feature to +-this many standard
            deviations. Scrimmage 5 contains values past 50 sigma; left alone
            they dominate the loss and the network spends its capacity on a
            handful of frames. Clipping is applied per feature using limits
            derived from the TRAINING data only -- see fit_clip below.

    Returns:
        (X, y, link_id) where link_id says which link each frame came from,
        so per-link performance can be reported.
    """
    Xs, ys, ids = [], [], []
    for i in link_idx:
        X, y = links[i]
        Xs.append(X)
        ys.append(y)
        ids.append(np.full(len(y), i, dtype=np.int64))
    X = np.concatenate(Xs)
    if clip_sigma is not None:
        X = np.clip(X, -clip_sigma, clip_sigma)
    return X, np.concatenate(ys), np.concatenate(ids)


def split_by_frame(links, fractions=(0.7, 0.15, 0.15), seed=0):
    """Split frames WITHIN each link -- the pilot-phase scenario.

    Every link contributes to train, val and test. Easier than split_by_link
    and a legitimate problem in its own right (a radio does get to observe a
    link before predicting on it), but it answers a different question and
    must never be quoted as though it answered the harder one.

    Frames are time-ordered, so the split is contiguous: train on the early
    part of each link, test on the later part. A random within-link split
    would let the model interpolate between neighbouring frames milliseconds
    apart, which is a genuine leak rather than a pilot phase.

    Returns:
        (train, val, test) as lists of (link_index, slice) pairs.
    """
    train, val, test = [], [], []
    for i, (_, y) in enumerate(links):
        n = len(y)
        a = int(n * fractions[0])
        b = a + int(n * fractions[1])
        train.append((i, slice(0, a)))
        val.append((i, slice(a, b)))
        test.append((i, slice(b, n)))
    return train, val, test


def assemble_slices(links, pairs, clip_sigma=None):
    """Concatenate (link_index, slice) pairs into flat arrays."""
    Xs, ys, ids = [], [], []
    for i, sl in pairs:
        X, y = links[i]
        Xs.append(X[sl])
        ys.append(y[sl])
        ids.append(np.full(len(y[sl]), i, dtype=np.int64))
    X = np.concatenate(Xs)
    if clip_sigma is not None:
        X = np.clip(X, -clip_sigma, clip_sigma)
    return X, np.concatenate(ys), np.concatenate(ids)


def majority_baseline(y) -> float:
    """Accuracy of always predicting the more common class.

    The number any model must beat to have done anything. With 65-75% of
    frames succeeding, a model reporting 70% accuracy may have learned
    nothing at all.
    """
    p = float(np.mean(y))
    return max(p, 1.0 - p)
