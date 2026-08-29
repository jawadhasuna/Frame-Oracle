"""Spectrum sensing: deciding whether a channel is occupied.

Two detectors, and the difference between them is the whole point.

ENERGY DETECTION compares received power against a fixed threshold. Simple,
optimal when you know the noise power exactly, and useless when you do not --
which is always. Noise floors drift with temperature, gain settings, and
whatever else is on the air. A threshold set for one noise level produces
either constant false alarms or constant misses at another.

CA-CFAR (Cell-Averaging Constant False Alarm Rate) estimates the noise from
the cells AROUND the one being tested, and scales its threshold accordingly.
The false alarm rate stays at the designed value even as the noise floor
moves. That property is what the name promises and what makes it usable.

The SCATTER PHY paper uses CA-CFAR as the second stage of its synchronisation
detector, after plain correlation, reporting better detection at low SNR than
correlation alone.

Convention throughout: "power" means a 1-D array of per-cell power values,
typically PSD bins across frequency.
"""

import numpy as np


def cfar_alpha(n_train: int, pfa: float, n_looks: int = 1) -> float:
    """Threshold multiplier for CA-CFAR with a square-law detector.

    The familiar textbook form

        alpha = N * (Pfa^(-1/N) - 1)

    is only valid for SINGLE-LOOK data, where each cell's power is
    exponentially distributed. That is one periodogram, not an averaged one.

    Averaging L periodograms (Welch) makes each cell Gamma(L) distributed and
    far less variable -- which is the point of averaging, but it means the
    single-look constant is enormously too conservative. Applying it to
    780-averaged data produces a threshold so high that nothing is ever
    detected, false alarms and real signals alike.

    In general, the ratio of the cell under test to the training-cell average
    follows an F distribution with (2L, 2NL) degrees of freedom, so:

        alpha = F^-1(1 - Pfa; 2L, 2NL)

    which reduces exactly to the closed form at L = 1. That equivalence is
    asserted in demo_cfar.py rather than assumed.

    Args:
        n_train: TOTAL training cells (both sides pooled).
        pfa: designed probability of false alarm.
        n_looks: how many periodograms were averaged to make each cell.

    Returns:
        Threshold multiplier applied to the noise estimate.
    """
    if not 0.0 < pfa < 1.0:
        raise ValueError("pfa must be in (0, 1)")
    if n_looks < 1:
        raise ValueError("n_looks must be at least 1")

    if n_looks == 1:
        return n_train * (pfa ** (-1.0 / n_train) - 1.0)

    from scipy.stats import f as f_dist

    return float(f_dist.isf(pfa, 2 * n_looks, 2 * n_train * n_looks))


def ca_cfar(power: np.ndarray, n_train: int = 16, n_guard: int = 2,
            pfa: float = 1e-3, n_looks: int = 1):
    """Cell-averaging CFAR across a 1-D power array.

    For each cell, the noise level is estimated from n_train cells on each
    side, skipping n_guard cells immediately adjacent.

    The guard cells matter: a real signal is wider than one bin, so its own
    energy leaks into neighbouring cells. Without guards, a strong signal
    inflates its own noise estimate and hides itself -- the detector is
    blinded by exactly what it is looking for.

    Args:
        power: per-cell power, e.g. PSD bins.
        n_train: training cells per side.
        n_guard: guard cells per side.
        pfa: designed probability of false alarm.
        n_looks: periodograms averaged per cell. Must match how `power` was
            produced -- welch_psd returns this. Passing 1 for averaged data
            makes the threshold far too high and the detector blind.

    Returns:
        (detections, threshold) both the same length as power. Edge cells
        without a full window are marked False and given inf threshold, so
        they never produce a detection.
    """
    power = np.asarray(power, dtype=np.float64)
    n = len(power)
    half = n_train + n_guard

    detections = np.zeros(n, dtype=bool)
    threshold = np.full(n, np.inf)

    if n < 2 * half + 1:
        return detections, threshold

    alpha = cfar_alpha(2 * n_train, pfa, n_looks)  # both sides pooled

    # Cumulative sums make the sliding window O(n) instead of O(n * n_train).
    csum = np.concatenate([[0.0], np.cumsum(power)])

    def window_sum(lo, hi):
        """Sum of power[lo:hi] for arrays of indices."""
        return csum[hi] - csum[lo]

    idx = np.arange(half, n - half)
    left = window_sum(idx - half, idx - n_guard)
    right = window_sum(idx + n_guard + 1, idx + half + 1)
    noise_est = (left + right) / (2.0 * n_train)

    threshold[idx] = alpha * noise_est
    detections[idx] = power[idx] > threshold[idx]

    return detections, threshold


def energy_detector(power: np.ndarray, noise_power: float,
                    pfa: float = 1e-3, n_looks: int = 1):
    """Fixed-threshold energy detection, given an assumed noise power.

    The threshold is -ln(Pfa) * noise_power, exact for exponentially
    distributed cell power.

    This is the detector CFAR replaces. It is included so the failure can be
    demonstrated rather than asserted: pass a noise_power that is wrong -- as
    it will be in any real deployment -- and watch the false alarm rate leave
    its design value entirely.
    """
    if n_looks == 1:
        threshold = -np.log(pfa) * noise_power
    else:
        # Gamma(L) cell statistics, same correction as CFAR needs.
        from scipy.stats import gamma
        threshold = noise_power * float(gamma.isf(pfa, a=n_looks,
                                                  scale=1.0 / n_looks))
    return np.asarray(power) > threshold, threshold


def roc_points(signal_power, noise_only_power, pfas, detector="cfar",
               **kwargs):
    """Trace detection probability against measured false alarm rate.

    Pd is measured on cells that contain a signal; Pfa is measured on a
    separate noise-only capture. Measuring Pfa on the same array that holds
    the signal would count the signal's own bins as false alarms.

    Returns:
        (measured_pfa, pd) arrays, one point per entry in pfas.
    """
    measured_pfa, pd = [], []

    for pfa in pfas:
        if detector == "cfar":
            det_sig, _ = ca_cfar(signal_power, pfa=pfa, **kwargs)
            det_noise, _ = ca_cfar(noise_only_power, pfa=pfa, **kwargs)
            valid = np.isfinite(ca_cfar(noise_only_power, pfa=pfa,
                                        **kwargs)[1])
            measured_pfa.append(det_noise[valid].mean())
            pd.append(det_sig[valid].mean())
        else:
            det_sig, _ = energy_detector(signal_power, pfa=pfa, **kwargs)
            det_noise, _ = energy_detector(noise_only_power, pfa=pfa, **kwargs)
            measured_pfa.append(det_noise.mean())
            pd.append(det_sig.mean())

    return np.array(measured_pfa), np.array(pd)
