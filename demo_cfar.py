"""Does CFAR actually keep the false alarm rate constant?

The claim is in the name. This tests it, and tests the detector it replaces
under the same conditions.

Run:  uv run demo_cfar.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sensing import ca_cfar, cfar_alpha, energy_detector
from signals import noise, occupied_band, welch_psd

N = 200_000
NPERSEG = 512
N_TRAIN, N_GUARD = 16, 4

rng = np.random.default_rng(0)

# --- 0. the generalisation must reduce to the textbook form at L = 1 ---------
# cfar_alpha uses a closed form for single-look data and an F distribution
# otherwise. If those disagree at L = 1, one of them is wrong.
print("checking cfar_alpha: F-distribution form vs closed form at L = 1")
from scipy.stats import f as f_dist

for n_tr in (8, 32, 64):
    for pfa in (1e-1, 1e-3, 1e-5):
        closed = n_tr * (pfa ** (-1.0 / n_tr) - 1.0)
        via_f = float(f_dist.isf(pfa, 2, 2 * n_tr))
        assert abs(closed - via_f) / closed < 1e-9, (n_tr, pfa, closed, via_f)
print("  they agree to 1e-9 -- the generalisation is consistent\n")

# --- 1. does the designed false alarm rate come out? -------------------------
_, probe, n_looks = welch_psd(noise(N, 1.0, rng=rng), nperseg=NPERSEG)
print(f"Welch averaging: {n_looks} periodograms per PSD bin")
print(f"single-look alpha at Pfa 1e-3 would be "
      f"{cfar_alpha(2 * N_TRAIN, 1e-3, 1):.2f}; "
      f"correct value is {cfar_alpha(2 * N_TRAIN, 1e-3, n_looks):.2f}")
print("Using the single-look constant on averaged data sets the threshold")
print("about 7x too high, and the detector finds nothing at all.\n")

print("designed vs measured false alarm rate (noise only, no signal)")
print(f"{'designed Pfa':>13} {'measured':>11} {'ratio':>8} {'alpha':>9}")
print("-" * 45)

for pfa in [1e-1, 1e-2, 1e-3, 1e-4]:
    measured = []
    for trial in range(40):
        nz = noise(N, 1.0, rng=np.random.default_rng(100 + trial))
        _, psd, L = welch_psd(nz, nperseg=NPERSEG)
        det, thr = ca_cfar(psd, N_TRAIN, N_GUARD, pfa, n_looks=L)
        measured.append(det[np.isfinite(thr)].mean())
    m = float(np.mean(measured))
    print(f"{pfa:>13.0e} {m:>11.2e} {m / pfa:>7.2f}x "
          f"{cfar_alpha(2 * N_TRAIN, pfa, n_looks):>9.3f}")

print("\nRatios near 1 mean the detector delivers what it promises.")
print("Some drift is expected: Welch segments overlap by 50%, so adjacent")
print("bins are correlated and the training cells are not fully independent")
print("as the theory assumes.")

# --- 2. the reason CFAR exists ------------------------------------------------
print("\n\nnoise floor moves, detectors stay put")
print("(energy detector calibrated once at 0 dB noise, Pfa 1e-3)")
print(f"\n{'noise floor':>12} {'CFAR Pfa':>11} {'energy Pfa':>12}")
print("-" * 38)

floors_db = [-10, -5, 0, 5, 10]
cfar_pfa, energy_pfa = [], []

for fdb in floors_db:
    npow = 10.0 ** (fdb / 10.0)
    c_hits, e_hits = [], []
    for trial in range(30):
        nz = noise(N, npow, rng=np.random.default_rng(200 + trial))
        _, psd, L = welch_psd(nz, nperseg=NPERSEG)
        det_c, thr = ca_cfar(psd, N_TRAIN, N_GUARD, 1e-3, n_looks=L)
        valid = np.isfinite(thr)
        c_hits.append(det_c[valid].mean())
        # Calibrated at 0 dB noise and never updated -- the realistic case.
        # welch_psd normalises by window energy, so noise power 1.0 gives
        # PSD bins averaging 1.0.
        det_e, _ = energy_detector(psd[valid], noise_power=1.0, pfa=1e-3,
                                   n_looks=L)
        e_hits.append(det_e.mean())
    cfar_pfa.append(np.mean(c_hits))
    energy_pfa.append(np.mean(e_hits))
    print(f"{fdb:>9} dB {np.mean(c_hits):>11.2e} {np.mean(e_hits):>12.2e}")

print("\nCFAR holds near its design point across a 20 dB swing in noise floor.")
print("The fixed threshold cannot: it was calibrated for a noise level that")
print("is no longer true, so it either cries wolf or goes blind.")

# --- 3. finding occupied channels in a band ----------------------------------
channels = [-0.30, -0.10, 0.15, 0.35]
snrs = [15, 6, 0, -4]
sig, occ = occupied_band(N, channels, snrs, noise_power=1.0, rng=rng)
freqs, psd, L = welch_psd(sig, nperseg=NPERSEG)
det, thr = ca_cfar(psd, N_TRAIN, N_GUARD, pfa=1e-3, n_looks=L)

print("\n\nfinding occupied channels (Pfa 1e-3)")
print(f"{'channel':>9} {'SNR':>7} {'detected':>10}")
print("-" * 28)
near_any = np.zeros(len(psd), dtype=bool)
for f, s in zip(channels, snrs):
    b = int(np.argmin(np.abs(freqs - f)))
    lo, hi = max(0, b - 3), min(len(psd), b + 4)
    near_any[lo:hi] = True
    print(f"{f:>9.2f} {s:>4} dB {'yes' if det[lo:hi].any() else 'no':>10}")

false_bins = int(det[~near_any].sum())
valid_far = int(np.isfinite(thr)[~near_any].sum())
print(f"\nbins flagged where nothing is transmitting: {false_bins} of "
      f"{valid_far}  ({false_bins / max(valid_far, 1):.2e})")

# --- pictures -----------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8.5))

ax1.plot(freqs, 10 * np.log10(psd + 1e-20), lw=1.0, label="PSD", alpha=0.85)
finite = np.isfinite(thr)
ax1.plot(freqs[finite], 10 * np.log10(thr[finite] + 1e-20), lw=1.6,
         color="tab:red", label="CFAR threshold")
ax1.scatter(freqs[det], 10 * np.log10(psd[det] + 1e-20), s=18,
            color="tab:green", zorder=4, label="detections")
for f, s in zip(channels, snrs):
    ax1.axvline(f, ls=":", color="grey", alpha=0.6)
    ax1.text(f, ax1.get_ylim()[1], f" {s} dB", fontsize=8, va="top",
             color="grey")
ax1.set_title("CA-CFAR over a band with four occupied channels\n"
              "the threshold follows the noise, so weak signals still stand out")
ax1.set_xlabel("normalised frequency")
ax1.set_ylabel("power (dB)")
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(alpha=0.3)

ax2.semilogy(floors_db, np.maximum(cfar_pfa, 1e-9), "o-", lw=2,
             label="CA-CFAR")
ax2.semilogy(floors_db, np.maximum(energy_pfa, 1e-9), "s--", lw=2,
             label="fixed-threshold energy detector")
ax2.axhline(1e-3, ls=":", color="black", alpha=0.7, label="designed Pfa = 1e-3")
ax2.set_xlabel("noise floor (dB)")
ax2.set_ylabel("measured false alarm rate")
ax2.set_title("Why CFAR exists: the noise floor is never what you calibrated for")
ax2.set_ylim(1e-9, 1.5)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3, which="both")

fig.tight_layout()
out = Path("figures")
out.mkdir(exist_ok=True)
fig.savefig(out / "cfar.png", dpi=140)
print(f"\nsaved {out / 'cfar.png'}")
plt.show()
