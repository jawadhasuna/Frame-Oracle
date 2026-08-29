# Frame-Oracle

Spectrum sensing and frame-error prediction, on real DARPA SC2 Colosseum
measurements.

Repo 3 of six in a DARPA Spectrum Collaboration Challenge project. Where
[Mod-Scope](https://github.com/jawadhasuna/Mod-Scope) asks *what signal is
this*, Frame-Oracle asks the two questions a radio actually needs answered:
**is this channel free**, and **will my transmission survive**.

> Status: Part A (sensing) and Part B (frame-error prediction) complete.

## Part A: CA-CFAR spectrum sensing

Energy detection compares power to a fixed threshold. It is optimal when you
know the noise power exactly, and useless otherwise -- which is always, since
noise floors drift with temperature, gain and whatever else is on the air.

CA-CFAR estimates noise from the cells *around* the one under test, so the
false alarm rate stays where you designed it. The SCATTER PHY paper uses
CA-CFAR as the second stage of its synchronisation detector for this reason.

Measured, holding Pfa at a designed 1e-3 while the noise floor moves 20 dB:

```
noise floor    CA-CFAR      fixed threshold
   -10 dB     1.41e-03            0.00e+00     blind
    -5 dB     1.41e-03            0.00e+00     blind
     0 dB     1.41e-03            9.18e-04     correct (calibrated here)
    +5 dB     1.41e-03            1.00e+00     every bin a false alarm
   +10 dB     1.41e-03            1.00e+00     useless
```

CFAR is constant to three digits. The fixed threshold is right at exactly one
noise level. Over a band with four occupied channels at 15, 6, 0 and -4 dB
SNR, CFAR found all four -- including the one weaker than the noise -- with
zero false alarms across 444 empty bins.

### The bug worth documenting

The textbook CFAR constant

```
alpha = N * (Pfa^(-1/N) - 1)
```

is only valid for **single-look** data, where each cell is exponentially
distributed. Welch averaging of 780 periodograms makes cells Gamma(780)
distributed and far less variable, so the single-look constant sets the
threshold about 7x too high: at Pfa 1e-3 it wants alpha 7.71 when the correct
value is 1.12.

The first working version therefore detected **nothing at all** -- zero false
alarms at every setting, which looks like success if you are not watching.

The general form uses an F distribution with (2L, 2NL) degrees of freedom and
reduces exactly to the closed form at L = 1. `demo_cfar.py` asserts that
equivalence rather than assuming it: a generalisation that cannot reproduce
the special case it generalises is wrong, and the check costs three lines.

## Part B: frame-error prediction on real SC2 data

Data from Jameel, Mohamed, Zhang and El Gamal (arXiv 2005.01446), collected on
the Colosseum testbed during SC2 scrimmages. Every record is one transmitted
frame:

```
features : SNR, MCS, centre frequency, bandwidth, 16 PSD bins   (20 total)
label    : 1 = received, 0 = frame error
scale    : 284 links / 6.5M frames (S4), 627 links / 11.7M frames (S5)
```

Those 16 PSD bins are the same quantity the SCATTER PHY reports upward to its
AI layer, and the same quantity Part A runs a detector on.

### The split is the whole experiment

Frames are grouped by radio link, and per-link success rates run from 0.26 to
1.00. A random split across frames puts the same link in train and test, so a
model can score well by recognising **which link** it is looking at -- easy
from centre frequency and bandwidth -- without learning anything about frames.

Two scenarios, both reported, matching the paper's distinction:

- **by_link** -- train and test on entirely different links. The deployment
  question: an unseen link, no history.
- **by_frame** -- a pilot period on each link, then predict its later frames.
  Legitimate and easier. Split contiguously in time, not randomly, since
  neighbouring frames are milliseconds apart.

### Results, scrimmage 4

```
metric                    by_link    by_frame
AUC                        0.7077      0.7936
balanced accuracy          63.93%      71.91%
specificity (failures)     67.86%      77.13%
accuracy @ tuned           74.35%      74.47%
majority baseline          70.43%      62.40%
gain over baseline          +3.92      +12.07
```

**AUC 0.71 on links never seen before.** Chance is 0.5, so the model learned
something transferable about what makes a frame survive, not just which link
it was shown.

**Specificity 67.86%**: two of every three impending failures are flagged
before transmitting. That is the operationally useful figure -- a radio that
expects a failure can drop MCS or change channel instead of spending the
airtime.

### Why the accuracy row is a trap

`accuracy @ tuned` is 74.35% and 74.47% -- apparently identical, apparently
showing the two scenarios are equally hard. They are not. The two test sets
have different class balances (70.43% vs 62.40% positive), so the baselines
differ and the raw accuracies are not comparable. Measured as gain over
baseline, by_frame is **three times** better. AUC, which is insensitive to
both threshold and class balance, agrees: 0.79 against 0.71.

This is the clearest demonstration in the project of why accuracy is a poor
headline metric, and it is offered as a caution rather than a lecture -- the
identical-looking numbers were nearly reported as equivalent.

### Two caveats stated rather than buried

**The features arrive pre-standardised**, and the standardisation was fitted
over the whole dataset before any split, so test statistics are present in the
training features. Unavoidable without the raw data, but it means these
numbers are not like-for-like against a pipeline that normalises after
splitting.

**The decision threshold is chosen on validation and applied to test.**
Choosing it on test would select the cut that happens to suit the test set --
a small leak, and one the first version of this code committed before it was
caught.

## Run it

```bash
uv run demo_cfar.py                # sensing: does CFAR hold its Pfa?
uv run inspect_sc2.py              # structure, class balance, feature scales
uv run train_frame_error.py        # both split scenarios, scrimmage 4
uv run train_frame_error.py --which scrimmage5 --epochs 40
```

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. A CUDA GPU helps
but is not required.

The SC2 pickles are not included -- 1.6 GB, and not ours to redistribute.
Download the **Link Prediction** sets from
[github.com/amahdeej/sc2-frame-error](https://github.com/amahdeej/sc2-frame-error)
and place them beside this repo. Note they contain PyTorch tensors, so
unpickling requires torch and executes code; fine for a published academic
dataset, worth knowing in general.

## Layout

| File | Purpose |
|------|---------|
| `sensing.py` | CA-CFAR and energy detection, with the L-look correction |
| `signals.py` | Controlled test signals: tones, occupied bands, Welch PSD |
| `demo_cfar.py` | Verifies Pfa is constant as the noise floor moves |
| `inspect_sc2.py` | Structure, class balance, feature scales, outliers |
| `data_sc2.py` | Loading and the two split scenarios |
| `train_frame_error.py` | MLP predictor, both scenarios, ROC and metrics |

## References

- A. S. M. M. Jameel, A. P. Mohamed, X. Zhang, A. El Gamal, "Deep Learning for
  Frame Error Prediction using a DARPA Spectrum Collaboration Challenge (SC2)
  Dataset," IEEE Networking Letters, 2021. arXiv:2005.01446
- F. A. P. de Figueiredo et al., "SCATTER PHY: An Open Source Physical Layer
  for the DARPA Spectrum Collaboration Challenge," *Electronics* 8(11), 2019.

Independent work. Not affiliated with or endorsed by DARPA.
