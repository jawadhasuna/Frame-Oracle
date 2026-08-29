"""Predict whether a frame will arrive, from real SC2 Colosseum measurements.

Runs both split scenarios so the difference between them is visible rather
than chosen quietly:

    by_link   train and test on DIFFERENT radio links   (deployment question)
    by_frame  pilot phase, then predict later frames    (easier, different)

Examples:
    uv run train_frame_error.py
    uv run train_frame_error.py --which scrimmage5 --epochs 40
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             roc_auc_score, roc_curve)

from data_sc2 import (FEATURE_NAMES, assemble, assemble_slices, load_links,
                      majority_baseline, split_by_frame, split_by_link)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--which", default="scrimmage4",
                   choices=["scrimmage4", "scrimmage5"])
    p.add_argument("--max-frames", type=int, default=4000,
                   help="frames sampled per link (0 = all)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 64, 32])
    p.add_argument("--clip-sigma", type=float, default=8.0,
                   help="clip features to +-N sigma; scrimmage5 has 50-sigma "
                        "outliers that otherwise dominate the loss")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


class MLP(nn.Module):
    """A plain feed-forward net over the 20 tabular features.

    Tabular data with 20 columns does not need convolutions or attention;
    it needs enough capacity and honest regularisation. Batch norm keeps the
    PSD columns (which have quite different scales even after the dataset's
    own standardisation) from dominating early training.
    """

    def __init__(self, n_in=20, hidden=(128, 64, 32), dropout=0.2):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def best_accuracy_threshold(y_true, prob):
    """The decision threshold that maximises plain accuracy.

    Training uses pos_weight to counter class imbalance, which deliberately
    biases the model toward predicting failures -- good for catching them,
    bad for raw accuracy at a 0.5 cut. Reporting accuracy at 0.5 therefore
    understates what the model can do, while reporting only the tuned number
    hides the operating point actually used. Both get reported.

    The threshold MUST be chosen on validation data and then applied to test.
    Picking it on test is a small but real leak: it selects the one cut point
    that happens to suit the test set, which is not a number that survives
    deployment.
    """
    order = np.argsort(prob)
    p, y = prob[order], y_true[order]
    # Sweep every candidate cut in one pass: below the cut predict 0, above 1.
    n_pos_above = np.cumsum(y[::-1])[::-1]
    n_neg_below = np.cumsum(1 - y) - (1 - y)
    correct = n_pos_above + n_neg_below
    i = int(np.argmax(correct))
    return float(p[i]), float(correct[i] / len(y))


def metrics(y_true, prob, threshold=0.5):
    """Accuracy alone is misleading at 65-75% class imbalance, so report the
    metrics that are not fooled by predicting the majority class."""
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "accuracy": float((pred == y_true).mean()),
        "balanced_accuracy": float((recall + specificity) / 2),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-9)),
        "auc": float(roc_auc_score(y_true, prob)),
        "avg_precision": float(average_precision_score(y_true, prob)),
        "threshold": float(threshold),
    }


def run(scenario, links, args, device):
    """Train and evaluate one split scenario."""
    if scenario == "by_link":
        tr, va, te = split_by_link(links, seed=args.seed)
        pack = lambda idx: assemble(links, idx, clip_sigma=args.clip_sigma)
    else:
        tr, va, te = split_by_frame(links, seed=args.seed)
        pack = lambda pairs: assemble_slices(links, pairs,
                                             clip_sigma=args.clip_sigma)

    Xtr, ytr, _ = pack(tr)
    Xva, yva, _ = pack(va)
    Xte, yte, ids_te = pack(te)

    print(f"\n{'=' * 62}")
    print(f"scenario: {scenario}")
    print("=" * 62)
    print(f"train {len(ytr):>9,} frames   positive {ytr.mean() * 100:.2f}%")
    print(f"val   {len(yva):>9,} frames   positive {yva.mean() * 100:.2f}%")
    print(f"test  {len(yte):>9,} frames   positive {yte.mean() * 100:.2f}%")
    base = majority_baseline(yte)
    print(f"majority-class baseline on test: {base * 100:.2f}%")

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).float().to(device)
    Xva_t = torch.from_numpy(Xva).to(device)
    Xte_t = torch.from_numpy(Xte).to(device)

    model = MLP(Xtr.shape[1], tuple(args.hidden)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Class weighting rather than resampling: it is equivalent in effect for
    # this imbalance level, costs no extra memory, and does not invent
    # synthetic frames. SMOTE would be the alternative if the minority class
    # were much rarer than 25-35%.
    pos_weight = torch.tensor([(1 - ytr.mean()) / max(ytr.mean(), 1e-9)],
                              device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    @torch.no_grad()
    def prob_of(X_t, bs=200_000):
        model.eval()
        out = []
        for i in range(0, len(X_t), bs):
            out.append(torch.sigmoid(model(X_t[i:i + bs])).cpu())
        return torch.cat(out).numpy()

    best_auc, best_state = 0.0, None
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(ytr_t), device=device)
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad(set_to_none=True)
            lossf(model(Xtr_t[b]), ytr_t[b]).backward()
            opt.step()
        sched.step()

        va_auc = roc_auc_score(yva, prob_of(Xva_t))
        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:>3}  val AUC {va_auc:.4f}"
                  f"{'  <- best' if va_auc == best_auc else ''}")

    model.load_state_dict(best_state)

    # Choose the operating point on VALIDATION, then apply it to test.
    thr_best, _ = best_accuracy_threshold(yva, prob_of(Xva_t))

    prob = prob_of(Xte_t)
    m = metrics(yte, prob)
    acc_best = float(((prob >= thr_best).astype(int) == yte).mean())
    m["best_threshold"] = thr_best
    m["accuracy_tuned"] = acc_best
    m["baseline_accuracy"] = float(base)
    m["train_seconds"] = time.time() - t0
    m["n_test"] = int(len(yte))

    print(f"\n  trained in {m['train_seconds']:.0f}s")
    print(f"  {'accuracy @ 0.5':<20} {m['accuracy'] * 100:>7.2f}%   "
          f"(baseline {base * 100:.2f}%, "
          f"{(m['accuracy'] - base) * 100:+.2f} points)")
    print(f"  {'accuracy @ tuned':<20} {acc_best * 100:>7.2f}%   "
          f"(threshold {thr_best:.3f} chosen on val, "
          f"{(acc_best - base) * 100:+.2f} points)")
    print(f"  {'balanced accuracy':<20} {m['balanced_accuracy'] * 100:>7.2f}%   "
          f"(baseline 50.00%)")
    print(f"  {'AUC':<20} {m['auc']:>8.4f}   (baseline 0.5)")
    print(f"  {'recall (success)':<20} {m['recall'] * 100:>7.2f}%")
    print(f"  {'specificity (error)':<20} {m['specificity'] * 100:>7.2f}%")
    print(f"  {'F1':<20} {m['f1']:>8.4f}")

    return m, (yte, prob, ids_te)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")
    print(f"loading {args.which}...")
    links = load_links(args.which,
                       max_frames_per_link=args.max_frames or None)
    total = sum(len(y) for _, y in links)
    print(f"{len(links)} links, {total:,} frames, "
          f"{len(FEATURE_NAMES)} features")

    results, curves = {}, {}
    for scenario in ("by_link", "by_frame"):
        results[scenario], curves[scenario] = run(scenario, links, args, device)

    # --- the comparison that matters -----------------------------------------
    bl, bf = results["by_link"], results["by_frame"]
    print(f"\n{'=' * 62}")
    print("the two scenarios are NOT interchangeable")
    print("=" * 62)
    print(f"{'metric':<22} {'by_link':>10} {'by_frame':>10} {'gap':>9}")
    print("-" * 54)
    for k in ("accuracy", "accuracy_tuned", "balanced_accuracy", "auc", "f1"):
        scale = 100 if k != "auc" else 1
        print(f"{k:<22} {bl[k] * scale:>10.2f} {bf[k] * scale:>10.2f} "
              f"{(bf[k] - bl[k]) * scale:>+9.2f}")

    print("\nby_frame is the easier problem: every link is partly known.")
    print("by_link is the deployment question -- an unseen link, no history.")
    print("Quoting the higher number without saying which split produced it")
    print("is the most common way these results get overstated.")

    Path("results").mkdir(exist_ok=True)
    with open(Path("results") / f"{args.which}_frame_error.json", "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)

    # --- plot -----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for scenario, (yte, prob, _) in curves.items():
        fpr, tpr, _ = roc_curve(yte, prob)
        ax1.plot(fpr, tpr, lw=2,
                 label=f"{scenario}  AUC {results[scenario]['auc']:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="chance")
    ax1.set_xlabel("false positive rate")
    ax1.set_ylabel("true positive rate")
    ax1.set_title(f"Frame-error prediction, {args.which}\nROC by split scenario")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    keys = ["accuracy_tuned", "balanced_accuracy", "auc", "f1"]
    x = np.arange(len(keys))
    ax2.bar(x - 0.2, [bl[k] for k in keys], 0.4, label="by_link")
    ax2.bar(x + 0.2, [bf[k] for k in keys], 0.4, label="by_frame")
    ax2.axhline(bl["baseline_accuracy"], ls="--", color="red", alpha=0.6,
                label="majority baseline")
    ax2.set_xticks(x, [k.replace("_", "\n") for k in keys], fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Same data, same model, different split")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = Path("figures")
    out.mkdir(exist_ok=True)
    fig.savefig(out / f"{args.which}_frame_error.png", dpi=140)
    print(f"\nsaved {out / f'{args.which}_frame_error.png'}")
    plt.show()


if __name__ == "__main__":
    main()
