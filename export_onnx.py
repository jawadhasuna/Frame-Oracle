"""Export the frame-error predictor to ONNX.

Darpa-Spect runs this in the browser: a visitor moves sliders for SNR, MCS,
bandwidth and the spectrum around them, and the page says whether the frame
would arrive. Tiny model, no server, instant response.

Exports the BY_LINK model by default. That is the one evaluated on links it
has never seen, so it is the only one whose score predicts behaviour on a new
link. The by_frame model scores higher and would be the wrong thing to ship:
its number assumes a pilot period on the very link being predicted, which a
web page has no way to provide.

Run:  uv run export_onnx.py
      uv run export_onnx.py --checkpoint scrimmage5_by_link
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_sc2 import FEATURE_NAMES
from train_frame_error import MLP


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="scrimmage4_by_link")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = get_args()
    tag = args.checkpoint
    out = Path(args.out or f"export/{tag}.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    ck = torch.load(Path("checkpoints") / f"{tag}.pt", map_location="cpu",
                    weights_only=False)
    a, m = ck["args"], ck["metrics"]

    model = MLP(ck["n_features"], tuple(a["hidden"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    print(f"checkpoint : {tag}")
    print(f"scenario   : {ck['scenario']}")
    print(f"features   : {ck['n_features']}")
    print(f"test AUC   : {m['auc']:.4f}   specificity {m['specificity']:.4f}")

    dummy = torch.randn(1, ck["n_features"])
    common = dict(input_names=["features"], output_names=["logit"],
                  dynamic_axes={"features": {0: "batch"},
                                "logit": {0: "batch"}},
                  opset_version=17)
    try:
        torch.onnx.export(model, (dummy,), str(out), dynamo=True, **common)
        print("exporter   : torch.export (dynamo)")
    except Exception as exc:
        print(f"exporter   : dynamo failed ({type(exc).__name__}), "
              f"falling back to TorchScript")
        torch.onnx.export(model, (dummy,), str(out), dynamo=False, **common)

    size_kb = out.stat().st_size / 1024
    print(f"\nexported   : {out}  ({size_kb:.0f} KB)")

    # --- verify against PyTorch ----------------------------------------------
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)

    # Random normal inputs are the right test here: the dataset's features are
    # already standardised, so N(0,1) is the distribution the model actually
    # sees rather than an artificial one.
    xb = rng.standard_normal((256, ck["n_features"])).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(xb)).numpy().ravel()
    # MLP.forward already squeezes, so the ONNX output may be (N,) or
    # (N, 1) depending on exporter. ravel() handles both.
    got = sess.run(["logit"], {"features": xb})[0].ravel()

    diff = float(np.abs(ref - got).max())
    same = float(((ref > 0) == (got > 0)).mean())
    print(f"\nmax logit difference : {diff:.2e}")
    print(f"decisions agreeing   : {same * 100:.2f}%")
    assert same == 1.0, "ONNX disagrees with PyTorch on the decision"
    assert diff < 1e-4, f"numerical drift too large: {diff:.2e}"
    print("export verified")

    # --- metadata the page needs ----------------------------------------------
    meta = {
        "task": "frame error prediction",
        "scenario": ck["scenario"],
        "features": ck["feature_names"],
        "input_name": "features",
        "output_name": "logit",
        "output_note": "raw logit; apply sigmoid for P(frame received)",
        "decision_threshold": m.get("best_threshold", 0.5),
        "threshold_note": "chosen on validation to maximise accuracy; "
                          "lower it to catch more failures at the cost of "
                          "false alarms",
        "preprocessing": "features are standardised (zero mean, unit variance) "
                         "using statistics from the SC2 dataset; the page must "
                         "supply values on that same scale",
        "test_metrics": {k: m[k] for k in
                         ("auc", "balanced_accuracy", "specificity", "recall",
                          "accuracy_tuned", "baseline_accuracy")},
        "honest_note": "AUC ~0.69-0.71 on unseen links across two scrimmages. "
                       "This is an information ceiling, not a tuning failure: "
                       "more data and more training both made it worse.",
        "size_kb": round(size_kb),
    }
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {out.with_suffix('.json')}")

    print("\nThe metadata carries the decision threshold and the honest AUC.")
    print("A demo that shows a confident yes/no without either would imply")
    print("far more certainty than this model has.")


if __name__ == "__main__":
    main()
