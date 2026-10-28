"""Mechanics demo: two deployed-model windows, one stable and one drifted.

This is a synthetic demonstration of what DriftReceipt measures, not a
benchmark. A "model" is simulated by a generator that emits logged
predictions (probabilities, a source tag, a brightness statistic, and the
true label) for a 3-class problem:

* **Stable pair** — reference and current windows drawn from the *same*
  distribution (different seeds). Expected: GO.
* **Drifted pair** — the current window plants two changes:
  (1) class 'ring' grows from ~33% to ~62% of traffic, and
  (2) on `scanner_b` rows the model's confidence collapses from ~0.87 to
  ~0.52 while its predictions stay just as accurate — the classic
  silent-calibration-rot failure. Expected: NO-GO, with the top slice
  naming `scanner_b`.

The script prints both receipts, saves them under ``examples/output/``,
and exits non-zero if either outcome differs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from driftreceipt import compare_windows  # noqa: E402

CLASS_NAMES = ["disc", "square", "ring"]
BASE_SHARES = np.array([0.34, 0.33, 0.33])
DRIFT_SHARES = np.array([0.19, 0.19, 0.62])


def make_window(rng: np.random.Generator, n: int, drift: bool) -> dict:
    """Simulated prediction log for one window."""
    shares = DRIFT_SHARES if drift else BASE_SHARES
    true_label = rng.choice(3, size=n, p=shares)
    # The simulated model is right 90% of the time in both windows.
    correct = rng.random(n) < 0.90
    pred = np.where(correct, true_label, (true_label + 1 + rng.integers(0, 2, size=n)) % 3)
    source = np.where(rng.random(n) < 0.40, "scanner_b", "scanner_a")
    brightness = rng.uniform(0.2, 0.9, size=n)

    conf = np.clip(rng.normal(0.87, 0.04, size=n), 0.60, 0.97)
    if drift:
        collapsed = source == "scanner_b"
        conf = np.where(
            collapsed, np.clip(rng.normal(0.52, 0.05, size=n), 0.40, 0.75), conf
        )
    probs = np.full((n, 3), 0.0)
    rest = (1.0 - conf) / 2.0
    for k in range(3):
        probs[:, k] = rest
    probs[np.arange(n), pred] = conf
    return {
        "probs": probs,
        "labels": true_label,
        "meta": {"source": source, "brightness": brightness},
    }


def run_pair(name: str, ref: dict, cur: dict) -> object:
    receipt = compare_windows(
        ref["probs"],
        cur["probs"],
        ref_meta=ref["meta"],
        cur_meta=cur["meta"],
        ref_labels=ref["labels"],
        cur_labels=cur["labels"],
        class_names=CLASS_NAMES,
    )
    print(f"\n##### {name} #####")
    print(receipt)
    out = Path(__file__).resolve().parent / "output"
    out.mkdir(exist_ok=True)
    stem = "stable" if name == "STABLE" else "drifted"
    (out / f"{stem}_receipt.txt").write_text(receipt.summary() + "\n")
    (out / f"{stem}_receipt.json").write_text(receipt.to_json(indent=2) + "\n")
    return receipt


def main() -> int:
    rng = np.random.default_rng(7)
    ref = make_window(rng, 1200, drift=False)

    stable_cur = make_window(np.random.default_rng(8), 1200, drift=False)
    stable = run_pair("STABLE", ref, stable_cur)

    drifted_cur = make_window(np.random.default_rng(9), 1200, drift=True)
    drifted = run_pair("DRIFTED", ref, drifted_cur)

    ok = True
    if stable.verdict != "go":
        print(f"\nFAIL: stable pair verdict is {stable.verdict!r}, expected 'go'")
        ok = False
    if drifted.verdict != "no-go":
        print(f"\nFAIL: drifted pair verdict is {drifted.verdict!r}, expected 'no-go'")
        ok = False
    top_slice = drifted.slices[0].definition if drifted.slices else ""
    if "scanner_b" not in top_slice:
        print(f"\nFAIL: drifted top slice is {top_slice!r}, expected it to name scanner_b")
        ok = False
    growing = drifted.checks["prediction_shift"].details["largest_shift_class"]
    if growing != "ring":
        print(f"\nFAIL: largest shift class is {growing!r}, expected 'ring'")
        ok = False
    print("\nDemo outcome:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
