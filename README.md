# DriftReceipt

**Your deployed classifier changed — here is the receipt: one call on two windows of logged predictions returns a GO / CAUTION / NO-GO verdict, what shifted, in which metadata slice, and the exact rows driving it.**

DatasetReceipt audits a dataset *before* training. DriftReceipt is its production sibling: it audits the *behaviour of a deployed model* between two windows. Aggregate accuracy can sit perfectly still while the model quietly changes underneath — one class swallowing the traffic, confidence rotting on one scanner while accuracy holds. DriftReceipt reads the prediction log you already have and prints the receipt.

## Install

DriftReceipt is not on PyPI yet. Install the GitHub version:

```
pip install "git+https://github.com/habib-analyst/driftreceipt.git"
```

For development:

```
git clone https://github.com/habib-analyst/driftreceipt.git
cd driftreceipt
python -m pip install -e ".[dev]"
python -m pytest
```

Requires Python 3.10+. The only runtime dependency is NumPy.

## Quickstart

```
import numpy as np

from driftreceipt import compare_windows

receipt = compare_windows(
    ref_probs, cur_probs,                      # (n, n_classes) logged probabilities
    ref_meta={"source": ref_sources},          # optional, per window
    cur_meta={"source": cur_sources},
    ref_labels=ref_y, cur_labels=cur_y,        # optional, per window
    class_names=["disc", "square", "ring"],
)

print(receipt)
# DRIFT RECEIPT — reference (n=1200) vs current (n=1200)
# VERDICT: NO-GO
# prediction_shift   NO-GO    ... largest share shift +0.257 ('ring')
# confidence         NO-GO    mean confidence 0.869 -> 0.726 ...
# performance        NO-GO    accuracy 0.899 -> 0.908 ...; ECE 0.041 -> 0.196 ...
# slices             NO-GO    worst slice: source == 'scanner_b' ...

receipt.verdict                      # "go" | "caution" | "no-go"
receipt.slices[0].definition         # "source == 'scanner_b'"
receipt.drivers["growing_class"]     # current rows behind the 'ring' surge
receipt.to_json()                    # diff this week's receipt against last week's
```

## The four checks

- **`prediction_shift`** — per-class predicted-share deltas and the Jensen–Shannon divergence between the two windows' prediction distributions (plus the soft, mean-probability JS in the details).
- **`confidence`** — mean-confidence delta and the JS divergence between the windows' confidence histograms, with mean predictive entropy in the details.
- **`performance`** — accuracy and ECE deltas, *only when labels are supplied for both windows*. Without labels the check reports **skipped** with the reason — never a silent pass.
- **`slices`** — the worst behaviour change inside readable metadata slices (below).

Every threshold (JS, share, confidence, accuracy, ECE, per-slice) is a parameter with a sensible default, and the full configuration is recorded on the receipt, so a verdict can be re-derived or challenged.

## Slices and drivers

Metadata columns shared by both windows become slices in the SliceSniff style: categorical equality (`source == 'scanner_b'`), numeric quantile bands whose edges come from the *reference* window (`0.46 <= brightness < 0.6262`), and missing-value predicates. For each slice with enough rows in both windows, the receipt measures the predicted-distribution JS, the confidence delta, and — with labels — the accuracy delta, and ranks the worst first.

The **drivers** name the evidence: the current-window row indices predicted into the fastest-growing class (most confident first), and the least-confident current rows inside the worst slice. A receipt is a starting point for an investigation, with the rows already attached.

## What DriftReceipt is not

- **Not a monitoring platform.** Evidently, Deepchecks, NannyML, whylogs/WhyLabs, and Fiddler/Arize-class platforms watch raw features and models continuously, with dashboards, alerting, and (for the hosted ones) accounts and servers. DriftReceipt needs none of that surface because it reads *prediction logs only*: no raw features, no model object, no DataFrame schema, no timestamps or chunking — two NumPy arrays and one function call, built for a CI step or a cron job that already exports probabilities.
- **Not a raw-feature drift detector.** Alibi Detect and the KS/MMD family test whether the *inputs* moved. DriftReceipt never sees inputs; it tests whether the model's *behaviour* moved. Input drift with stable behaviour is their story, not this one's.
- **Not a performance estimator.** NannyML's CBPE estimates accuracy before labels arrive; DriftReceipt reports accuracy drift only from real labels, and says `skipped` when there are none.
- **Not new statistics.** JS divergence, ECE, and share deltas are standard instruments (Evidently uses JS for categorical drift too). The product is the contract — prediction-only input, one verdict, slice-level behaviour drift, a JSON-diffable artifact — not a claim that drift mathematics was invented here.
- **Not a guarantee.** Two windows and fixed thresholds: a slow drift under the caution band, or a failure mode no shared metadata column expresses, will not appear on the receipt. Slices are single predicates in 0.1.0.

## Measured demo (synthetic, reproducible)

`examples/drift_demo.py` simulates prediction logs for a 3-class model (1,200 rows per window) and compares two pairs. The script prints whatever it measures and exits non-zero if the outcomes differ:

- **Stable pair** (same distribution, different seed): **GO** — prediction JS 0.001, largest share shift +0.042, mean-confidence Δ −0.000, ECE Δ +0.019.
- **Drifted pair** — class `ring` grows ~33% → ~62% of traffic, and `scanner_b` confidence collapses ~0.87 → ~0.52 with predictions just as accurate: **NO-GO** on all four checks. Accuracy moved +0.009 (invisible) while ECE went 0.041 → 0.196; the top slice is `source == 'scanner_b'` (confidence 0.869 → 0.522, accuracy 0.894 → 0.908). It is a mechanics demo, not a benchmark.

Receipts (text + JSON) land in `examples/output/`.

## License

MIT. Issues and pull requests are welcome on GitHub.
