# Contributing

Issues and pull requests are welcome at
https://github.com/habib-analyst/driftreceipt/issues.

Ground rules that keep the library what it is:

- **NumPy stays the only runtime dependency.** No pandas, no plotting, no
  framework imports — the receipt must be computable anywhere predictions
  are logged.
- **Deterministic, always.** No randomness, no wall-clock dependence, no
  dict-order dependence in any output. Two runs on the same windows must
  produce byte-identical receipts (there is a test for this).
- **Skipped means skipped.** A check that cannot run (no labels, no
  metadata, no slice with enough support) reports `skipped` with a reason.
  It never silently passes.
- **Thresholds are configuration, recorded on the receipt.** New checks
  must expose their caution / no-go thresholds as parameters and record
  them in `receipt.config`.
- Run `python -m pytest -q` and `python examples/drift_demo.py` before
  opening a pull request; the demo asserts its planted outcomes and exits
  non-zero on a mismatch.
