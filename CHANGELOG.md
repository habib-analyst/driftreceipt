# Changelog

## 0.1.0 — 2026-10-28

First release.

- `compare_windows(ref_probs, cur_probs, ...)` — one call comparing a
  reference window of logged classifier predictions against a current
  window, returning a `DriftReceipt` with one verdict:
  `go` / `caution` / `no-go`.
- Four checks: `prediction_shift` (per-class share deltas + Jensen-Shannon
  divergence), `confidence` (mean-confidence delta + confidence-histogram
  drift), `performance` (accuracy / ECE deltas, labels only), and `slices`
  (worst behaviour change inside readable metadata slices).
- Slice drift over shared metadata columns: categorical equality and
  reference-derived numeric quantile bands, SliceSniff-style definitions,
  with exact support counts in both windows.
- Driver rows: the current-window rows behind the fastest-growing class
  and inside the worst slice, by index.
- JSON-serializable receipt (`to_dict()` / `to_json()`) with the full
  threshold configuration recorded, for CI gates and week-over-week diffs.
- NumPy is the only runtime dependency. Inputs are plain arrays — no raw
  features, no model object, no DataFrames, no server.
