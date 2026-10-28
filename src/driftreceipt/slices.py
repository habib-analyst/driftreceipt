"""Slice-level behaviour drift over shared metadata columns.

A *slice* is a single readable predicate over one metadata column, in the
style of the SliceSniff library: categorical equality
(``source == 'scanner_b'``), numeric quantile bands whose edges are derived
from the **reference** window (so a definition means the same rows in both
windows), and explicit missing-value predicates
(``brightness is missing``).

For every slice that has enough rows in *both* windows we measure how the
model's behaviour inside the slice changed: the Jensen-Shannon divergence
of the predicted-class distribution, the mean-confidence delta, and — when
labels exist for both windows — the accuracy delta. Depth is deliberately
one predicate in 0.1.0: a slice must keep meaningful support in both
windows, and single predicates are the slices a human can act on first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats import accuracy, class_shares, confidence, js_divergence, predicted_classes

__all__ = ["SliceDrift", "column_predicates", "mine_slice_drift"]


@dataclass
class SliceDrift:
    """Behaviour change inside one metadata slice, reference vs current."""

    definition: str
    column: str
    ref_support: int
    cur_support: int
    ref_mean_confidence: float
    cur_mean_confidence: float
    confidence_delta: float
    prediction_js: float
    ref_accuracy: float | None
    cur_accuracy: float | None
    accuracy_delta: float | None
    score: float

    def to_dict(self) -> dict:
        return {
            "definition": self.definition,
            "column": self.column,
            "ref_support": self.ref_support,
            "cur_support": self.cur_support,
            "ref_mean_confidence": self.ref_mean_confidence,
            "cur_mean_confidence": self.cur_mean_confidence,
            "confidence_delta": self.confidence_delta,
            "prediction_js": self.prediction_js,
            "ref_accuracy": self.ref_accuracy,
            "cur_accuracy": self.cur_accuracy,
            "accuracy_delta": self.accuracy_delta,
            "score": self.score,
        }


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _missing_mask(column: np.ndarray) -> np.ndarray:
    if column.dtype.kind == "f":
        return np.isnan(column)
    return np.array([_is_missing(v) for v in column], dtype=bool)


def _format_number(value: float) -> str:
    return f"{value:.4g}"


def _format_value(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _format_number(float(value))
    return f"'{value}'"


def _is_numeric(column: np.ndarray) -> bool:
    return column.dtype.kind in "fiu"


def column_predicates(
    name: str, ref_column: np.ndarray, cur_column: np.ndarray
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Readable single-predicate slices for one shared metadata column.

    Returns (definition, ref_mask, cur_mask) triples. Numeric columns with
    many distinct values become quantile bands derived from the reference
    window; everything else becomes equality predicates over the values
    present in both windows. A missing-value predicate is added when either
    window has missing entries.
    """
    predicates: list[tuple[str, np.ndarray, np.ndarray]] = []
    ref_missing = _missing_mask(ref_column)
    cur_missing = _missing_mask(cur_column)
    if ref_missing.any() or cur_missing.any():
        predicates.append((f"{name} is missing", ref_missing, cur_missing))

    if _is_numeric(ref_column) and _is_numeric(cur_column):
        ref_vals = ref_column[~ref_missing].astype(float)
        unique = np.unique(ref_vals)
        if len(unique) > 10:
            edges = np.unique(np.quantile(ref_vals, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
            for i in range(len(edges) - 1):
                lo, hi = float(edges[i]), float(edges[i + 1])
                if i == len(edges) - 2:
                    ref_mask = (ref_column >= lo) & (ref_column <= hi)
                    cur_mask = (cur_column >= lo) & (cur_column <= hi)
                    definition = f"{_format_number(lo)} <= {name} <= {_format_number(hi)}"
                else:
                    ref_mask = (ref_column >= lo) & (ref_column < hi)
                    cur_mask = (cur_column >= lo) & (cur_column < hi)
                    definition = f"{_format_number(lo)} <= {name} < {_format_number(hi)}"
                predicates.append((definition, ref_mask, cur_mask))
            return predicates

    # Categorical path: equality over values present in both windows.
    ref_present = {_format_value(v): v for v in ref_column[~ref_missing]}
    cur_values = {_format_value(v) for v in cur_column[~cur_missing]}
    for key in sorted(ref_present):
        if key not in cur_values:
            continue
        value = ref_present[key]
        ref_mask = np.array([False if m else v == value for v, m in zip(ref_column, ref_missing)])
        cur_mask = np.array([False if m else v == value for v, m in zip(cur_column, cur_missing)])
        predicates.append((f"{name} == {_format_value(value)}", ref_mask, cur_mask))
    return predicates


def mine_slice_drift(
    ref_probs: np.ndarray,
    cur_probs: np.ndarray,
    ref_meta: dict[str, np.ndarray],
    cur_meta: dict[str, np.ndarray],
    shared_columns: list[str],
    n_classes: int,
    ref_labels: np.ndarray | None = None,
    cur_labels: np.ndarray | None = None,
    min_support: int = 30,
) -> list[SliceDrift]:
    """All qualifying slices, ranked by behaviour change (worst first)."""
    ref_pred = predicted_classes(ref_probs)
    cur_pred = predicted_classes(cur_probs)
    ref_conf = confidence(ref_probs)
    cur_conf = confidence(cur_probs)
    have_labels = ref_labels is not None and cur_labels is not None

    slices: list[SliceDrift] = []
    for column in shared_columns:
        for definition, ref_mask, cur_mask in column_predicates(
            column, ref_meta[column], cur_meta[column]
        ):
            ref_support = int(ref_mask.sum())
            cur_support = int(cur_mask.sum())
            if ref_support < min_support or cur_support < min_support:
                continue
            ref_shares = class_shares(ref_pred[ref_mask], n_classes)
            cur_shares = class_shares(cur_pred[cur_mask], n_classes)
            pred_js = js_divergence(ref_shares, cur_shares)
            ref_mean_conf = float(ref_conf[ref_mask].mean())
            cur_mean_conf = float(cur_conf[cur_mask].mean())
            conf_delta = cur_mean_conf - ref_mean_conf
            ref_acc = cur_acc = acc_delta = None
            if have_labels:
                ref_acc = accuracy(ref_pred[ref_mask], ref_labels[ref_mask])
                cur_acc = accuracy(cur_pred[cur_mask], cur_labels[cur_mask])
                acc_delta = cur_acc - ref_acc
            score = pred_js + abs(conf_delta) + (abs(acc_delta) if acc_delta is not None else 0.0)
            slices.append(
                SliceDrift(
                    definition=definition,
                    column=column,
                    ref_support=ref_support,
                    cur_support=cur_support,
                    ref_mean_confidence=ref_mean_conf,
                    cur_mean_confidence=cur_mean_conf,
                    confidence_delta=conf_delta,
                    prediction_js=pred_js,
                    ref_accuracy=ref_acc,
                    cur_accuracy=cur_acc,
                    accuracy_delta=acc_delta,
                    score=score,
                )
            )
    slices.sort(key=lambda s: (-s.score, s.definition))
    return slices
