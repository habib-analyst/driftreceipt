"""DriftReceipt: one-call drift audit of a deployed classifier's behaviour.

``compare_windows`` takes two windows of model predictions — a reference
window (e.g. last week's logged probabilities) and a current window — plus
optional per-window metadata and labels, and returns a :class:`DriftReceipt`
with per-check verdicts and one overall verdict: ``go`` / ``caution`` /
``no-go``. Inputs are plain NumPy arrays: no raw features, no model object,
no DataFrames, no server.

Checks:
  * ``prediction_shift`` — predicted-class shares and Jensen-Shannon
    divergence between the two windows' prediction distributions.
  * ``confidence`` — mean-confidence delta and confidence-histogram drift.
  * ``performance`` — accuracy and ECE deltas (only when labels are given
    for both windows; otherwise reported ``skipped``, never a silent pass).
  * ``slices`` — worst behaviour change inside readable metadata slices
    (only when metadata is given for both windows).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .slices import SliceDrift, mine_slice_drift
from .stats import (
    accuracy,
    class_shares,
    confidence,
    confidence_histogram,
    expected_calibration_error,
    js_divergence,
    predicted_classes,
)

__all__ = ["CheckResult", "DriftReceipt", "compare_windows"]

_LIBRARY_VERSION = "0.1.0"

_STATUS_ORDER = {"pass": 0, "caution": 1, "no-go": 2}
_VERDICT_BY_STATUS = {0: "go", 1: "caution", 2: "no-go"}


@dataclass
class CheckResult:
    """One named check on the receipt."""

    name: str
    status: str  # "pass" | "caution" | "no-go" | "skipped"
    summary: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "summary": self.summary,
            "details": self.details,
        }


@dataclass
class DriftReceipt:
    """The full receipt returned by :func:`compare_windows`."""

    verdict: str  # "go" | "caution" | "no-go"
    ref_size: int
    cur_size: int
    n_classes: int
    class_names: list[str]
    checks: dict[str, CheckResult]
    slices: list[SliceDrift]
    drivers: dict
    config: dict

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "ref_size": self.ref_size,
            "cur_size": self.cur_size,
            "n_classes": self.n_classes,
            "class_names": list(self.class_names),
            "checks": {name: check.to_dict() for name, check in self.checks.items()},
            "slices": [s.to_dict() for s in self.slices],
            "drivers": self.drivers,
            "config": self.config,
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    def summary(self) -> str:
        lines = [
            "=" * 64,
            f"DRIFT RECEIPT — reference (n={self.ref_size}) vs current (n={self.cur_size})",
            "=" * 64,
            f"VERDICT: {self.verdict.upper()}",
            "",
        ]
        for check in self.checks.values():
            lines.append(f"{check.name:<18} {check.status.upper():<8} {check.summary}")
        if self.slices:
            lines.append("")
            lines.append("Top drifting slices:")
            for i, s in enumerate(self.slices, 1):
                acc_part = ""
                if s.accuracy_delta is not None:
                    acc_part = f", accuracy {s.ref_accuracy:.3f} -> {s.cur_accuracy:.3f}"
                lines.append(
                    f"  {i}. {s.definition}  (ref n={s.ref_support}, cur n={s.cur_support}; "
                    f"confidence {s.ref_mean_confidence:.3f} -> {s.cur_mean_confidence:.3f}, "
                    f"slice JS {s.prediction_js:.3f}{acc_part})"
                )
        grow = self.drivers.get("growing_class")
        if grow and grow["indices"]:
            lines.append("")
            lines.append(
                f"Drivers — fastest-growing class '{grow['class']}' "
                f"(share delta {grow['share_delta']:+.3f}), current rows: {grow['indices']}"
            )
        worst = self.drivers.get("worst_slice")
        if worst and worst["indices"]:
            lines.append(
                f"Drivers — worst slice '{worst['definition']}', "
                f"least-confident current rows: {worst['indices']}"
            )
        lines.append("=" * 64)
        return "\n".join(lines)

    __str__ = summary


def _band(value: float, caution: float, no_go: float) -> str:
    if value >= no_go:
        return "no-go"
    if value >= caution:
        return "caution"
    return "pass"


def _worse(a: str, b: str) -> str:
    return a if _STATUS_ORDER[a] >= _STATUS_ORDER[b] else b


def _validate_probs(name: str, probs) -> np.ndarray:
    arr = np.asarray(probs, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of class probabilities, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    if arr.shape[1] < 2:
        raise ValueError(f"{name} must have at least 2 class columns, got {arr.shape[1]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    if np.any(arr < 0):
        raise ValueError(f"{name} contains negative probabilities")
    row_sums = arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError(
            f"{name} rows must sum to 1 (max deviation {float(np.max(np.abs(row_sums - 1.0))):.2e})"
        )
    return arr


def _validate_labels(
    name: str, labels, n_rows: int, n_classes: int, class_names: list[str] | None
) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim != 1 or len(arr) != n_rows:
        raise ValueError(f"{name} must be a 1D array of length {n_rows}, got shape {arr.shape}")
    if class_names is not None and arr.dtype.kind in "US":
        mapping = {c: i for i, c in enumerate(class_names)}
        unknown = [v for v in arr if v not in mapping]
        if unknown:
            raise ValueError(f"{name} contains labels not in class_names: {unknown[:3]}")
        return np.array([mapping[v] for v in arr], dtype=int)
    if arr.dtype.kind == "f":
        if not np.all(np.isfinite(arr)) or not np.all(arr == np.floor(arr)):
            raise ValueError(f"{name} must contain integer class indices")
        arr = arr.astype(int)
    if arr.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer class indices or class names")
    if arr.min() < 0 or arr.max() >= n_classes:
        raise ValueError(f"{name} values must be in [0, {n_classes})")
    return arr.astype(int)


def _validate_meta(name: str, meta, n_rows: int) -> dict[str, np.ndarray]:
    if not isinstance(meta, dict):
        raise ValueError(f"{name} must be a dict of column name -> 1D array")
    out: dict[str, np.ndarray] = {}
    for column, values in meta.items():
        arr = np.asarray(values)
        if arr.ndim != 1 or len(arr) != n_rows:
            raise ValueError(
                f"{name}[{column!r}] must be a 1D array of length {n_rows}, got shape {arr.shape}"
            )
        out[str(column)] = arr
    return out


def compare_windows(
    ref_probs,
    cur_probs,
    ref_meta: dict | None = None,
    cur_meta: dict | None = None,
    ref_labels=None,
    cur_labels=None,
    class_names: list[str] | None = None,
    *,
    js_caution: float = 0.03,
    js_no_go: float = 0.10,
    share_caution: float = 0.10,
    share_no_go: float = 0.20,
    conf_caution: float = 0.05,
    conf_no_go: float = 0.10,
    acc_caution: float = 0.05,
    acc_no_go: float = 0.10,
    ece_caution: float = 0.05,
    ece_no_go: float = 0.10,
    slice_js_caution: float = 0.15,
    slice_js_no_go: float = 0.30,
    slice_conf_caution: float = 0.15,
    slice_conf_no_go: float = 0.25,
    slice_acc_caution: float = 0.15,
    slice_acc_no_go: float = 0.25,
    min_slice_support: int = 30,
    top_slices: int = 5,
    top_drivers: int = 20,
) -> DriftReceipt:
    """Compare a reference prediction window against a current one.

    Parameters are two ``(n, n_classes)`` probability matrices, optional
    metadata dicts (column name -> per-row values) and optional true labels
    per window. All thresholds are configurable; each caution threshold must
    be below its no-go threshold. Returns a :class:`DriftReceipt`.
    """
    for label, caution, no_go in [
        ("js", js_caution, js_no_go),
        ("share", share_caution, share_no_go),
        ("conf", conf_caution, conf_no_go),
        ("acc", acc_caution, acc_no_go),
        ("ece", ece_caution, ece_no_go),
        ("slice_js", slice_js_caution, slice_js_no_go),
        ("slice_conf", slice_conf_caution, slice_conf_no_go),
        ("slice_acc", slice_acc_caution, slice_acc_no_go),
    ]:
        if not (0 <= caution < no_go):
            raise ValueError(f"{label} thresholds must satisfy 0 <= caution < no_go")
    if min_slice_support < 1:
        raise ValueError("min_slice_support must be >= 1")
    if top_slices < 0 or top_drivers < 0:
        raise ValueError("top_slices and top_drivers must be >= 0")

    ref = _validate_probs("ref_probs", ref_probs)
    cur = _validate_probs("cur_probs", cur_probs)
    if ref.shape[1] != cur.shape[1]:
        raise ValueError(
            f"ref_probs and cur_probs must have the same number of classes "
            f"({ref.shape[1]} != {cur.shape[1]})"
        )
    n_classes = ref.shape[1]
    if class_names is None:
        names = [f"class_{i}" for i in range(n_classes)]
    else:
        names = [str(c) for c in class_names]
        if len(names) != n_classes:
            raise ValueError(f"class_names must have {n_classes} entries, got {len(names)}")
        if len(set(names)) != len(names):
            raise ValueError("class_names must be unique")

    if (ref_labels is None) != (cur_labels is None):
        raise ValueError("ref_labels and cur_labels must be provided together or not at all")
    ref_y = cur_y = None
    if ref_labels is not None:
        ref_y = _validate_labels("ref_labels", ref_labels, len(ref), n_classes, class_names)
        cur_y = _validate_labels("cur_labels", cur_labels, len(cur), n_classes, class_names)

    if (ref_meta is None) != (cur_meta is None):
        raise ValueError("ref_meta and cur_meta must be provided together or not at all")
    ref_m = cur_m = None
    shared_columns: list[str] = []
    ignored_columns: list[str] = []
    if ref_meta is not None:
        ref_m = _validate_meta("ref_meta", ref_meta, len(ref))
        cur_m = _validate_meta("cur_meta", cur_meta, len(cur))
        shared_columns = sorted(set(ref_m) & set(cur_m))
        ignored_columns = sorted((set(ref_m) | set(cur_m)) - set(shared_columns))

    ref_pred = predicted_classes(ref)
    cur_pred = predicted_classes(cur)
    ref_conf = confidence(ref)
    cur_conf = confidence(cur)

    checks: dict[str, CheckResult] = {}

    # --- prediction shift -------------------------------------------------
    ref_shares = class_shares(ref_pred, n_classes)
    cur_shares = class_shares(cur_pred, n_classes)
    share_deltas = cur_shares - ref_shares
    pred_js = js_divergence(ref_shares, cur_shares)
    soft_js = js_divergence(ref.mean(axis=0), cur.mean(axis=0))
    # Largest shift by magnitude; ties break toward the growing class (the
    # actionable side of the shift, and the class the driver rows describe),
    # then toward the lower class index. Fully deterministic.
    top_idx = max(
        range(n_classes),
        key=lambda i: (abs(float(share_deltas[i])), float(share_deltas[i]), -i),
    )
    status = _worse(
        _band(pred_js, js_caution, js_no_go),
        _band(abs(float(share_deltas[top_idx])), share_caution, share_no_go),
    )
    checks["prediction_shift"] = CheckResult(
        name="prediction_shift",
        status=status,
        summary=(
            f"predicted-class JS divergence {pred_js:.3f}; largest share shift "
            f"{share_deltas[top_idx]:+.3f} ('{names[top_idx]}')"
        ),
        details={
            "js_divergence": pred_js,
            "soft_js_divergence": soft_js,
            "ref_shares": ref_shares.tolist(),
            "cur_shares": cur_shares.tolist(),
            "share_deltas": share_deltas.tolist(),
            "largest_shift_class": names[top_idx],
            "largest_shift_class_index": top_idx,
            "largest_shift_delta": float(share_deltas[top_idx]),
        },
    )

    # --- confidence -------------------------------------------------------
    ref_mean_conf = float(ref_conf.mean())
    cur_mean_conf = float(cur_conf.mean())
    conf_delta = cur_mean_conf - ref_mean_conf
    conf_js = js_divergence(
        confidence_histogram(ref_conf), confidence_histogram(cur_conf)
    )
    status = _worse(
        _band(abs(conf_delta), conf_caution, conf_no_go),
        _band(conf_js, js_caution, js_no_go),
    )
    checks["confidence"] = CheckResult(
        name="confidence",
        status=status,
        summary=(
            f"mean confidence {ref_mean_conf:.3f} -> {cur_mean_conf:.3f} "
            f"(delta {conf_delta:+.3f}); confidence-histogram JS {conf_js:.3f}"
        ),
        details={
            "ref_mean_confidence": ref_mean_conf,
            "cur_mean_confidence": cur_mean_conf,
            "confidence_delta": conf_delta,
            "confidence_histogram_js": conf_js,
            "ref_mean_entropy": float(-(np.where(ref > 0, ref * np.log(ref), 0.0)).sum(axis=1).mean()),
            "cur_mean_entropy": float(-(np.where(cur > 0, cur * np.log(cur), 0.0)).sum(axis=1).mean()),
        },
    )

    # --- performance (labels only) ----------------------------------------
    if ref_y is not None:
        ref_acc = accuracy(ref_pred, ref_y)
        cur_acc = accuracy(cur_pred, cur_y)
        acc_delta = cur_acc - ref_acc
        ref_ece = expected_calibration_error(ref, ref_y)
        cur_ece = expected_calibration_error(cur, cur_y)
        ece_delta = cur_ece - ref_ece
        status = _worse(
            _band(max(0.0, -acc_delta), acc_caution, acc_no_go),
            _band(max(0.0, ece_delta), ece_caution, ece_no_go),
        )
        checks["performance"] = CheckResult(
            name="performance",
            status=status,
            summary=(
                f"accuracy {ref_acc:.3f} -> {cur_acc:.3f} (delta {acc_delta:+.3f}); "
                f"ECE {ref_ece:.3f} -> {cur_ece:.3f} (delta {ece_delta:+.3f})"
            ),
            details={
                "ref_accuracy": ref_acc,
                "cur_accuracy": cur_acc,
                "accuracy_delta": acc_delta,
                "ref_ece": ref_ece,
                "cur_ece": cur_ece,
                "ece_delta": ece_delta,
            },
        )
    else:
        checks["performance"] = CheckResult(
            name="performance",
            status="skipped",
            summary="no labels provided for either window — accuracy/ECE drift not measured",
            details={"reason": "labels not provided"},
        )

    # --- slices ------------------------------------------------------------
    all_slices: list[SliceDrift] = []
    if ref_m is not None:
        all_slices = mine_slice_drift(
            ref,
            cur,
            ref_m,
            cur_m,
            shared_columns,
            n_classes,
            ref_labels=ref_y,
            cur_labels=cur_y,
            min_support=min_slice_support,
        )
        if all_slices:
            worst = all_slices[0]
            status = "pass"
            for s in all_slices:
                status = _worse(status, _band(s.prediction_js, slice_js_caution, slice_js_no_go))
                status = _worse(
                    status, _band(abs(s.confidence_delta), slice_conf_caution, slice_conf_no_go)
                )
                if s.accuracy_delta is not None:
                    status = _worse(
                        status,
                        _band(abs(s.accuracy_delta), slice_acc_caution, slice_acc_no_go),
                    )
            checks["slices"] = CheckResult(
                name="slices",
                status=status,
                summary=(
                    f"worst slice: {worst.definition} (confidence delta "
                    f"{worst.confidence_delta:+.3f}, slice JS {worst.prediction_js:.3f})"
                ),
                details={
                    "slices_evaluated": len(all_slices),
                    "shared_columns": shared_columns,
                    "ignored_columns": ignored_columns,
                    "worst_slice": worst.definition,
                },
            )
        else:
            checks["slices"] = CheckResult(
                name="slices",
                status="skipped",
                summary=(
                    f"no metadata slice had >= {min_slice_support} rows in both windows"
                ),
                details={
                    "reason": "no slice met min_slice_support in both windows",
                    "shared_columns": shared_columns,
                    "ignored_columns": ignored_columns,
                },
            )
    else:
        checks["slices"] = CheckResult(
            name="slices",
            status="skipped",
            summary="no metadata provided for either window — slice drift not measured",
            details={"reason": "metadata not provided"},
        )

    # --- drivers ------------------------------------------------------------
    drivers: dict = {}
    grow_idx = int(np.argmax(share_deltas))
    if share_deltas[grow_idx] > 0 and top_drivers > 0:
        rows = np.where(cur_pred == grow_idx)[0]
        ranked = sorted(rows.tolist(), key=lambda i: (-float(cur_conf[i]), i))
        drivers["growing_class"] = {
            "class": names[grow_idx],
            "class_index": grow_idx,
            "share_delta": float(share_deltas[grow_idx]),
            "indices": ranked[:top_drivers],
        }
    else:
        drivers["growing_class"] = {
            "class": names[grow_idx],
            "class_index": grow_idx,
            "share_delta": float(share_deltas[grow_idx]),
            "indices": [],
        }
    if all_slices and top_drivers > 0:
        worst = all_slices[0]
        # Recompute the worst slice's current-window mask for driver rows.
        from .slices import column_predicates

        cur_mask = None
        for definition, _ref_mask, mask in column_predicates(
            worst.column, ref_m[worst.column], cur_m[worst.column]
        ):
            if definition == worst.definition:
                cur_mask = mask
                break
        rows = np.where(cur_mask)[0] if cur_mask is not None else np.array([], dtype=int)
        ranked = sorted(rows.tolist(), key=lambda i: (float(cur_conf[i]), i))
        drivers["worst_slice"] = {
            "definition": worst.definition,
            "indices": ranked[:top_drivers],
        }

    verdict_status = 0
    for check in checks.values():
        if check.status != "skipped":
            verdict_status = max(verdict_status, _STATUS_ORDER[check.status])
    verdict = _VERDICT_BY_STATUS[verdict_status]

    config = {
        "library_version": _LIBRARY_VERSION,
        "thresholds": {
            "js_caution": js_caution,
            "js_no_go": js_no_go,
            "share_caution": share_caution,
            "share_no_go": share_no_go,
            "conf_caution": conf_caution,
            "conf_no_go": conf_no_go,
            "acc_caution": acc_caution,
            "acc_no_go": acc_no_go,
            "ece_caution": ece_caution,
            "ece_no_go": ece_no_go,
            "slice_js_caution": slice_js_caution,
            "slice_js_no_go": slice_js_no_go,
            "slice_conf_caution": slice_conf_caution,
            "slice_conf_no_go": slice_conf_no_go,
            "slice_acc_caution": slice_acc_caution,
            "slice_acc_no_go": slice_acc_no_go,
        },
        "min_slice_support": min_slice_support,
        "top_slices": top_slices,
        "top_drivers": top_drivers,
        "shared_columns": shared_columns,
        "ignored_columns": ignored_columns,
        "labels_provided": ref_y is not None,
        "metadata_provided": ref_m is not None,
    }

    return DriftReceipt(
        verdict=verdict,
        ref_size=len(ref),
        cur_size=len(cur),
        n_classes=n_classes,
        class_names=names,
        checks=checks,
        slices=all_slices[:top_slices],
        drivers=drivers,
        config=config,
    )
