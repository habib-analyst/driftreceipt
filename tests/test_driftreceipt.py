"""Tests for driftreceipt.

Includes the three contract tests from the build brief — identical windows
give a GO with exactly zero drift, a planted class-share shift is flagged
with the right class, and a planted confidence collapse on one metadata
slice names that slice on the receipt — plus determinism, the skipped-check
semantics, driver rows, and input validation.
"""

import json

import numpy as np
import pytest

from driftreceipt import compare_windows


def probs_for_counts(counts, conf=0.9):
    """Deterministic probability matrix with the given predicted-class counts."""
    rows = []
    for cls, count in enumerate(counts):
        for _ in range(count):
            row = np.full(3, (1.0 - conf) / 2.0)
            row[cls] = conf
            rows.append(row)
    return np.array(rows)


def jittered_probs(rng, pred, conf_mean=0.87):
    n = len(pred)
    conf = np.clip(rng.normal(conf_mean, 0.03, size=n), 0.55, 0.98)
    probs = np.repeat(((1.0 - conf) / 2.0)[:, None], 3, axis=1)
    probs[np.arange(n), pred] = conf
    return probs


# ---------------------------------------------------------------------------
# Contract test 1: identical windows -> GO, exactly zero drift
# ---------------------------------------------------------------------------


def test_identical_windows_go_with_zero_drift():
    rng = np.random.default_rng(0)
    pred = rng.integers(0, 3, size=300)
    probs = jittered_probs(rng, pred)
    labels = pred.copy()
    meta = {"source": np.where(rng.random(300) < 0.4, "scanner_b", "scanner_a")}
    receipt = compare_windows(
        probs, probs.copy(), ref_meta=meta, cur_meta=meta,
        ref_labels=labels, cur_labels=labels.copy(),
    )
    assert receipt.verdict == "go"
    pred_check = receipt.checks["prediction_shift"]
    assert pred_check.details["js_divergence"] == 0.0
    assert pred_check.details["share_deltas"] == [0.0, 0.0, 0.0]
    assert receipt.checks["confidence"].details["confidence_delta"] == 0.0
    assert receipt.checks["performance"].details["accuracy_delta"] == 0.0
    assert receipt.checks["performance"].details["ece_delta"] == 0.0
    for check in receipt.checks.values():
        assert check.status == "pass"
    assert receipt.drivers["growing_class"]["indices"] == []


# ---------------------------------------------------------------------------
# Contract test 2: planted class-share shift -> flagged, right class named
# ---------------------------------------------------------------------------


def test_planted_class_share_shift_flagged_with_right_class():
    ref = probs_for_counts([150, 90, 60])
    cur = probs_for_counts([60, 90, 150])
    receipt = compare_windows(ref, cur, class_names=["a", "b", "c"])
    check = receipt.checks["prediction_shift"]
    assert check.status == "no-go"
    assert check.details["largest_shift_class"] == "c"
    assert check.details["largest_shift_class_index"] == 2
    assert check.details["share_deltas"][2] == pytest.approx(0.30)
    assert receipt.verdict == "no-go"
    # Drivers: current rows predicted as the growing class 'c'.
    drivers = receipt.drivers["growing_class"]
    assert drivers["class"] == "c"
    assert drivers["indices"], "expected driver rows for the growing class"
    cur_pred = np.argmax(cur, axis=1)
    assert all(cur_pred[i] == 2 for i in drivers["indices"])


# ---------------------------------------------------------------------------
# Contract test 3: planted confidence collapse on one slice -> slice named
# ---------------------------------------------------------------------------


def test_planted_slice_confidence_collapse_is_named():
    n = 500
    rng = np.random.default_rng(1)
    source = np.where(np.arange(n) % 5 < 2, "scanner_b", "scanner_a")
    pred = np.arange(n) % 3
    ref = jittered_probs(rng, pred, conf_mean=0.90)
    cur = jittered_probs(rng, pred, conf_mean=0.90)
    collapsed = source == "scanner_b"
    cur[collapsed] = jittered_probs(rng, pred[collapsed], conf_mean=0.55)
    meta = {"source": source}
    receipt = compare_windows(ref, cur, ref_meta=meta, cur_meta=meta)
    assert receipt.checks["slices"].status == "no-go"
    assert receipt.slices, "expected slice cards on the receipt"
    top = receipt.slices[0]
    assert top.definition == "source == 'scanner_b'"
    assert top.confidence_delta == pytest.approx(-0.35, abs=0.05)
    assert "scanner_b" in receipt.summary()
    # Worst-slice drivers are current rows inside that slice.
    worst = receipt.drivers["worst_slice"]
    assert worst["definition"] == "source == 'scanner_b'"
    assert all(source[i] == "scanner_b" for i in worst["indices"])
    assert receipt.verdict == "no-go"


# ---------------------------------------------------------------------------
# Behavioural coverage
# ---------------------------------------------------------------------------


def test_stable_windows_from_same_distribution_go():
    def window(seed):
        rng = np.random.default_rng(seed)
        true = rng.choice(3, size=2000, p=[0.34, 0.33, 0.33])
        pred = np.where(rng.random(2000) < 0.9, true, (true + 1) % 3)
        return jittered_probs(rng, pred), true

    ref_probs, ref_y = window(10)
    cur_probs, cur_y = window(11)
    receipt = compare_windows(ref_probs, cur_probs, ref_labels=ref_y, cur_labels=cur_y)
    assert receipt.verdict == "go"


def test_determinism_and_json_round_trip():
    rng = np.random.default_rng(2)
    ref = jittered_probs(rng, rng.integers(0, 3, size=200))
    cur = jittered_probs(rng, rng.integers(0, 3, size=200))
    meta = {"brightness": rng.uniform(0, 1, size=200)}
    a = compare_windows(ref, cur, ref_meta=meta, cur_meta=meta)
    b = compare_windows(ref, cur, ref_meta=meta, cur_meta=meta)
    assert a.to_dict() == b.to_dict()
    assert json.loads(a.to_json()) == a.to_dict()


def test_skipped_checks_are_reported_not_passed():
    ref = probs_for_counts([100, 100, 100])
    cur = probs_for_counts([100, 100, 100])
    receipt = compare_windows(ref, cur)
    assert receipt.checks["performance"].status == "skipped"
    assert receipt.checks["slices"].status == "skipped"
    assert receipt.verdict == "go"  # skipped checks do not vote


def test_accuracy_drop_flagged_when_labels_present():
    ref = probs_for_counts([100, 100, 100], conf=0.9)
    cur = probs_for_counts([100, 100, 100], conf=0.9)
    ref_labels = np.argmax(ref, axis=1)  # perfect
    cur_labels = np.argmax(cur, axis=1)
    cur_labels[:90] = (cur_labels[:90] + 1) % 3  # 30% now wrong
    receipt = compare_windows(ref, cur, ref_labels=ref_labels, cur_labels=cur_labels)
    perf = receipt.checks["performance"]
    assert perf.details["accuracy_delta"] == pytest.approx(-0.30)
    assert perf.status == "no-go"
    assert receipt.verdict == "no-go"


def test_slice_accuracy_delta_reported_with_labels():
    n = 400
    rng = np.random.default_rng(3)
    group = np.where(np.arange(n) % 2 == 0, "x", "y")
    pred = np.arange(n) % 3
    probs = jittered_probs(rng, pred, conf_mean=0.88)
    ref_labels = pred.copy()
    cur_labels = pred.copy()
    cur_labels[group == "x"] = (pred[group == "x"] + 1) % 3  # slice x fully wrong now
    meta = {"group": group}
    receipt = compare_windows(
        probs, probs.copy(), ref_meta=meta, cur_meta=meta,
        ref_labels=ref_labels, cur_labels=cur_labels,
    )
    by_def = {s.definition: s for s in receipt.slices}
    assert by_def["group == 'x'"].accuracy_delta == pytest.approx(-1.0)
    assert by_def["group == 'y'"].accuracy_delta == pytest.approx(0.0)


def test_numeric_metadata_produces_bands():
    rng = np.random.default_rng(4)
    ref = jittered_probs(rng, rng.integers(0, 3, size=400), conf_mean=0.9)
    cur = jittered_probs(rng, rng.integers(0, 3, size=400), conf_mean=0.9)
    bright = rng.uniform(0, 1, size=400)
    dim = bright < 0.3
    cur[dim] = jittered_probs(rng, np.argmax(cur[dim], axis=1), conf_mean=0.55)
    meta = {"brightness": bright}
    receipt = compare_windows(ref, cur, ref_meta=meta, cur_meta=meta)
    assert receipt.slices
    assert "<=" in receipt.slices[0].definition
    assert receipt.slices[0].confidence_delta < -0.1


def test_slice_support_gate():
    ref = probs_for_counts([100, 100, 100])
    cur = probs_for_counts([100, 100, 100])
    meta = {"tiny": np.array(["rare"] * 5 + ["common"] * 295)}
    receipt = compare_windows(ref, cur, ref_meta=meta, cur_meta=meta)
    # The 5-row 'rare' slice never qualifies; only 'common' is evaluated.
    assert receipt.checks["slices"].details["slices_evaluated"] == 1
    assert all("rare" not in s.definition for s in receipt.slices)
    # When no value is shared between windows, nothing qualifies -> skipped.
    disjoint_ref = {"tiny": np.array(["x"] * 300)}
    disjoint_cur = {"tiny": np.array(["y"] * 300)}
    receipt2 = compare_windows(ref, cur, ref_meta=disjoint_ref, cur_meta=disjoint_cur)
    assert receipt2.checks["slices"].status == "skipped"


def test_string_labels_with_class_names():
    ref = probs_for_counts([100, 100, 100])
    cur = probs_for_counts([100, 100, 100])
    names = ["a", "b", "c"]
    ref_labels = np.array([names[i] for i in np.argmax(ref, axis=1)])
    cur_labels = ref_labels.copy()
    receipt = compare_windows(ref, cur, ref_labels=ref_labels, cur_labels=cur_labels,
                              class_names=names)
    assert receipt.checks["performance"].status == "pass"
    assert receipt.checks["performance"].details["ref_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_rows_not_summing_to_one():
    bad = np.array([[0.5, 0.5, 0.5]] * 10)
    good = probs_for_counts([4, 3, 3])
    with pytest.raises(ValueError, match="sum to 1"):
        compare_windows(bad, good)


def test_rejects_nan_and_negative():
    good = probs_for_counts([4, 3, 3])
    bad = good.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        compare_windows(bad, good)
    bad2 = good.copy()
    bad2[0] = [1.1, -0.1, 0.0]
    with pytest.raises(ValueError, match="negative"):
        compare_windows(bad2, good)


def test_rejects_class_count_mismatch():
    ref = probs_for_counts([4, 3, 3])
    cur = np.tile([0.5, 0.5], (10, 1))
    with pytest.raises(ValueError, match="same number of classes"):
        compare_windows(ref, cur)


def test_rejects_one_sided_labels_and_meta():
    ref = probs_for_counts([4, 3, 3])
    cur = probs_for_counts([4, 3, 3])
    with pytest.raises(ValueError, match="together"):
        compare_windows(ref, cur, ref_labels=np.zeros(10, dtype=int))
    with pytest.raises(ValueError, match="together"):
        compare_windows(ref, cur, ref_meta={"s": np.arange(10)})


def test_rejects_out_of_range_labels():
    ref = probs_for_counts([4, 3, 3])
    cur = probs_for_counts([4, 3, 3])
    with pytest.raises(ValueError, match=r"\[0, 3\)"):
        compare_windows(ref, cur, ref_labels=np.zeros(10, dtype=int),
                        cur_labels=np.full(10, 3))


def test_rejects_bad_thresholds_and_names():
    ref = probs_for_counts([4, 3, 3])
    cur = probs_for_counts([4, 3, 3])
    with pytest.raises(ValueError, match="caution < no_go"):
        compare_windows(ref, cur, js_caution=0.2, js_no_go=0.1)
    with pytest.raises(ValueError, match="class_names"):
        compare_windows(ref, cur, class_names=["only", "two"])
    with pytest.raises(ValueError, match="length"):
        compare_windows(ref, cur, ref_meta={"s": np.arange(5)},
                        cur_meta={"s": np.arange(5)})
