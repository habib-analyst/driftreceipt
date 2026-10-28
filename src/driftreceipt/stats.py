"""Small distribution statistics used by the drift checks.

Everything here is a deterministic function of plain NumPy arrays: class
shares, Jensen-Shannon divergence, confidence summaries, accuracy, and
expected calibration error. No randomness, no fitted state.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "predicted_classes",
    "confidence",
    "class_shares",
    "js_divergence",
    "confidence_histogram",
    "accuracy",
    "expected_calibration_error",
]


def predicted_classes(probs: np.ndarray) -> np.ndarray:
    """Argmax class per row of a probability matrix."""
    return np.argmax(probs, axis=1)


def confidence(probs: np.ndarray) -> np.ndarray:
    """Top-class probability per row."""
    return np.max(probs, axis=1)


def class_shares(pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Fraction of rows predicted as each class (sums to 1 for non-empty input)."""
    counts = np.bincount(pred, minlength=n_classes).astype(float)
    total = counts.sum()
    if total == 0:
        return counts
    return counts / total


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base 2) between two probability vectors.

    Bounded in [0, 1]: 0 for identical distributions, 1 for disjoint ones.
    Zero entries are handled exactly (0 * log 0 = 0); no smoothing is added,
    so identical inputs give exactly 0.0.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def confidence_histogram(conf: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Normalised histogram of confidences over fixed bins on [0, 1]."""
    counts, _ = np.histogram(conf, bins=n_bins, range=(0.0, 1.0))
    total = counts.sum()
    if total == 0:
        return counts.astype(float)
    return counts.astype(float) / total


def accuracy(pred: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of rows where the predicted class equals the label."""
    return float(np.mean(pred == labels))


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """ECE over equal-width confidence bins: sum_b (n_b / N) * |acc_b - conf_b|."""
    conf = confidence(probs)
    pred = predicted_classes(probs)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    n = len(labels)
    for b in range(n_bins):
        in_bin = bin_idx == b
        count = int(in_bin.sum())
        if count:
            ece += (count / n) * abs(float(correct[in_bin].mean()) - float(conf[in_bin].mean()))
    _ = edges  # edges documented by construction; bin index math above is authoritative
    return float(ece)
