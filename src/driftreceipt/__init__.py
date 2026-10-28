"""DriftReceipt — a one-call drift receipt for a deployed classifier.

Hand :func:`compare_windows` two windows of logged predictions (reference
vs current) as plain probability arrays, plus optional metadata and labels,
and get back one verdict — GO / CAUTION / NO-GO — with the checks, the
worst drifting metadata slices, and the current rows driving the shift.
NumPy is the only dependency.
"""

from .core import _LIBRARY_VERSION, CheckResult, DriftReceipt, compare_windows
from .slices import SliceDrift

__version__ = _LIBRARY_VERSION

__all__ = [
    "CheckResult",
    "DriftReceipt",
    "SliceDrift",
    "compare_windows",
    "__version__",
]
