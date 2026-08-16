"""Centralized target assignment."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def assign_targets(starts: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    if starts.shape != targets.shape or starts.ndim != 2 or starts.shape[1] != 3:
        raise ValueError("starts and targets must both have shape (N, 3)")
    cost = cdist(starts, targets)
    rows, columns = linear_sum_assignment(cost)
    if not np.array_equal(rows, np.arange(len(starts))):
        raise RuntimeError("unexpected non-canonical assignment row order")
    assigned = targets[columns]
    return assigned, float(cost[rows, columns].sum()), columns
