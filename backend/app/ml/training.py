from __future__ import annotations

import numpy as np


def check_training_ready(
    y: np.ndarray,
    min_total: int,
    min_per_class: int,
    active_labels: list[int] | None = None,
) -> tuple[bool, str]:
    if len(y) < min_total:
        return False, f"Need at least {min_total} labeled images (have {len(y)})"
    labels, counts = np.unique(y, return_counts=True)
    if active_labels is not None:
        for lab in active_labels:
            if lab not in labels:
                return False, f"Label {lab} has no examples yet"
            idx = np.where(labels == lab)[0]
            if counts[idx[0]] < min_per_class:
                return False, f"Label {lab} needs at least {min_per_class} examples"
    else:
        if np.min(counts) < min_per_class:
            return False, f"Each class needs at least {min_per_class} examples"
    return True, "ok"
