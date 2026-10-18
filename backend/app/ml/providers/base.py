from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ProviderRequirements:
    min_labeled_total: int = 30
    min_labeled_per_class: int = 3


@dataclass
class TrainConfig:
    provider_config: dict[str, Any] = field(default_factory=dict)
    min_labeled_total: int = 30
    min_labeled_per_class: int = 3
    random_state: int = 42


@dataclass
class TrainReport:
    success: bool
    message: str
    accuracy: float | None = None
    duration_ms: float = 0.0


class SuggestionProvider(ABC):
    id: str
    display_name: str

    @abstractmethod
    def config_schema(self) -> dict[str, Any]: ...

    @abstractmethod
    def fit(
        self, X: np.ndarray, y: np.ndarray, *, config: TrainConfig
    ) -> TrainReport: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray | None: ...

    @property
    @abstractmethod
    def is_ready(self) -> bool: ...


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


def holdout_accuracy(
    clf, X: np.ndarray, y: np.ndarray, random_state: int = 42
) -> float | None:
    if len(y) < 10:
        return None
    from sklearn.model_selection import train_test_split

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
    except ValueError:
        return None
    from sklearn.base import clone

    model = clone(clf) if hasattr(clf, "get_params") else clf
    model.fit(X_tr, y_tr)
    return float(model.score(X_te, y_te))
