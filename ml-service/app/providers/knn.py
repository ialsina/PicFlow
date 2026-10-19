from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from app.providers.base import (
    SuggestionProvider,
    TrainConfig,
    TrainReport,
    holdout_accuracy,
)


class KNNProvider(SuggestionProvider):
    id = "knn"
    display_name = "k-Nearest Neighbors"

    def __init__(self) -> None:
        self._model: KNeighborsClassifier | None = None

    def config_schema(self) -> dict[str, Any]:
        return {"n_neighbors": {"type": "integer", "default": 5, "minimum": 1}}

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def fit(self, X: np.ndarray, y: np.ndarray, *, config: TrainConfig) -> TrainReport:
        t0 = time.perf_counter()
        n_neighbors = int(config.provider_config.get("n_neighbors", 5))
        n_neighbors = min(n_neighbors, max(1, len(y) - 1))
        self._model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self._model.fit(X, y)
        acc = holdout_accuracy(
            KNeighborsClassifier(n_neighbors=n_neighbors), X, y, config.random_state
        )
        ms = (time.perf_counter() - t0) * 1000
        return TrainReport(
            success=True, message="kNN fitted", accuracy=acc, duration_ms=ms
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self._model is None:
            return None
        return self._model.predict_proba(X)
