from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.neural_network import MLPClassifier

from app.ml.providers.base import (
    SuggestionProvider,
    TrainConfig,
    TrainReport,
    holdout_accuracy,
)


class SklearnMLPProvider(SuggestionProvider):
    id = "sklearn_mlp"
    display_name = "sklearn MLP"

    def __init__(self) -> None:
        self._model: MLPClassifier | None = None

    def config_schema(self) -> dict[str, Any]:
        return {
            "max_iter": {"type": "integer", "default": 200, "minimum": 1},
            "hidden_layer_sizes": {"type": "array", "default": [128, 64]},
        }

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def fit(self, X: np.ndarray, y: np.ndarray, *, config: TrainConfig) -> TrainReport:
        t0 = time.perf_counter()
        max_iter = int(config.provider_config.get("max_iter", 200))
        hidden = config.provider_config.get("hidden_layer_sizes", [128, 64])
        if isinstance(hidden, list):
            hidden = tuple(int(x) for x in hidden)
        early = len(y) >= 10
        template = MLPClassifier(
            hidden_layer_sizes=hidden,
            max_iter=max_iter,
            random_state=config.random_state,
            early_stopping=early,
            validation_fraction=0.1 if early else 0.0,
        )
        self._model = MLPClassifier(
            hidden_layer_sizes=hidden,
            max_iter=max_iter,
            random_state=config.random_state,
            early_stopping=early,
            validation_fraction=0.1 if early else 0.0,
        )
        self._model.fit(X, y)
        acc = holdout_accuracy(template, X, y, config.random_state)
        ms = (time.perf_counter() - t0) * 1000
        return TrainReport(
            success=True, message="MLP fitted", accuracy=acc, duration_ms=ms
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self._model is None:
            return None
        return self._model.predict_proba(X)
