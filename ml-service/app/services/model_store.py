from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from app.providers.base import SuggestionProvider, TrainConfig, TrainReport
from app.providers.registry import create_provider


@dataclass
class _SessionModels:
    by_provider: dict[str, str] = field(default_factory=dict)


class ModelStore:
    def __init__(self) -> None:
        self._models: dict[str, SuggestionProvider] = {}
        self._sessions: dict[str, _SessionModels] = {}
        self._lock = threading.Lock()

    def model_id(self, session_id: str, provider_id: str) -> str:
        return f"{session_id}:{provider_id}"

    def fit(
        self,
        session_id: str,
        provider_id: str,
        X: np.ndarray,
        y: np.ndarray,
        config: TrainConfig,
    ) -> tuple[str, TrainReport]:
        provider = create_provider(provider_id)
        report = provider.fit(X, y, config=config)
        mid = self.model_id(session_id, provider_id)
        with self._lock:
            old = self._sessions.get(session_id)
            if old and provider_id in old.by_provider:
                old_mid = old.by_provider[provider_id]
                self._models.pop(old_mid, None)
            self._models[mid] = provider
            sess = self._sessions.setdefault(session_id, _SessionModels())
            sess.by_provider[provider_id] = mid
        return mid, report

    def get(self, model_id: str) -> SuggestionProvider | None:
        with self._lock:
            return self._models.get(model_id)

    def purge_session(self, session_id: str) -> None:
        with self._lock:
            sess = self._sessions.pop(session_id, None)
            if sess:
                for mid in sess.by_provider.values():
                    self._models.pop(mid, None)


store = ModelStore()
