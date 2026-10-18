from __future__ import annotations

from app.ml.providers.base import SuggestionProvider
from app.ml.providers.knn import KNNProvider
from app.ml.providers.pytorch_mlp import PyTorchMLPProvider
from app.ml.providers.sklearn_mlp import SklearnMLPProvider

_PROVIDERS: dict[str, type[SuggestionProvider]] = {
    KNNProvider.id: KNNProvider,
    SklearnMLPProvider.id: SklearnMLPProvider,
    PyTorchMLPProvider.id: PyTorchMLPProvider,
}


def create_provider(provider_id: str) -> SuggestionProvider:
    cls = _PROVIDERS.get(provider_id)
    if cls is None:
        raise ValueError(f"Unknown provider: {provider_id}")
    return cls()


def list_providers() -> list[SuggestionProvider]:
    return [cls() for cls in _PROVIDERS.values()]
