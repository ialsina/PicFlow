from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PICFLOW_", env_file=".env", extra="ignore"
    )

    data_dir: Path = Field(default=Path("data"))
    allowed_roots: str = Field(
        default="",
        description="Comma-separated absolute paths allowed for image_root",
    )
    cors_origins: str = Field(default="http://localhost:5173,http://127.0.0.1:5173")
    histogram_bins: int = 18
    min_labeled_total: int = 30
    min_labeled_per_class: int = 3
    suggestion_provider: str = "sklearn_mlp"
    train_debounce_ms: int = 400

    def allowed_root_paths(self) -> list[Path]:
        if not self.allowed_roots.strip():
            return []
        return [
            Path(p.strip()).resolve()
            for p in self.allowed_roots.split(",")
            if p.strip()
        ]

    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)


def default_session_config() -> dict[str, Any]:
    return {
        "histogram_bins": settings.histogram_bins,
        "min_labeled_total": settings.min_labeled_total,
        "min_labeled_per_class": settings.min_labeled_per_class,
        "suggestion_provider": settings.suggestion_provider,
        "label_slots": 10,
        "active_labels": list(range(10)),
        "sklearn_mlp": {"max_iter": 200, "hidden_layer_sizes": [128, 64]},
        "knn": {"n_neighbors": 5},
        "pytorch_mlp": {
            "epochs": 50,
            "lr": 0.001,
            "batch_size": 32,
            "hidden": [128, 64],
        },
        "projection": {"default": "pca", "umap_min_labeled_for_auto": 100},
        "show_kmeans_overlay": False,
    }


def merge_config(
    base: dict[str, Any], overrides: dict[str, Any] | None
) -> dict[str, Any]:
    if not overrides:
        return dict(base)
    out = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def config_json(config: dict[str, Any]) -> str:
    return json.dumps(config)
