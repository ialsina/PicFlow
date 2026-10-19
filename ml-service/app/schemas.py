from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TrainConfigBody(BaseModel):
    provider_config: dict[str, Any] = Field(default_factory=dict)
    min_labeled_total: int = 30
    min_labeled_per_class: int = 3
    random_state: int = 42


class FitRequest(BaseModel):
    X: list[list[float]]
    y: list[int]
    config: TrainConfigBody = Field(default_factory=TrainConfigBody)


class FitResponse(BaseModel):
    model_id: str
    success: bool
    message: str
    accuracy: float | None = None
    duration_ms: float = 0.0


class PredictRequest(BaseModel):
    X: list[list[float]]


class PredictResponse(BaseModel):
    labels: list[int]
    proba: list[list[float]] | None = None


class ProjectionMetaItem(BaseModel):
    id: int
    idx: int
    label: int | None = None


class ProjectionRequest(BaseModel):
    method: Literal["pca", "umap"] = "pca"
    X: list[list[float]]
    image_ids: list[int]
    meta: list[ProjectionMetaItem]
    n_clusters: int = 10
    show_kmeans_overlay: bool = False


class ProjectionPoint(BaseModel):
    id: int
    x: float
    y: float
    label: int | None = None


class ProjectionResponse(BaseModel):
    points: list[ProjectionPoint]
    explained_variance: list[float] | None = None
    kmeans_labels: list[int] | None = None


class ProviderInfo(BaseModel):
    id: str
    name: str
    config_schema: dict[str, Any]
