from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    image_root: str
    config: dict[str, Any] | None = None


class SessionSummary(BaseModel):
    id: str
    image_root: str
    config: dict[str, Any]
    index_status: str
    index_progress: float
    index_message: str
    total_images: int
    labeled_count: int
    unlabeled_count: int


class SessionListResponse(BaseModel):
    items: list[SessionSummary]


class SessionStatus(BaseModel):
    index_status: str
    index_progress: float
    index_message: str
    training_status: str
    training_message: str
    projection_status: str
    projection_message: str
    projection_method: str | None = None


class ImageListItem(BaseModel):
    id: int
    idx: int
    path: str
    filename: str
    label: int | None
    label_source: str | None
    suggested_label: int | None = None
    suggestion_confidence: float | None = None


class ImageListResponse(BaseModel):
    items: list[ImageListItem]
    total: int
    offset: int
    limit: int


class ImageDetail(BaseModel):
    id: int
    idx: int
    path: str
    filename: str
    label: int | None
    label_source: str | None


class SetLabelRequest(BaseModel):
    label: int | None = Field(default=None, ge=0, le=9)
    source: Literal["user", "accepted_suggestion"] = "user"


class ProjectionPoint(BaseModel):
    id: int
    x: float
    y: float
    label: int | None
    idx: int
    kmeans_cluster: int | None = None


class ProjectionResponse(BaseModel):
    points: list[ProjectionPoint]
    method: Literal["pca", "umap"]
    explained_variance: list[float] | None = None


class SuggestionResponse(BaseModel):
    ready: bool
    label: int | None = None
    confidence: float | None = None
    provider: str | None = None
    reason: str | None = None


class TrainReport(BaseModel):
    success: bool
    provider: str
    message: str
    accuracy: float | None = None
    duration_ms: float | None = None
    labeled_count: int = 0


class StatsResponse(BaseModel):
    total: int
    labeled: int
    unlabeled: int
    per_label: dict[str, int]
    model_ready: bool
    provider: str | None
    last_train_ms: float | None
    holdout_accuracy: float | None
    training_status: str
    projection_method: str | None


class ProviderInfo(BaseModel):
    id: str
    name: str
    config_schema: dict[str, Any]


class PatchConfigRequest(BaseModel):
    config: dict[str, Any]


class LabelImportResult(BaseModel):
    applied: int
    skipped_unknown: int
    skipped_ambiguous: int
    skipped_invalid: int
    errors: list[str] = Field(default_factory=list)
    config_updated: bool = False
