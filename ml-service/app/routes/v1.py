from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from app.projection import compute_kmeans, compute_pca, compute_umap
from app.providers.base import TrainConfig
from app.providers.registry import list_providers
from app.schemas import (
    FitRequest,
    FitResponse,
    PredictRequest,
    PredictResponse,
    ProjectionRequest,
    ProjectionResponse,
    ProjectionPoint,
    ProviderInfo,
)
from app.security import require_api_key
from app.services.model_store import store

router = APIRouter(prefix="/v1")


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/providers", response_model=list[ProviderInfo], dependencies=[Depends(require_api_key)])
def get_providers():
    return [
        ProviderInfo(id=p.id, name=p.display_name, config_schema=p.config_schema())
        for p in list_providers()
    ]


@router.post(
    "/sessions/{session_id}/models/{provider_id}/fit",
    response_model=FitResponse,
    dependencies=[Depends(require_api_key)],
)
def fit_model(session_id: str, provider_id: str, body: FitRequest):
    X = np.asarray(body.X, dtype=np.float64)
    y = np.asarray(body.y, dtype=np.int64)
    if X.ndim != 2 or len(y) != len(X):
        raise HTTPException(400, "X and y shape mismatch")
    try:
        cfg = TrainConfig(
            provider_config=body.config.provider_config,
            min_labeled_total=body.config.min_labeled_total,
            min_labeled_per_class=body.config.min_labeled_per_class,
            random_state=body.config.random_state,
        )
        model_id, report = store.fit(session_id, provider_id, X, y, cfg)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:
        raise HTTPException(500, str(e)) from e
    return FitResponse(
        model_id=model_id,
        success=report.success,
        message=report.message,
        accuracy=report.accuracy,
        duration_ms=report.duration_ms,
    )


@router.post(
    "/models/{model_id}/predict",
    response_model=PredictResponse,
    dependencies=[Depends(require_api_key)],
)
def predict(model_id: str, body: PredictRequest):
    provider = store.get(model_id)
    if provider is None or not provider.is_ready:
        raise HTTPException(404, "Model not found or not ready")
    X = np.asarray(body.X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    try:
        labels = provider.predict(X)
        proba = provider.predict_proba(X)
    except RuntimeError as e:
        raise HTTPException(500, str(e)) from e
    proba_out = proba.tolist() if proba is not None else None
    return PredictResponse(labels=[int(x) for x in labels], proba=proba_out)


@router.post(
    "/sessions/{session_id}/projection",
    response_model=ProjectionResponse,
    dependencies=[Depends(require_api_key)],
)
def project(session_id: str, body: ProjectionRequest):
    del session_id  # stateless projection per request
    X = np.asarray(body.X, dtype=np.float64)
    if len(X) == 0:
        return ProjectionResponse(points=[], explained_variance=None, kmeans_labels=None)
    if len(body.image_ids) != len(X) or len(body.meta) != len(X):
        raise HTTPException(400, "image_ids and meta must match X rows")
    try:
        if body.method == "umap":
            coords = compute_umap(X)
            ev = None
        else:
            coords, ev = compute_pca(X)
        kmeans = (
            compute_kmeans(X, body.n_clusters)
            if body.show_kmeans_overlay
            else None
        )
    except (ImportError, ValueError, RuntimeError) as e:
        raise HTTPException(500, str(e)) from e
    points = []
    for i, m in enumerate(body.meta):
        points.append(
            ProjectionPoint(
                id=m.id,
                x=float(coords[i, 0]),
                y=float(coords[i, 1]),
                label=m.label,
            )
        )
    kmeans_out = [int(k) for k in kmeans] if kmeans is not None else None
    return ProjectionResponse(
        points=points,
        explained_variance=ev,
        kmeans_labels=kmeans_out,
    )


@router.delete(
    "/sessions/{session_id}",
    status_code=204,
    dependencies=[Depends(require_api_key)],
)
def delete_session(session_id: str):
    store.purge_session(session_id)
