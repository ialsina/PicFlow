from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class MlServiceError(Exception):
    pass


class MlServiceUnavailable(MlServiceError):
    pass


@dataclass
class FitResult:
    model_id: str
    success: bool
    message: str
    accuracy: float | None
    duration_ms: float


@dataclass
class PredictResult:
    labels: list[int]
    proba: list[list[float]] | None


@dataclass
class ProjectionResult:
    points: list[dict[str, Any]]
    explained_variance: list[float] | None
    kmeans_labels: list[int] | None


def _headers() -> dict[str, str]:
    h: dict[str, str] = {"Content-Type": "application/json"}
    key = settings.ml_api_key.strip()
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _base_url() -> str:
    return settings.ml_base_url.rstrip("/")


def _client(timeout: float) -> httpx.Client:
    return httpx.Client(
        timeout=timeout,
        verify=settings.ml_tls_verify,
        headers=_headers(),
    )


def is_enabled() -> bool:
    return settings.ml_enabled


def check_health() -> bool:
    if not is_enabled():
        return False
    try:
        with _client(settings.ml_predict_timeout_s) as client:
            r = client.get(f"{_base_url()}/v1/health")
            return r.status_code == 200
    except httpx.HTTPError:
        return False


def list_providers() -> list[dict[str, Any]]:
    if not is_enabled():
        raise MlServiceUnavailable("ML service is disabled")
    try:
        with _client(settings.ml_timeout_s) as client:
            r = client.get(f"{_base_url()}/v1/providers")
            r.raise_for_status()
            return r.json()
    except httpx.HTTPError as e:
        raise MlServiceUnavailable(str(e)) from e


def fit(
    session_id: str,
    provider_id: str,
    X: list[list[float]],
    y: list[int],
    *,
    provider_config: dict[str, Any],
    min_labeled_total: int,
    min_labeled_per_class: int,
    random_state: int = 42,
) -> FitResult:
    if not is_enabled():
        raise MlServiceUnavailable("ML service is disabled")
    body = {
        "X": X,
        "y": y,
        "config": {
            "provider_config": provider_config,
            "min_labeled_total": min_labeled_total,
            "min_labeled_per_class": min_labeled_per_class,
            "random_state": random_state,
        },
    }
    try:
        with _client(settings.ml_fit_timeout_s) as client:
            r = client.post(
                f"{_base_url()}/v1/sessions/{session_id}/models/{provider_id}/fit",
                json=body,
            )
            if r.status_code >= 400:
                detail = r.text
                try:
                    payload = r.json()
                except ValueError:
                    logger.debug(
                        "ML fit error response is not JSON; using raw body",
                        exc_info=True,
                    )
                else:
                    if isinstance(payload, dict):
                        detail = payload.get("detail", detail)
                raise MlServiceError(str(detail))
            data = r.json()
            return FitResult(
                model_id=data["model_id"],
                success=data["success"],
                message=data["message"],
                accuracy=data.get("accuracy"),
                duration_ms=float(data.get("duration_ms", 0)),
            )
    except httpx.HTTPError as e:
        raise MlServiceUnavailable(str(e)) from e


def predict(model_id: str, X: list[list[float]]) -> PredictResult:
    if not is_enabled():
        raise MlServiceUnavailable("ML service is disabled")
    try:
        with _client(settings.ml_predict_timeout_s) as client:
            r = client.post(
                f"{_base_url()}/v1/models/{model_id}/predict",
                json={"X": X},
            )
            if r.status_code >= 400:
                raise MlServiceError(r.text)
            data = r.json()
            return PredictResult(labels=data["labels"], proba=data.get("proba"))
    except httpx.HTTPError as e:
        raise MlServiceUnavailable(str(e)) from e


def project(
    session_id: str,
    *,
    method: str,
    X: list[list[float]],
    image_ids: list[int],
    meta: list[dict[str, Any]],
    n_clusters: int,
    show_kmeans_overlay: bool,
) -> ProjectionResult:
    if not is_enabled():
        raise MlServiceUnavailable("ML service is disabled")
    body = {
        "method": method,
        "X": X,
        "image_ids": image_ids,
        "meta": meta,
        "n_clusters": n_clusters,
        "show_kmeans_overlay": show_kmeans_overlay,
    }
    try:
        with _client(settings.ml_timeout_s) as client:
            r = client.post(
                f"{_base_url()}/v1/sessions/{session_id}/projection",
                json=body,
            )
            if r.status_code >= 400:
                raise MlServiceError(r.text)
            data = r.json()
            return ProjectionResult(
                points=data["points"],
                explained_variance=data.get("explained_variance"),
                kmeans_labels=data.get("kmeans_labels"),
            )
    except httpx.HTTPError as e:
        raise MlServiceUnavailable(str(e)) from e


def purge_session(session_id: str) -> None:
    if not is_enabled():
        return
    try:
        with _client(settings.ml_predict_timeout_s) as client:
            r = client.delete(f"{_base_url()}/v1/sessions/{session_id}")
            if r.status_code not in (204, 404):
                logger.warning("ml purge session %s: %s", session_id, r.status_code)
    except httpx.HTTPError as e:
        logger.warning("ml purge session %s failed: %s", session_id, e)
