from __future__ import annotations

import logging

from app.schemas import ProviderInfo
from app.services import ml_client
from app.services.ml_client import MlServiceUnavailable
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/providers", tags=["providers"])


@router.get("", response_model=list[ProviderInfo])
def get_providers():
    try:
        items = ml_client.list_providers()
    except MlServiceUnavailable as e:
        raise HTTPException(503, f"ML service unavailable: {e}") from e
    return [ProviderInfo(**p) for p in items]
