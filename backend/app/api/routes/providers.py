from __future__ import annotations

from app.ml.providers.registry import list_providers
from app.schemas import ProviderInfo
from fastapi import APIRouter

router = APIRouter(prefix="/api/providers", tags=["providers"])


@router.get("", response_model=list[ProviderInfo])
def get_providers():
    return [
        ProviderInfo(id=p.id, name=p.display_name, config_schema=p.config_schema())
        for p in list_providers()
    ]
