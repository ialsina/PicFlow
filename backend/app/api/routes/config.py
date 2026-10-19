from __future__ import annotations

from app.config import settings
from app.schemas import AppConfigResponse
from fastapi import APIRouter

router = APIRouter(prefix="/api/config", tags=["config"])


@router.get("", response_model=AppConfigResponse)
def get_app_config():
    roots = settings.allowed_root_paths()
    return AppConfigResponse(
        default_image_root=settings.default_image_root(),
        allowed_image_roots=[str(p) for p in roots],
    )
