from __future__ import annotations

from app.schemas import (
    CreateSessionRequest,
    ImageDetail,
    ImageListResponse,
    PatchConfigRequest,
    ProjectionResponse,
    SessionListResponse,
    SessionStatus,
    SessionSummary,
    SetLabelRequest,
    StatsResponse,
    SuggestionResponse,
    TrainReport,
)
from app.services import session_service
from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=SessionSummary)
def create_session(body: CreateSessionRequest):
    sid = session_service.create_session(body.image_root, body.config)
    sess = session_service.get_session(sid)
    if not sess:
        raise HTTPException(500, "Failed to create session")
    return SessionSummary(**sess)


@router.get("", response_model=SessionListResponse)
def list_sessions(limit: int = Query(50, ge=1, le=200)):
    items = session_service.list_sessions(limit)
    return SessionListResponse(items=[SessionSummary(**s) for s in items])


@router.get("/{session_id}", response_model=SessionSummary)
def get_session(session_id: str):
    sess = session_service.get_session(session_id)
    if not sess:
        raise HTTPException(404, "Session not found")
    return SessionSummary(**sess)


@router.delete("/{session_id}", status_code=204)
def delete_session(session_id: str):
    if not session_service.delete_session(session_id):
        raise HTTPException(404, "Session not found")
    return Response(status_code=204)


@router.get("/{session_id}/status", response_model=SessionStatus)
def get_status(session_id: str):
    st = session_service.get_status(session_id)
    if not st:
        raise HTTPException(404, "Session not found")
    return SessionStatus(**st)


@router.get("/{session_id}/image-ids")
def list_image_ids(session_id: str, unlabeled_only: bool = False):
    ids = session_service.list_image_ids(session_id, unlabeled_only)
    if ids is None:
        raise HTTPException(404, "Session not found")
    return {"items": ids}


@router.get("/{session_id}/images", response_model=ImageListResponse)
def list_images(
    session_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    unlabeled_only: bool = False,
):
    data = session_service.list_images(session_id, offset, limit, unlabeled_only)
    if data is None:
        raise HTTPException(404, "Session not found")
    return ImageListResponse(**data)


@router.get("/{session_id}/images/{image_id}", response_model=ImageDetail)
def get_image(session_id: str, image_id: int):
    img = session_service.get_image(session_id, image_id)
    if not img:
        raise HTTPException(404, "Image not found")
    return ImageDetail(**img)


@router.get("/{session_id}/images/{image_id}/file")
def get_image_file(
    session_id: str, image_id: int, thumb: int | None = Query(None, ge=32, le=2048)
):
    path = session_service.get_image_path(session_id, image_id)
    if not path:
        raise HTTPException(404, "Image not found")
    if thumb:
        from io import BytesIO

        from fastapi.responses import Response
        from PIL import Image

        with Image.open(path) as im:
            im.thumbnail((thumb, thumb))
            buf = BytesIO()
            fmt = im.format or "JPEG"
            im.save(buf, format=fmt)
            return Response(content=buf.getvalue(), media_type=f"image/{fmt.lower()}")
    return FileResponse(path)


@router.put("/{session_id}/images/{image_id}/label", response_model=ImageDetail)
def set_label(session_id: str, image_id: int, body: SetLabelRequest):
    ok = session_service.set_label(session_id, image_id, body.label, body.source)
    if not ok:
        raise HTTPException(404, "Image not found")
    img = session_service.get_image(session_id, image_id)
    return ImageDetail(**img)


@router.get(
    "/{session_id}/images/{image_id}/suggestion", response_model=SuggestionResponse
)
def get_suggestion(session_id: str, image_id: int):
    return SuggestionResponse(**session_service.get_suggestion(session_id, image_id))


@router.get("/{session_id}/projection")
def get_projection(
    session_id: str, method: str | None = Query(None, pattern="^(pca|umap)$")
):
    data = session_service.get_projection(session_id, method)
    if data is None:
        raise HTTPException(404, "Session not found")
    if data.get("pending"):
        return data
    return ProjectionResponse(**{k: v for k, v in data.items() if k != "pending"})


@router.get("/{session_id}/stats", response_model=StatsResponse)
def get_stats(session_id: str):
    st = session_service.get_stats(session_id)
    if not st:
        raise HTTPException(404, "Session not found")
    return StatsResponse(**st)


@router.post("/{session_id}/train", response_model=TrainReport)
def train(session_id: str):
    if not session_service.get_session(session_id):
        raise HTTPException(404, "Session not found")
    rep = session_service.force_train(session_id)
    return TrainReport(
        success=rep["success"],
        provider=rep["provider"],
        message=rep["message"],
        accuracy=rep.get("accuracy"),
        duration_ms=rep.get("duration_ms"),
    )


@router.patch("/{session_id}/config")
def patch_config(session_id: str, body: PatchConfigRequest):
    cfg = session_service.patch_config(session_id, body.config)
    if cfg is None:
        raise HTTPException(404, "Session not found")
    return {"config": cfg}
