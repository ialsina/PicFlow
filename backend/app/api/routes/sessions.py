from __future__ import annotations

from typing import Annotated

from app.schemas import (
    CreateSessionRequest,
    ImageDetail,
    ImageListResponse,
    LabelImportResult,
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
from app.services.label_io import export_archive_stem
from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

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


@router.get("/{session_id}/labels/export")
def export_labels(session_id: str):
    data = session_service.export_labels_zip(session_id)
    if data is None:
        raise HTTPException(404, "Session not found")
    filename = f"{export_archive_stem(session_id)}.zip"
    return StreamingResponse(
        iter([data]),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{session_id}/labels/import", response_model=LabelImportResult)
async def import_labels(
    session_id: str,
    labels: Annotated[UploadFile | None, File()] = None,
    label_names: Annotated[UploadFile | None, File()] = None,
    archive: Annotated[UploadFile | None, File()] = None,
):
    labels_bytes: bytes | None = None
    names_bytes: bytes | None = None
    archive_bytes: bytes | None = None
    if labels is not None:
        labels_bytes = await labels.read()
    if label_names is not None:
        names_bytes = await label_names.read()
    if archive is not None:
        archive_bytes = await archive.read()
    result = session_service.import_labels_from_upload(
        session_id,
        labels_file=labels_bytes,
        label_names_file=names_bytes,
        archive_file=archive_bytes,
    )
    if result is None:
        raise HTTPException(404, "Session not found")
    return LabelImportResult(**result)
