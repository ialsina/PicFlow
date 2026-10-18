from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException

from app.config import settings


def resolve_allowed_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {resolved}")
    if not resolved.is_dir():
        raise HTTPException(
            status_code=400, detail=f"Path is not a directory: {resolved}"
        )
    roots = settings.allowed_root_paths()
    if roots and not any(resolved == r or r in resolved.parents for r in roots):
        raise HTTPException(
            status_code=403,
            detail="image_root is not under PICFLOW_ALLOWED_ROOTS",
        )
    return resolved


def assert_file_under_root(file_path: Path, root: Path) -> Path:
    file_resolved = file_path.resolve()
    root_resolved = root.resolve()
    if root_resolved not in file_resolved.parents and file_resolved != root_resolved:
        raise HTTPException(status_code=403, detail="Image path outside session root")
    if not file_resolved.is_file():
        raise HTTPException(status_code=404, detail="Image file not found")
    return file_resolved
