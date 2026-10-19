from __future__ import annotations

import json
import math
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import default_session_config, merge_config
from app.db import get_db
from app.ml.features import extract_rgb_histogram, iter_files_recursive
from app.ml.projection import compute_kmeans, compute_pca, compute_umap
from app.ml.providers.base import TrainConfig, check_training_ready
from app.ml.providers.registry import create_provider
from app.models import Image, LabelingSession
from app.security import assert_file_under_root, resolve_allowed_path
from app.services.label_io import (
    MAX_LABEL,
    MIN_LABEL,
    LabelImportStats,
    LabelNameRow,
    build_export_zip,
    build_label_names_csv,
    build_labels_csv,
    build_session_image_index,
    default_label_name,
    extract_zip_payloads,
    parse_label_names_csv,
    parse_labels_csv,
    relative_image_path,
    resolve_image_id,
)

# Errors from DB, JSON config, NumPy/sklearn/torch, and optional deps (e.g. umap).
_ML_PIPELINE_ERRORS: tuple[type[BaseException], ...] = (
    SQLAlchemyError,
    json.JSONDecodeError,
    ArithmeticError,
    ImportError,
    IndexError,
    MemoryError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass
class RuntimeState:
    provider: object | None = None
    provider_id: str | None = None
    training_status: str = "idle"
    training_message: str = ""
    last_train_ms: float | None = None
    holdout_accuracy: float | None = None
    projection_coords: np.ndarray | None = None
    projection_method: str | None = None
    explained_variance: list[float] | None = None
    kmeans_labels: np.ndarray | None = None
    projection_status: str = "idle"
    projection_message: str = ""
    train_timer: threading.Timer | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


_runtime_states: dict[str, RuntimeState] = {}


def _runtime(session_id: str) -> RuntimeState:
    if session_id not in _runtime_states:
        _runtime_states[session_id] = RuntimeState()
    return _runtime_states[session_id]


def _feature_matrix(db: Session, session_id: str) -> tuple[np.ndarray, list[dict]]:
    rows = db.scalars(
        select(Image).where(Image.session_id == session_id).order_by(Image.idx)
    ).all()
    vecs = []
    meta = []
    for r in rows:
        if r.features is None:
            continue
        vec = np.frombuffer(r.features, dtype=np.float64)
        vecs.append(vec)
        meta.append({"id": r.id, "idx": r.idx, "label": r.label})
    if not vecs:
        return np.zeros((0, 1)), []
    return np.vstack(vecs), meta


def create_session(image_root: str, config_overrides: dict | None) -> str:
    root = resolve_allowed_path(image_root)
    session_id = str(uuid.uuid4())
    config = merge_config(default_session_config(), config_overrides)
    with get_db() as db:
        db.add(
            LabelingSession(
                id=session_id,
                image_root=str(root),
                config_json=json.dumps(config),
                created_at=time.time(),
                index_status="running",
            )
        )
    thread = threading.Thread(
        target=_index_session, args=(session_id, root), daemon=True
    )
    thread.start()
    return session_id


def _index_progress_estimate(scanned: int) -> float:
    """Progress while total file count is unknown (large trees)."""
    if scanned <= 0:
        return 0.0
    return min(0.99, math.log1p(scanned) / math.log1p(250_000))


def _upsert_image(
    db: Session,
    *,
    session_id: str,
    path: str,
    mtime: float,
    idx: int,
    features: bytes,
) -> None:
    stmt = pg_insert(Image).values(
        session_id=session_id,
        path=path,
        mtime=mtime,
        idx=idx,
        features=features,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_images_session_path",
        set_={
            "mtime": stmt.excluded.mtime,
            "idx": stmt.excluded.idx,
            "features": stmt.excluded.features,
        },
    )
    db.execute(stmt)


def _index_session(session_id: str, root: Path) -> None:
    try:
        config = default_session_config()
        with get_db() as db:
            row = db.get(LabelingSession, session_id)
            if row:
                config = json.loads(row.config_json)
            db.execute(
                update(LabelingSession)
                .where(LabelingSession.id == session_id)
                .values(index_message="Indexing images…")
            )
        bins = int(config.get("histogram_bins", 18))

        indexed = 0
        scanned = 0
        commit_batch = 100
        report_every = 250
        with get_db() as db:
            for path in iter_files_recursive(root):
                scanned += 1
                feat = extract_rgb_histogram(path, histogram_bins=bins)
                if feat is None:
                    if scanned % report_every == 0:
                        db.execute(
                            update(LabelingSession)
                            .where(LabelingSession.id == session_id)
                            .values(
                                index_progress=_index_progress_estimate(scanned),
                                index_message=(
                                    f"Indexed {indexed} RGB images "
                                    f"({scanned:,} files scanned)"
                                ),
                                total_images=indexed,
                            )
                        )
                        db.commit()
                    continue
                mtime = path.stat().st_mtime
                _upsert_image(
                    db,
                    session_id=session_id,
                    path=str(path),
                    mtime=mtime,
                    idx=indexed,
                    features=feat.astype(np.float64).tobytes(),
                )
                indexed += 1
                if (
                    indexed == 1
                    or indexed % commit_batch == 0
                    or scanned % report_every == 0
                ):
                    db.execute(
                        update(LabelingSession)
                        .where(LabelingSession.id == session_id)
                        .values(
                            index_progress=_index_progress_estimate(scanned),
                            index_message=(
                                f"Indexed {indexed} RGB images "
                                f"({scanned:,} files scanned)"
                            ),
                            total_images=indexed,
                        )
                    )
                    db.commit()
            count = db.scalar(
                select(func.count())
                .select_from(Image)
                .where(Image.session_id == session_id)
            )
            db.execute(
                update(LabelingSession)
                .where(LabelingSession.id == session_id)
                .values(
                    index_status="done",
                    index_progress=1.0,
                    index_message=f"Indexed {count} RGB images",
                    total_images=count or 0,
                )
            )
        _recompute_projection(session_id, method="pca")
        _schedule_train(session_id)
    except _ML_PIPELINE_ERRORS as e:
        with get_db() as db:
            db.execute(
                update(LabelingSession)
                .where(LabelingSession.id == session_id)
                .values(index_status="error", index_message=f"Indexing failed: {e}")
            )


def get_session(session_id: str) -> dict | None:
    with get_db() as db:
        row = db.get(LabelingSession, session_id)
        if not row:
            return None
        labeled = db.scalar(
            select(func.count())
            .select_from(Image)
            .where(Image.session_id == session_id, Image.label.is_not(None))
        )
        total = db.scalar(
            select(func.count())
            .select_from(Image)
            .where(Image.session_id == session_id)
        )
        labeled = labeled or 0
        total = total or 0
        return {
            "id": row.id,
            "image_root": row.image_root,
            "config": json.loads(row.config_json),
            "index_status": row.index_status,
            "index_progress": row.index_progress,
            "index_message": row.index_message,
            "total_images": total,
            "labeled_count": labeled,
            "unlabeled_count": total - labeled,
        }


def list_sessions(limit: int = 50) -> list[dict]:
    total_sq = (
        select(func.count())
        .where(Image.session_id == LabelingSession.id)
        .correlate(LabelingSession)
        .scalar_subquery()
    )
    labeled_sq = (
        select(func.count())
        .where(
            Image.session_id == LabelingSession.id,
            Image.label.is_not(None),
        )
        .correlate(LabelingSession)
        .scalar_subquery()
    )
    with get_db() as db:
        rows = db.execute(
            select(LabelingSession, total_sq, labeled_sq)
            .order_by(LabelingSession.created_at.desc())
            .limit(limit)
        ).all()
    out: list[dict] = []
    for row, total_images, labeled_count in rows:
        total = int(total_images or 0)
        labeled = int(labeled_count or 0)
        out.append(
            {
                "id": row.id,
                "image_root": row.image_root,
                "config": json.loads(row.config_json),
                "index_status": row.index_status,
                "index_progress": row.index_progress,
                "index_message": row.index_message,
                "total_images": total,
                "labeled_count": labeled,
                "unlabeled_count": total - labeled,
            }
        )
    return out


def delete_session(session_id: str) -> bool:
    rt = _runtime_states.get(session_id)
    if rt is not None:
        with rt.lock:
            if rt.train_timer is not None:
                rt.train_timer.cancel()
        del _runtime_states[session_id]
    with get_db() as db:
        result = db.execute(
            delete(LabelingSession).where(LabelingSession.id == session_id)
        )
        return result.rowcount > 0


def get_status(session_id: str) -> dict | None:
    sess = get_session(session_id)
    if not sess:
        return None
    rt = _runtime(session_id)
    with rt.lock:
        return {
            **{k: sess[k] for k in ("index_status", "index_progress", "index_message")},
            "training_status": rt.training_status,
            "training_message": rt.training_message,
            "projection_status": rt.projection_status,
            "projection_message": rt.projection_message,
            "projection_method": rt.projection_method,
        }


def list_images(
    session_id: str, offset: int = 0, limit: int = 50, unlabeled_only: bool = False
) -> dict | None:
    if not get_session(session_id):
        return None
    filters = [Image.session_id == session_id]
    if unlabeled_only:
        filters.append(Image.label.is_(None))
    with get_db() as db:
        total = db.scalar(select(func.count()).select_from(Image).where(*filters))
        rows = db.scalars(
            select(Image)
            .where(*filters)
            .order_by(Image.idx)
            .limit(limit)
            .offset(offset)
        ).all()
    items = []
    for r in rows:
        sug = get_suggestion(session_id, r.id)
        items.append(
            {
                "id": r.id,
                "idx": r.idx,
                "path": r.path,
                "filename": Path(r.path).name,
                "label": r.label,
                "label_source": r.label_source,
                "suggested_label": sug.get("label") if sug.get("ready") else None,
                "suggestion_confidence": sug.get("confidence"),
            }
        )
    return {
        "items": items,
        "total": total or 0,
        "offset": offset,
        "limit": limit,
    }


def get_image(session_id: str, image_id: int) -> dict | None:
    with get_db() as db:
        r = db.scalar(
            select(Image).where(
                Image.session_id == session_id,
                Image.id == image_id,
            )
        )
    if not r:
        return None
    return {
        "id": r.id,
        "idx": r.idx,
        "path": r.path,
        "filename": Path(r.path).name,
        "label": r.label,
        "label_source": r.label_source,
    }


def get_image_path(session_id: str, image_id: int) -> Path | None:
    sess = get_session(session_id)
    if not sess:
        return None
    with get_db() as db:
        r = db.scalar(
            select(Image.path).where(
                Image.session_id == session_id,
                Image.id == image_id,
            )
        )
    if not r:
        return None
    root = Path(sess["image_root"])
    return assert_file_under_root(Path(r), root)


def set_label(session_id: str, image_id: int, label: int | None, source: str) -> bool:
    with get_db() as db:
        result = db.execute(
            update(Image)
            .where(Image.session_id == session_id, Image.id == image_id)
            .values(
                label=label,
                label_source=source if label is not None else None,
            )
        )
        if result.rowcount == 0:
            return False
    _schedule_train(session_id)
    _schedule_projection(session_id)
    return True


def _schedule_train(session_id: str) -> None:
    rt = _runtime(session_id)
    with rt.lock:
        if rt.train_timer:
            rt.train_timer.cancel()
        from app.config import settings

        delay = settings.train_debounce_ms / 1000.0
        rt.train_timer = threading.Timer(delay, lambda: _train(session_id))
        rt.train_timer.daemon = True
        rt.train_timer.start()


def _schedule_projection(session_id: str) -> None:
    threading.Thread(
        target=_recompute_projection, args=(session_id, "pca"), daemon=True
    ).start()


def _labeled_xy(
    db: Session, session_id: str, config: dict
) -> tuple[np.ndarray, np.ndarray] | None:
    rows = db.scalars(
        select(Image).where(
            Image.session_id == session_id,
            Image.label.is_not(None),
            Image.features.is_not(None),
        )
    ).all()
    if not rows:
        return None
    X = np.vstack([np.frombuffer(r.features, dtype=np.float64) for r in rows])
    y = np.array([int(r.label) for r in rows], dtype=np.int64)
    ok, _ = check_training_ready(
        y,
        int(config.get("min_labeled_total", 30)),
        int(config.get("min_labeled_per_class", 3)),
        active_labels=None,
    )
    if not ok:
        return None
    return X, y


def _train(session_id: str) -> None:
    rt = _runtime(session_id)
    with rt.lock:
        rt.training_status = "running"
        rt.training_message = "Training..."
    try:
        with get_db() as db:
            row = db.get(LabelingSession, session_id)
            if not row:
                return
            config = json.loads(row.config_json)
            data = _labeled_xy(db, session_id, config)
            provider_id = config.get("suggestion_provider", "sklearn_mlp")
            if data is None:
                with rt.lock:
                    rt.provider = None
                    rt.provider_id = provider_id
                    rt.training_status = "waiting"
                    rt.training_message = "Not enough labeled data yet"
                    rt.holdout_accuracy = None
                return
            X, y = data
            provider = create_provider(provider_id)
            pc = config.get(provider_id, {})
            report = provider.fit(
                X,
                y,
                config=TrainConfig(
                    provider_config=pc,
                    min_labeled_total=int(config.get("min_labeled_total", 30)),
                    min_labeled_per_class=int(config.get("min_labeled_per_class", 3)),
                ),
            )
            with rt.lock:
                rt.provider = provider
                rt.provider_id = provider_id
                rt.training_status = "done" if report.success else "error"
                rt.training_message = report.message
                rt.last_train_ms = report.duration_ms
                rt.holdout_accuracy = report.accuracy
    except _ML_PIPELINE_ERRORS as e:
        with rt.lock:
            rt.training_status = "error"
            rt.training_message = str(e)


def force_train(session_id: str) -> dict:
    _train(session_id)
    rt = _runtime(session_id)
    with rt.lock:
        return {
            "success": rt.training_status == "done",
            "provider": rt.provider_id or "",
            "message": rt.training_message,
            "accuracy": rt.holdout_accuracy,
            "duration_ms": rt.last_train_ms,
        }


def get_suggestion(session_id: str, image_id: int) -> dict:
    rt = _runtime(session_id)
    with rt.lock:
        provider = rt.provider
        provider_id = rt.provider_id
        if provider is None or not getattr(provider, "is_ready", False):
            return {
                "ready": False,
                "reason": rt.training_message or "Model not ready",
                "provider": provider_id,
            }
    with get_db() as db:
        r = db.scalar(
            select(Image).where(
                Image.session_id == session_id,
                Image.id == image_id,
            )
        )
    if not r or r.features is None:
        return {"ready": False, "reason": "No features", "provider": provider_id}
    if r.label is not None:
        return {"ready": False, "reason": "Already labeled", "provider": provider_id}
    X = np.frombuffer(r.features, dtype=np.float64).reshape(1, -1)
    try:
        pred = provider.predict(X)[0]
        conf = None
        proba = provider.predict_proba(X)
        if proba is not None:
            conf = float(np.max(proba))
        return {
            "ready": True,
            "label": int(pred),
            "confidence": conf,
            "provider": provider_id,
        }
    except _ML_PIPELINE_ERRORS as e:
        return {"ready": False, "reason": str(e), "provider": provider_id}


def _recompute_projection(session_id: str, method: str = "pca") -> None:
    rt = _runtime(session_id)
    with rt.lock:
        rt.projection_status = "running"
        rt.projection_message = f"Computing {method.upper()}..."
    try:
        with get_db() as db:
            row = db.get(LabelingSession, session_id)
            config = json.loads(row.config_json) if row else default_session_config()
            X, meta = _feature_matrix(db, session_id)
            if len(X) == 0:
                with rt.lock:
                    rt.projection_status = "idle"
                    rt.projection_message = "No features"
                return
            if method == "umap":
                coords = compute_umap(X)
                ev = None
            else:
                coords, ev = compute_pca(X)
            n_clusters = len(config.get("active_labels", list(range(10))))
            kmeans = (
                compute_kmeans(X, n_clusters)
                if config.get("show_kmeans_overlay")
                else None
            )
            with rt.lock:
                rt.projection_coords = coords
                rt.projection_method = method
                rt.explained_variance = ev
                rt.kmeans_labels = kmeans
                rt._projection_meta = meta
                rt.projection_status = "done"
                rt.projection_message = f"{method.upper()} ready"
    except _ML_PIPELINE_ERRORS as e:
        with rt.lock:
            rt.projection_status = "error"
            rt.projection_message = str(e)


def get_projection(session_id: str, method: str | None = None) -> dict | None:
    sess = get_session(session_id)
    if not sess:
        return None
    rt = _runtime(session_id)
    with rt.lock:
        if method and method != rt.projection_method:
            threading.Thread(
                target=_recompute_projection, args=(session_id, method), daemon=True
            ).start()
            return {
                "points": [],
                "method": method,
                "explained_variance": None,
                "pending": True,
            }
        if rt.projection_coords is None:
            threading.Thread(
                target=_recompute_projection, args=(session_id, "pca"), daemon=True
            ).start()
            return {
                "points": [],
                "method": "pca",
                "explained_variance": None,
                "pending": True,
            }
        coords = rt.projection_coords
        meta = getattr(rt, "_projection_meta", [])
        kmeans = rt.kmeans_labels
        ev = rt.explained_variance
        proj_method = rt.projection_method or "pca"
    points = []
    for i, m in enumerate(meta):
        pt = {
            "id": m["id"],
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "label": m["label"],
            "idx": m["idx"],
            "kmeans_cluster": int(kmeans[i]) if kmeans is not None else None,
        }
        points.append(pt)
    return {"points": points, "method": proj_method, "explained_variance": ev}


def get_stats(session_id: str) -> dict | None:
    sess = get_session(session_id)
    if not sess:
        return None
    rt = _runtime(session_id)
    per_label: dict[str, int] = {str(i): 0 for i in range(10)}
    with get_db() as db:
        rows = db.execute(
            select(Image.label, func.count())
            .where(Image.session_id == session_id, Image.label.is_not(None))
            .group_by(Image.label)
        ).all()
        for label, count in rows:
            per_label[str(label)] = count
    with rt.lock:
        model_ready = rt.provider is not None and getattr(
            rt.provider, "is_ready", False
        )
        return {
            "total": sess["total_images"],
            "labeled": sess["labeled_count"],
            "unlabeled": sess["unlabeled_count"],
            "per_label": per_label,
            "model_ready": model_ready,
            "provider": rt.provider_id,
            "last_train_ms": rt.last_train_ms,
            "holdout_accuracy": rt.holdout_accuracy,
            "training_status": rt.training_status,
            "projection_method": rt.projection_method,
        }


def export_labels_zip(session_id: str) -> bytes | None:
    sess = get_session(session_id)
    if not sess:
        return None
    image_root = Path(sess["image_root"])
    config = sess["config"]
    label_names: dict[str, str] = dict(config.get("label_names") or {})

    with get_db() as db:
        rows = db.execute(
            select(Image.path, Image.label)
            .where(Image.session_id == session_id, Image.label.is_not(None))
            .order_by(Image.idx)
        ).all()

    export_rows: list[tuple[str, str, int]] = []
    used_labels: set[int] = set()
    for path, label in rows:
        if label is None:
            continue
        rel = relative_image_path(image_root, path)
        name = Path(path).name
        export_rows.append((rel, name, int(label)))
        used_labels.add(int(label))

    labels_csv = build_labels_csv(export_rows)
    names_csv = build_label_names_csv(label_names, used_labels)
    return build_export_zip(labels_csv, names_csv)


def import_labels(
    session_id: str,
    *,
    labels_data: bytes,
    label_names_data: bytes | None = None,
) -> dict | None:
    sess = get_session(session_id)
    if not sess:
        return None

    stats = LabelImportStats()
    try:
        label_rows = parse_labels_csv(labels_data)
    except ValueError as e:
        stats.add_error(str(e))
        stats.skipped_invalid += 1
        return _import_result_dict(stats, config_updated=False)

    name_rows: list[LabelNameRow] = []
    if label_names_data:
        try:
            name_rows = parse_label_names_csv(label_names_data)
        except ValueError as e:
            stats.add_error(str(e))
            stats.skipped_invalid += 1
            return _import_result_dict(stats, config_updated=False)

    image_root = Path(sess["image_root"])
    with get_db() as db:
        path_rows = db.execute(
            select(Image.id, Image.path).where(Image.session_id == session_id)
        ).all()

    index = build_session_image_index(
        image_root, [(int(r[0]), r[1]) for r in path_rows]
    )

    updates: dict[int, int] = {}
    for i, row in enumerate(label_rows, start=1):
        if row.label < MIN_LABEL or row.label > MAX_LABEL:
            stats.skipped_invalid += 1
            stats.add_error(f"Row {i}: label {row.label} out of range 0–9")
            continue
        resolved = resolve_image_id(row, index)
        if resolved == "no_key":
            stats.skipped_invalid += 1
            stats.add_error(f"Row {i}: missing image_path and image_name")
            continue
        if resolved == "ambiguous":
            stats.skipped_ambiguous += 1
            stats.add_error(f"Row {i}: ambiguous image_name (duplicate basename)")
            continue
        if resolved == "unknown":
            stats.skipped_unknown += 1
            key = row.image_path or row.image_name or "?"
            stats.add_error(f"Row {i}: image not found ({key})")
            continue
        updates[int(resolved)] = row.label

    config_updated = False
    if updates:
        with get_db() as db:
            for image_id, label in updates.items():
                db.execute(
                    update(Image)
                    .where(Image.session_id == session_id, Image.id == image_id)
                    .values(label=label, label_source="user")
                )
        stats.applied = len(updates)
        _schedule_train(session_id)
        _schedule_projection(session_id)

    if name_rows or updates:
        config = dict(sess["config"])
        label_names = dict(config.get("label_names") or {})
        active = list(config.get("active_labels") or [0, 1])
        slots = int(config.get("label_slots", 10))
        active_set = set(active)
        for label_id in updates.values():
            if label_id not in active_set and len(active_set) < slots:
                active_set.add(label_id)
        for nr in name_rows:
            if nr.label < MIN_LABEL or nr.label > MAX_LABEL:
                stats.skipped_invalid += 1
                stats.add_error(f"label_names: label {nr.label} out of range 0–9")
                continue
            default = default_label_name(nr.label)
            if nr.name.strip() == default:
                label_names.pop(str(nr.label), None)
            else:
                label_names[str(nr.label)] = nr.name.strip()
            if nr.label not in active_set and len(active_set) < slots:
                active_set.add(nr.label)
        new_active = sorted(active_set)
        names_changed = label_names != config.get("label_names")
        active_changed = new_active != config.get("active_labels")
        if name_rows and (names_changed or active_changed):
            merged = merge_config(
                config,
                {"label_names": label_names, "active_labels": new_active},
            )
            with get_db() as db:
                db.execute(
                    update(LabelingSession)
                    .where(LabelingSession.id == session_id)
                    .values(config_json=json.dumps(merged))
                )
            config_updated = True
            rt = _runtime(session_id)
            with rt.lock:
                if rt.provider_id != merged.get("suggestion_provider"):
                    rt.provider = None
        elif updates and active_changed:
            merged = merge_config(config, {"active_labels": new_active})
            with get_db() as db:
                db.execute(
                    update(LabelingSession)
                    .where(LabelingSession.id == session_id)
                    .values(config_json=json.dumps(merged))
                )
            config_updated = True

    return _import_result_dict(stats, config_updated=config_updated)


def import_labels_from_upload(
    session_id: str,
    *,
    labels_file: bytes | None,
    label_names_file: bytes | None,
    archive_file: bytes | None,
) -> dict | None:
    labels_data: bytes | None = labels_file
    names_data: bytes | None = label_names_file
    if archive_file:
        try:
            labels_data, names_from_zip = extract_zip_payloads(archive_file)
        except (ValueError, zipfile.BadZipFile) as e:
            stats = LabelImportStats()
            stats.add_error(str(e))
            stats.skipped_invalid += 1
            return _import_result_dict(stats, config_updated=False)
        if names_from_zip is not None:
            names_data = names_from_zip
    if not labels_data:
        stats = LabelImportStats()
        stats.add_error("No labels file provided")
        stats.skipped_invalid += 1
        return _import_result_dict(stats, config_updated=False)
    return import_labels(
        session_id, labels_data=labels_data, label_names_data=names_data
    )


def _import_result_dict(stats: LabelImportStats, config_updated: bool) -> dict:
    return {
        "applied": stats.applied,
        "skipped_unknown": stats.skipped_unknown,
        "skipped_ambiguous": stats.skipped_ambiguous,
        "skipped_invalid": stats.skipped_invalid,
        "errors": stats.errors,
        "config_updated": config_updated,
    }


def patch_config(session_id: str, overrides: dict) -> dict | None:
    sess = get_session(session_id)
    if not sess:
        return None
    config = merge_config(sess["config"], overrides)
    with get_db() as db:
        db.execute(
            update(LabelingSession)
            .where(LabelingSession.id == session_id)
            .values(config_json=json.dumps(config))
        )
    rt = _runtime(session_id)
    with rt.lock:
        if rt.provider_id != config.get("suggestion_provider"):
            rt.provider = None
    _schedule_train(session_id)
    return config


def list_image_ids(session_id: str, unlabeled_only: bool = False) -> list[dict] | None:
    if not get_session(session_id):
        return None
    filters = [Image.session_id == session_id]
    if unlabeled_only:
        filters.append(Image.label.is_(None))
    with get_db() as db:
        rows = db.scalars(select(Image).where(*filters).order_by(Image.idx)).all()
    return [{"id": r.id, "idx": r.idx, "label": r.label} for r in rows]


def find_image_by_idx(
    session_id: str, idx: int, unlabeled_only: bool = False
) -> int | None:
    filters = [Image.session_id == session_id, Image.idx == idx]
    if unlabeled_only:
        filters.append(Image.label.is_(None))
    with get_db() as db:
        image_id = db.scalar(select(Image.id).where(*filters))
    return image_id


def next_unlabeled_id(session_id: str, after_idx: int | None = None) -> int | None:
    with get_db() as db:
        if after_idx is None:
            image_id = db.scalar(
                select(Image.id)
                .where(Image.session_id == session_id, Image.label.is_(None))
                .order_by(Image.idx)
                .limit(1)
            )
        else:
            image_id = db.scalar(
                select(Image.id)
                .where(
                    Image.session_id == session_id,
                    Image.label.is_(None),
                    Image.idx > after_idx,
                )
                .order_by(Image.idx)
                .limit(1)
            )
            if image_id is None:
                image_id = db.scalar(
                    select(Image.id)
                    .where(Image.session_id == session_id, Image.label.is_(None))
                    .order_by(Image.idx)
                    .limit(1)
                )
    return image_id
