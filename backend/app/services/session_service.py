from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.config import default_session_config, merge_config
from app.db import get_conn, init_db
from app.ml.features import collect_rgb_image_paths, extract_rgb_histogram
from app.ml.projection import compute_kmeans, compute_pca, compute_umap
from app.ml.providers.base import TrainConfig, check_training_ready
from app.ml.providers.registry import create_provider
from app.security import assert_file_under_root, resolve_allowed_path

# Errors from DB, JSON config, NumPy/sklearn/torch, and optional deps (e.g. umap).
_ML_PIPELINE_ERRORS: tuple[type[BaseException], ...] = (
    sqlite3.Error,
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


def _load_config(row) -> dict:
    return json.loads(row["config_json"])


def _feature_matrix(conn, session_id: str) -> tuple[np.ndarray, list[dict]]:
    rows = conn.execute(
        "SELECT id, idx, path, label, features FROM images WHERE session_id = ? ORDER BY idx",
        (session_id,),
    ).fetchall()
    ids = []
    vecs = []
    meta = []
    for r in rows:
        if r["features"] is None:
            continue
        vec = np.frombuffer(r["features"], dtype=np.float64)
        vecs.append(vec)
        ids.append(r["id"])
        meta.append({"id": r["id"], "idx": r["idx"], "label": r["label"]})
    if not vecs:
        return np.zeros((0, 1)), []
    return np.vstack(vecs), meta


def create_session(image_root: str, config_overrides: dict | None) -> str:
    init_db()
    root = resolve_allowed_path(image_root)
    session_id = str(uuid.uuid4())
    config = merge_config(default_session_config(), config_overrides)
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO sessions (id, image_root, config_json, created_at, index_status)
            VALUES (?, ?, ?, ?, 'running')
            """,
            (session_id, str(root), json.dumps(config), time.time()),
        )
    thread = threading.Thread(
        target=_index_session, args=(session_id, root), daemon=True
    )
    thread.start()
    return session_id


def _index_session(session_id: str, root: Path) -> None:
    rgb_paths = collect_rgb_image_paths(root)
    total = len(rgb_paths)
    config = default_session_config()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT config_json FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row:
            config = json.loads(row["config_json"])
    bins = int(config.get("histogram_bins", 18))

    indexed = 0
    with get_conn() as conn:
        for i, path in enumerate(rgb_paths):
            feat = extract_rgb_histogram(path, histogram_bins=bins)
            if feat is None:
                continue
            mtime = path.stat().st_mtime
            conn.execute(
                """
                INSERT OR REPLACE INTO images (session_id, path, mtime, idx, features)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    str(path),
                    mtime,
                    indexed,
                    feat.astype(np.float64).tobytes(),
                ),
            )
            indexed += 1
            if total and (i + 1) % max(1, total // 50) == 0:
                conn.execute(
                    """
                    UPDATE sessions SET index_progress = ?, index_message = ?, total_images = ?
                    WHERE id = ?
                    """,
                    ((i + 1) / total, f"Indexed {i + 1}/{total}", indexed, session_id),
                )
        count = conn.execute(
            "SELECT COUNT(*) AS c FROM images WHERE session_id = ?", (session_id,)
        ).fetchone()["c"]
        conn.execute(
            """
            UPDATE sessions SET index_status = 'done', index_progress = 1, index_message = ?,
            total_images = ? WHERE id = ?
            """,
            (f"Indexed {count} RGB images", count, session_id),
        )
    _recompute_projection(session_id, method="pca")
    _schedule_train(session_id)


def get_session(session_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row:
            return None
        labeled = conn.execute(
            "SELECT COUNT(*) AS c FROM images WHERE session_id = ? AND label IS NOT NULL",
            (session_id,),
        ).fetchone()["c"]
        total = conn.execute(
            "SELECT COUNT(*) AS c FROM images WHERE session_id = ?", (session_id,)
        ).fetchone()["c"]
        return {
            "id": row["id"],
            "image_root": row["image_root"],
            "config": json.loads(row["config_json"]),
            "index_status": row["index_status"],
            "index_progress": row["index_progress"],
            "index_message": row["index_message"],
            "total_images": total,
            "labeled_count": labeled,
            "unlabeled_count": total - labeled,
        }


def list_sessions(limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT s.*,
                (SELECT COUNT(*) FROM images i WHERE i.session_id = s.id) AS total_images,
                (SELECT COUNT(*) FROM images i WHERE i.session_id = s.id AND i.label IS NOT NULL)
                    AS labeled_count
            FROM sessions s
            ORDER BY s.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    out: list[dict] = []
    for row in rows:
        total = int(row["total_images"])
        labeled = int(row["labeled_count"])
        out.append(
            {
                "id": row["id"],
                "image_root": row["image_root"],
                "config": json.loads(row["config_json"]),
                "index_status": row["index_status"],
                "index_progress": row["index_progress"],
                "index_message": row["index_message"],
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
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cur.rowcount > 0


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
    where = "session_id = ?"
    params: list = [session_id]
    if unlabeled_only:
        where += " AND label IS NULL"
    with get_conn() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS c FROM images WHERE {where}", params
        ).fetchone()["c"]
        rows = conn.execute(
            f"""
            SELECT id, idx, path, label, label_source FROM images
            WHERE {where} ORDER BY idx LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ).fetchall()
    items = []
    for r in rows:
        sug = get_suggestion(session_id, r["id"])
        items.append(
            {
                "id": r["id"],
                "idx": r["idx"],
                "path": r["path"],
                "filename": Path(r["path"]).name,
                "label": r["label"],
                "label_source": r["label_source"],
                "suggested_label": sug.get("label") if sug.get("ready") else None,
                "suggestion_confidence": sug.get("confidence"),
            }
        )
    return {"items": items, "total": total, "offset": offset, "limit": limit}


def get_image(session_id: str, image_id: int) -> dict | None:
    with get_conn() as conn:
        r = conn.execute(
            "SELECT id, idx, path, label, label_source FROM images WHERE session_id = ? AND id = ?",
            (session_id, image_id),
        ).fetchone()
    if not r:
        return None
    return {
        "id": r["id"],
        "idx": r["idx"],
        "path": r["path"],
        "filename": Path(r["path"]).name,
        "label": r["label"],
        "label_source": r["label_source"],
    }


def get_image_path(session_id: str, image_id: int) -> Path | None:
    sess = get_session(session_id)
    if not sess:
        return None
    with get_conn() as conn:
        r = conn.execute(
            "SELECT path FROM images WHERE session_id = ? AND id = ?",
            (session_id, image_id),
        ).fetchone()
    if not r:
        return None
    root = Path(sess["image_root"])
    return assert_file_under_root(Path(r["path"]), root)


def set_label(session_id: str, image_id: int, label: int | None, source: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            """
            UPDATE images SET label = ?, label_source = ? WHERE session_id = ? AND id = ?
            """,
            (label, source if label is not None else None, session_id, image_id),
        )
        if cur.rowcount == 0:
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
    conn, session_id: str, config: dict
) -> tuple[np.ndarray, np.ndarray] | None:
    rows = conn.execute(
        """
        SELECT features, label FROM images
        WHERE session_id = ? AND label IS NOT NULL AND features IS NOT NULL
        """,
        (session_id,),
    ).fetchall()
    if not rows:
        return None
    X = np.vstack([np.frombuffer(r["features"], dtype=np.float64) for r in rows])
    y = np.array([int(r["label"]) for r in rows], dtype=np.int64)
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
        with get_conn() as conn:
            row = conn.execute(
                "SELECT config_json FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if not row:
                return
            config = json.loads(row["config_json"])
            data = _labeled_xy(conn, session_id, config)
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
    with get_conn() as conn:
        r = conn.execute(
            "SELECT features, label FROM images WHERE session_id = ? AND id = ?",
            (session_id, image_id),
        ).fetchone()
    if not r or r["features"] is None:
        return {"ready": False, "reason": "No features", "provider": provider_id}
    if r["label"] is not None:
        return {"ready": False, "reason": "Already labeled", "provider": provider_id}
    X = np.frombuffer(r["features"], dtype=np.float64).reshape(1, -1)
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
        with get_conn() as conn:
            row = conn.execute(
                "SELECT config_json FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            config = json.loads(row["config_json"]) if row else default_session_config()
            X, meta = _feature_matrix(conn, session_id)
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
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT label, COUNT(*) AS c FROM images WHERE session_id = ? AND label IS NOT NULL GROUP BY label",
            (session_id,),
        ).fetchall()
        for r in rows:
            per_label[str(r["label"])] = r["c"]
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


def patch_config(session_id: str, overrides: dict) -> dict | None:
    sess = get_session(session_id)
    if not sess:
        return None
    config = merge_config(sess["config"], overrides)
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET config_json = ? WHERE id = ?",
            (json.dumps(config), session_id),
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
    where = "session_id = ?"
    params: list = [session_id]
    if unlabeled_only:
        where += " AND label IS NULL"
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT id, idx, label FROM images WHERE {where} ORDER BY idx",
            params,
        ).fetchall()
    return [{"id": r["id"], "idx": r["idx"], "label": r["label"]} for r in rows]


def find_image_by_idx(
    session_id: str, idx: int, unlabeled_only: bool = False
) -> int | None:
    where = "session_id = ? AND idx = ?"
    params: list = [session_id, idx]
    if unlabeled_only:
        where += " AND label IS NULL"
    with get_conn() as conn:
        r = conn.execute(f"SELECT id FROM images WHERE {where}", params).fetchone()
    return r["id"] if r else None


def next_unlabeled_id(session_id: str, after_idx: int | None = None) -> int | None:
    with get_conn() as conn:
        if after_idx is None:
            r = conn.execute(
                """
                SELECT id FROM images WHERE session_id = ? AND label IS NULL
                ORDER BY idx LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        else:
            r = conn.execute(
                """
                SELECT id FROM images WHERE session_id = ? AND label IS NULL AND idx > ?
                ORDER BY idx LIMIT 1
                """,
                (session_id, after_idx),
            ).fetchone()
            if not r:
                r = conn.execute(
                    """
                    SELECT id FROM images WHERE session_id = ? AND label IS NULL
                    ORDER BY idx LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
    return r["id"] if r else None
