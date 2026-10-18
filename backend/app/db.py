from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from app.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    image_root TEXT NOT NULL,
    config_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    index_status TEXT NOT NULL DEFAULT 'pending',
    index_progress REAL NOT NULL DEFAULT 0,
    index_message TEXT NOT NULL DEFAULT '',
    total_images INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    path TEXT NOT NULL,
    mtime REAL NOT NULL,
    idx INTEGER NOT NULL,
    label INTEGER,
    label_source TEXT,
    features BLOB,
    UNIQUE(session_id, path),
    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_images_session ON images(session_id);
CREATE INDEX IF NOT EXISTS idx_images_session_idx ON images(session_id, idx);
"""


def db_path() -> Path:
    return settings.data_dir / "picflow.db"


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_conn():
    conn = connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
