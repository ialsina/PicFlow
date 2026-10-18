from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 5},
)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def init_db() -> None:
    alembic_ini = Path(__file__).resolve().parent.parent / "alembic.ini"
    cfg = Config(str(alembic_ini))
    cfg.set_main_option("sqlalchemy.url", settings.database_url)
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            command.upgrade(cfg, "head")
            return
        except OperationalError as exc:
            if attempt >= max_attempts - 1:
                raise RuntimeError(
                    "Could not reach PostgreSQL after "
                    f"{max_attempts} attempts. Check PICFLOW_DATABASE_URL "
                    f"({settings.database_url!r})."
                ) from exc
            time.sleep(1.0)


@contextmanager
def get_db():
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
