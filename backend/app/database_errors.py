from __future__ import annotations

from sqlalchemy.exc import SQLAlchemyError

from app.config import settings


def database_unavailable_detail(exc: SQLAlchemyError) -> str:
    url = settings.database_url
    err = str(exc)
    err_lower = err.lower()

    if "postgres" in url and (
        "resolve host" in err_lower or "name resolution" in err_lower
    ):
        return (
            f"Database unavailable ({url!r}). "
            "Hostname 'postgres' only resolves inside the Docker Compose network. "
            "If you run uvicorn on your machine (IDE, terminal), use localhost and "
            "the published Postgres port instead, e.g. "
            "postgresql+psycopg://picflow:picflow@localhost:5434/picflow when "
            "PICFLOW_POSTGRES_PORT=5434. "
            "Stop any local uvicorn on port 8000 if you intend to use the Docker "
            "backend only. "
            f"({err})"
        )

    if "connection refused" in err_lower and "localhost" in url:
        return (
            f"Database unavailable ({url!r}). Connection refused — is Postgres "
            "running? With compose and PICFLOW_POSTGRES_PORT=5434, the URL must "
            "use localhost:5434, not :5432. "
            f"({err})"
        )

    return (
        f"Database unavailable. Check PostgreSQL and PICFLOW_DATABASE_URL "
        f"({url!r}). {err}"
    )
