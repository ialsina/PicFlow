from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.routes import providers, sessions
from app.config import settings
from app.db import engine, init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title="PicFlow Labeling API", lifespan=lifespan)


@app.exception_handler(SQLAlchemyError)
async def database_error_handler(
    _request: Request, exc: SQLAlchemyError
) -> JSONResponse:
    logger.exception("Database error")
    return JSONResponse(
        status_code=503,
        content={
            "detail": (
                "Database unavailable. Ensure PostgreSQL is running and "
                "PICFLOW_DATABASE_URL is correct "
                f"(currently {settings.database_url!r})."
            ),
        },
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(sessions.router)
app.include_router(providers.router)


@app.get("/api/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Database unavailable. Ensure PostgreSQL is running and "
                f"PICFLOW_DATABASE_URL is set (currently {settings.database_url!r}). "
                f"Error: {exc}"
            ),
        ) from exc
    return {"status": "ok"}
