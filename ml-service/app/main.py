from __future__ import annotations

from fastapi import FastAPI

from app.routes.v1 import router as v1_router

app = FastAPI(title="PicFlow ML Service")
app.include_router(v1_router)
