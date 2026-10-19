from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PICFLOW_ML_", env_file=".env", extra="ignore"
    )

    api_key: str = Field(
        default="",
        description="Bearer token; empty allows unauthenticated requests (dev only)",
    )
    data_dir: Path = Field(default=Path("data"))


settings = Settings()
