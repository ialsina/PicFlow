from __future__ import annotations

from sqlalchemy import (
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class LabelingSession(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    image_root: Mapped[str] = mapped_column(Text, nullable=False)
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    index_status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    index_progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    index_message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    total_images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    images: Mapped[list[Image]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )


class Image(Base):
    __tablename__ = "images"
    __table_args__ = (
        UniqueConstraint("session_id", "path", name="uq_images_session_path"),
        Index("idx_images_session", "session_id"),
        Index("idx_images_session_idx", "session_id", "idx"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    path: Mapped[str] = mapped_column(Text, nullable=False)
    mtime: Mapped[float] = mapped_column(Float, nullable=False)
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[int | None] = mapped_column(Integer, nullable=True)
    label_source: Mapped[str | None] = mapped_column(String, nullable=True)
    features: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    session: Mapped[LabelingSession] = relationship(back_populates="images")
