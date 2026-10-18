"""initial schema

Revision ID: 20260318_0001
Revises:
Create Date: 2026-03-18

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260318_0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("image_root", sa.Text(), nullable=False),
        sa.Column("config_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("index_status", sa.String(), nullable=False),
        sa.Column("index_progress", sa.Float(), nullable=False),
        sa.Column("index_message", sa.Text(), nullable=False),
        sa.Column("total_images", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "images",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("mtime", sa.Float(), nullable=False),
        sa.Column("idx", sa.Integer(), nullable=False),
        sa.Column("label", sa.Integer(), nullable=True),
        sa.Column("label_source", sa.String(), nullable=True),
        sa.Column("features", sa.LargeBinary(), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", "path", name="uq_images_session_path"),
    )
    op.create_index("idx_images_session", "images", ["session_id"], unique=False)
    op.create_index(
        "idx_images_session_idx", "images", ["session_id", "idx"], unique=False
    )


def downgrade() -> None:
    op.drop_index("idx_images_session_idx", table_name="images")
    op.drop_index("idx_images_session", table_name="images")
    op.drop_table("images")
    op.drop_table("sessions")
