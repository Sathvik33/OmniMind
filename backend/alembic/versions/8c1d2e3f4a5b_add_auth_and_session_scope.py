"""Add auth fields, chat title, and artifact session scoping

Revision ID: 8c1d2e3f4a5b
Revises: 7a8b9c0d1e2f
Create Date: 2026-08-01 14:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = "8c1d2e3f4a5b"
down_revision = "7a8b9c0d1e2f"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("users", sa.Column("email", sa.String(), nullable=True))
    op.add_column("users", sa.Column("password_hash", sa.String(), nullable=True))
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.add_column("sessions", sa.Column("title", sa.String(), nullable=True))
    op.add_column(
        "sessions",
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
    )

    op.add_column("artifacts", sa.Column("session_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_artifacts_session_id",
        "artifacts",
        "sessions",
        ["session_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_artifacts_session_id", "artifacts", ["session_id"])


def downgrade():
    op.drop_index("ix_artifacts_session_id", table_name="artifacts")
    op.drop_constraint("fk_artifacts_session_id", "artifacts", type_="foreignkey")
    op.drop_column("artifacts", "session_id")

    op.drop_column("sessions", "updated_at")
    op.drop_column("sessions", "title")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_column("users", "password_hash")
    op.drop_column("users", "email")
