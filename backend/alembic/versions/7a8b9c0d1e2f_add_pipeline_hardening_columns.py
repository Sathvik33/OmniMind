"""Add pipeline hardening columns and enum values to ingestion_jobs

Revision ID: 7a8b9c0d1e2f
Revises: 6753009576a3
Create Date: 2026-07-26 22:58:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '7a8b9c0d1e2f'
down_revision = '6753009576a3'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS failed_stage VARCHAR;")
    op.execute("ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS traceback TEXT;")
    op.execute("ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS is_retryable INTEGER DEFAULT 1;")
    op.execute("ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT NOW();")

    for val in ['queued', 'running', 'completed', 'failed', 'parsing', 'vision_captioning', 'chunking_embedding', 'storing', 'dead_letter', 'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'PARSING', 'VISION_CAPTIONING', 'CHUNKING_EMBEDDING', 'STORING', 'DEAD_LETTER']:
        op.execute(f"ALTER TYPE processingstatus ADD VALUE IF NOT EXISTS '{val}';")



def downgrade():
    op.drop_column('ingestion_jobs', 'last_heartbeat')
    op.drop_column('ingestion_jobs', 'is_retryable')
    op.drop_column('ingestion_jobs', 'traceback')
    op.drop_column('ingestion_jobs', 'failed_stage')
