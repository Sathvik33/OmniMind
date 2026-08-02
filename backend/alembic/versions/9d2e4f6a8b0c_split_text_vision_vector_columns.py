"""Split text (1024) and vision (768) vector columns with ANN indexes

Revision ID: 9d2e4f6a8b0c
Revises: 8c1d2e3f4a5b
Create Date: 2026-08-02 10:40:00.000000

"""
from alembic import op


revision = "9d2e4f6a8b0c"
down_revision = "8c1d2e3f4a5b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE vector_embeddings "
        "ADD COLUMN IF NOT EXISTS embedding_text vector(1024)"
    )
    op.execute(
        "ALTER TABLE vector_embeddings "
        "ADD COLUMN IF NOT EXISTS embedding_vision vector(768)"
    )

    # Backfill from legacy unbounded column by embedding_type
    op.execute(
        """
        UPDATE vector_embeddings
        SET embedding_text = embedding::vector(1024)
        WHERE embedding_type <> 'vision'
          AND embedding IS NOT NULL
          AND embedding_text IS NULL
          AND vector_dims(embedding) = 1024
        """
    )
    op.execute(
        """
        UPDATE vector_embeddings
        SET embedding_vision = embedding::vector(768)
        WHERE embedding_type = 'vision'
          AND embedding IS NOT NULL
          AND embedding_vision IS NULL
          AND vector_dims(embedding) = 768
        """
    )

    # Partial HNSW indexes for per-modality ANN (pgvector >= 0.5)
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_ve_embedding_text_hnsw
        ON vector_embeddings
        USING hnsw (embedding_text vector_cosine_ops)
        WHERE embedding_text IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_ve_embedding_vision_hnsw
        ON vector_embeddings
        USING hnsw (embedding_vision vector_cosine_ops)
        WHERE embedding_vision IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_ve_embedding_vision_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_ve_embedding_text_hnsw")
    op.execute("ALTER TABLE vector_embeddings DROP COLUMN IF EXISTS embedding_vision")
    op.execute("ALTER TABLE vector_embeddings DROP COLUMN IF EXISTS embedding_text")
