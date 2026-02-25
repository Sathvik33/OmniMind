from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(default=500, gt=0)
    overlap: int = Field(default=50, ge=0)
    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)