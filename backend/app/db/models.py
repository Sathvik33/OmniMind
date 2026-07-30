from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON, Enum as SAEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import enum
from .database import Base

class ProcessingStatus(enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PARSING = "parsing"
    VISION_CAPTIONING = "vision_captioning"
    CHUNKING_EMBEDDING = "chunking_embedding"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    sessions = relationship("Session", back_populates="user")
    artifacts = relationship("Artifact", back_populates="owner")

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatHistory", back_populates="session", cascade="all, delete-orphan")

class ChatRole(enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ChatHistory(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(SAEnum(ChatRole), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    session = relationship("Session", back_populates="messages")
    feedbacks = relationship("UserFeedback", back_populates="message", cascade="all, delete-orphan")

class UserFeedback(Base):
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("chat_history.id"), nullable=False)
    rating = Column(Integer, nullable=False) # e.g., 1 for thumbs up, -1 for thumbs down
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    message = relationship("ChatHistory", back_populates="feedbacks")

class Artifact(Base):
    __tablename__ = "artifacts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False) # MinIO path reference
    modality = Column(String, nullable=False) # document, image, video, audio
    upload_status = Column(SAEnum(ProcessingStatus, values_callable=lambda x: [e.value for e in x]), default=ProcessingStatus.COMPLETED)
    processing_status = Column(SAEnum(ProcessingStatus, values_callable=lambda x: [e.value for e in x]), default=ProcessingStatus.QUEUED)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    owner = relationship("User", back_populates="artifacts")
    metadata_entries = relationship("Metadata", back_populates="artifact", cascade="all, delete-orphan")
    vectors = relationship("VectorEmbedding", back_populates="artifact", cascade="all, delete-orphan")

class Metadata(Base):
    __tablename__ = "artifact_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    artifact_id = Column(Integer, ForeignKey("artifacts.id"), nullable=False)
    key = Column(String, index=True, nullable=False)
    value = Column(JSON, nullable=False) # Allows storing bounding boxes, EXIF, temporal data
    
    artifact = relationship("Artifact", back_populates="metadata_entries")

class VectorEmbedding(Base):
    __tablename__ = "vector_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    artifact_id = Column(Integer, ForeignKey("artifacts.id"), nullable=False)
    embedding_type = Column(String, index=True, nullable=False) # e.g., 'text', 'vision', 'ocr', 'summary'
    content = Column(Text, nullable=True) # Raw text chunk if applicable
    embedding = Column(Vector()) # Unbounded dimension to support both BGE-M3 (1024) and SigLIP2 (768)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    artifact = relationship("Artifact", back_populates="vectors")

class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    artifact_id = Column(Integer, ForeignKey("artifacts.id"), nullable=False)
    status = Column(SAEnum(ProcessingStatus, values_callable=lambda x: [e.value for e in x]), default=ProcessingStatus.QUEUED)

    retry_count = Column(Integer, default=0)
    failed_stage = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    traceback = Column(Text, nullable=True)
    is_retryable = Column(Integer, default=1)
    last_heartbeat = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    artifact = relationship("Artifact", backref="jobs")


class ModelCategory(enum.Enum):
    EMBEDDING = "embedding"
    PARSER = "parser"
    VISION = "vision"
    LLM = "llm"
    RERANKER = "reranker"

class AIModelRegistry(Base):
    __tablename__ = "ai_model_registry"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    version = Column(String, nullable=False)
    category = Column(SAEnum(ModelCategory), nullable=False)
    is_active = Column(Integer, default=1) # 1 for active, 0 for inactive
    config = Column(JSON, nullable=True) # Dimensions, API keys, endpoints
    created_at = Column(DateTime(timezone=True), server_default=func.now())
