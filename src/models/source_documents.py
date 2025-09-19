import enum
from datetime import datetime
from sqlalchemy import String, Integer, Enum, DateTime, func, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.core.database import Base

class IngestStatus(enum.Enum):
    PENDING = 'PENDING'
    PROCESSING = 'PROCESSING'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'

class SourceDocument(Base):
    __tablename__ = 'source_documents'
    id: Mapped[int] = mapped_column(primary_key=True)
    media_id : Mapped[int] = mapped_column(Integer, unique = True, index = True)
    file_name: Mapped[str] = mapped_column(String)
    file_path: Mapped[str] = mapped_column(String, unique=True)
    page_count: Mapped[int] = mapped_column(Integer)
    status: Mapped[IngestStatus] = mapped_column(
        Enum(IngestStatus), default=IngestStatus.PENDING
    )
    created_at: Mapped[datetime] =   mapped_column(DateTime, default=func.now())
    processed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    chunks:Mapped[list['Chunk']] = relationship(
        back_populates='source_documents',
        cascade='all, delete-orphan'               
    )

