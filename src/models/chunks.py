import enum
from typing import Optional, List
from sqlalchemy import String, Enum, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.core.database import Base
from pgvector.sqlalchemy import Vector
from .source_documents import SourceDocument
from uuid import uuid4
class ChunkLevel(enum.Enum):
    PARENT = "PARENT"
    CHILD = "CHILD"


class Chunk(Base):
    __tablename__ = 'chunks'
    id: Mapped[str] = mapped_column(String, primary_key=True, default = lambda: str(uuid4()))
    content : Mapped[str] = mapped_column(String)
    chunk_level : Mapped[ChunkLevel] = mapped_column(Enum(ChunkLevel))
    chunk_metadata : Mapped[dict] = mapped_column(JSON)
    embedding : Mapped[Vector] = mapped_column(Vector(1024))

    source_doc_id : Mapped[int] = mapped_column(
        ForeignKey('source_documents.id')
    )

    source_documents : Mapped['SourceDocument'] = relationship(back_populates='chunks')
    parent_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey('chunks.id'),
        nullable = True
    )
    parent : Mapped[Optional['Chunk']] = relationship(
        back_populates='children', remote_side=[id]
    )
    children: Mapped[List['Chunk']] = relationship(
        back_populates="parent"
    )



