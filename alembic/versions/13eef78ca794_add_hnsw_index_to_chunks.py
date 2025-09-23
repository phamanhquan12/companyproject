"""Add HNSW index to chunks

Revision ID: 13eef78ca794
Revises: c4a52047b903
Create Date: 2025-09-23 10:18:16.726702

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector.sqlalchemy

# revision identifiers, used by Alembic.
revision: str = '13eef78ca794'
down_revision: Union[str, Sequence[str], None] = 'c4a52047b903'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index(
        'ix_chunks_embedding',          # A unique name for the index
        'chunks',                       # The table name
        ['embedding'],                  # The column to index
        unique=False,
        postgresql_using='hnsw',        # Specify the HNSW index type
        postgresql_ops={'embedding': 'vector_l2_ops'} # Specify the distance operator
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_chunks_embedding', table_name='chunks')
