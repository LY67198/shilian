"""drop pgvector embedding columns (migrate vector storage to Milvus)

只做 embedding 列删除 + 相关索引删除，不动其他表（包括 LangGraph checkpoint_*）
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'ead09c555370'
down_revision: Union[str, None] = 'f3a4b5c6d7e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """删除 pgvector embedding 列（向量已迁到 Milvus）"""
    # knowledge_chunks.embedding
    op.drop_index(
        'ix_knowledge_chunks_embedding',
        table_name='knowledge_chunks',
        postgresql_using='hnsw',
    )
    op.drop_column('knowledge_chunks', 'embedding')

    # question_bank.embedding
    op.drop_index(
        'ix_question_bank_embedding',
        table_name='question_bank',
        postgresql_using='hnsw',
    )
    op.drop_column('question_bank', 'embedding')


def downgrade() -> None:
    """回滚：恢复 pgvector embedding 列（不推荐，Milvus 才是目标）"""
    from sqlalchemy.dialects import postgresql
    from pgvector.sqlalchemy import Vector

    # knowledge_chunks
    op.add_column(
        'knowledge_chunks',
        op.Column('embedding', Vector(dim=1024), nullable=True),
    )
    op.create_index(
        'ix_knowledge_chunks_embedding',
        'knowledge_chunks',
        ['embedding'],
        unique=False,
        postgresql_using='hnsw',
    )

    # question_bank
    op.add_column(
        'question_bank',
        op.Column('embedding', Vector(dim=1024), nullable=True),
    )
    op.create_index(
        'ix_question_bank_embedding',
        'question_bank',
        ['embedding'],
        unique=False,
        postgresql_using='hnsw',
    )