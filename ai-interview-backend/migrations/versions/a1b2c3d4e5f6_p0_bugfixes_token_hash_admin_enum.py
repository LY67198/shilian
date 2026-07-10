"""P0 bugfixes: clear bcrypt tokens + Admin.role String→Enum

Revision ID: a1b2c3d4e5f6
Revises: ead09c555370
Create Date: 2026-07-10
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'ead09c555370'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # P0-2: 清空所有旧 bcrypt token，强制重新登录
    op.execute("DELETE FROM admin_tokens")
    op.execute("DELETE FROM tokens")

    # P0-3: Admin.role String(20) → Enum(UserRole)
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE userrole AS ENUM ('admin', 'superadmin');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)
    op.execute("""
        ALTER TABLE admins
        ALTER COLUMN role TYPE userrole
        USING role::text::userrole;
    """)
    op.execute("""
        ALTER TABLE admins
        ALTER COLUMN role SET DEFAULT 'admin'::userrole;
    """)


def downgrade():
    # P0-3: revert role to varchar
    op.execute("""
        ALTER TABLE admins
        ALTER COLUMN role TYPE VARCHAR(20)
        USING role::text;
    """)
    op.execute("DROP TYPE IF EXISTS userrole CASCADE")

    # P0-2: cannot restore bcrypt tokens (data already lost)
