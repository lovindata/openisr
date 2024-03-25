"""empty message

Revision ID: ef77c28dd953
Revises: 4a6ce2b1abb0
Create Date: 2024-03-25 06:54:34.892880

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision: str = 'ef77c28dd953'
down_revision: Union[str, None] = '4a6ce2b1abb0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('card_downloads', sa.Column('image_bytes', sa.LargeBinary(), nullable=False))
    op.add_column('card_thumbnails', sa.Column('thumbnail_bytes', sa.LargeBinary(), nullable=False))
    op.drop_column('card_thumbnails', 'data')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('card_thumbnails', sa.Column('data', sqlite.JSON(), nullable=False))
    op.drop_column('card_thumbnails', 'thumbnail_bytes')
    op.drop_column('card_downloads', 'image_bytes')
    # ### end Alembic commands ###