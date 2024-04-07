"""empty message

Revision ID: 3fbd179e5f1d
Revises: 
Create Date: 2024-04-07 13:10:06.177089

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3fbd179e5f1d'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('card_downloads',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('data', sa.JSON(), nullable=False),
    sa.Column('image_bytes', sa.LargeBinary(), nullable=False),
    sa.Column('image_id', sa.Integer(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_card_downloads_image_id'), 'card_downloads', ['image_id'], unique=True)
    op.create_table('card_thumbnails',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('thumbnail_bytes', sa.LargeBinary(), nullable=False),
    sa.Column('image_id', sa.Integer(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_card_thumbnails_image_id'), 'card_thumbnails', ['image_id'], unique=True)
    op.create_table('cards',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('data', sa.JSON(), nullable=False),
    sa.Column('image_id', sa.Integer(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_cards_image_id'), 'cards', ['image_id'], unique=True)
    op.create_table('images',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('data', sa.LargeBinary(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_images_id'), 'images', ['id'], unique=False)
    op.create_table('processes',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('image_id', sa.Integer(), nullable=True),
    sa.Column('extension', sa.Enum('JPEG', 'PNG', 'WEBP', name='extensionval'), nullable=False),
    sa.Column('source_width', sa.Integer(), nullable=False),
    sa.Column('source_height', sa.Integer(), nullable=False),
    sa.Column('scaling_bicubic_target_width', sa.Integer(), nullable=True),
    sa.Column('scaling_bicubic_target_height', sa.Integer(), nullable=True),
    sa.Column('scaling_ai_scale', sa.Integer(), nullable=True),
    sa.Column('status_started_at', sa.DateTime(), nullable=False),
    sa.Column('status_ended_successful_at', sa.DateTime(), nullable=True),
    sa.Column('status_ended_failed_at', sa.DateTime(), nullable=True),
    sa.Column('status_ended_failed_error', sa.String(), nullable=True),
    sa.Column('status_ended_failed_stacktrace', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['image_id'], ['images.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_processes_id'), 'processes', ['id'], unique=False)
    op.create_index(op.f('ix_processes_image_id'), 'processes', ['image_id'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_processes_image_id'), table_name='processes')
    op.drop_index(op.f('ix_processes_id'), table_name='processes')
    op.drop_table('processes')
    op.drop_index(op.f('ix_images_id'), table_name='images')
    op.drop_table('images')
    op.drop_index(op.f('ix_cards_image_id'), table_name='cards')
    op.drop_table('cards')
    op.drop_index(op.f('ix_card_thumbnails_image_id'), table_name='card_thumbnails')
    op.drop_table('card_thumbnails')
    op.drop_index(op.f('ix_card_downloads_image_id'), table_name='card_downloads')
    op.drop_table('card_downloads')
    # ### end Alembic commands ###