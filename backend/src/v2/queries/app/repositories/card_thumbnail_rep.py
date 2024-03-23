from dataclasses import dataclass
from io import BytesIO
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.indexable import index_property
from sqlalchemy.orm import Mapped, Session, mapped_column
from v2.commands.images.models.image_mod import ImageMod
from v2.confs.envs_conf import envs_conf_impl
from v2.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from v2.helpers.pil_utils import build_thumbnail
from v2.queries.app.models.card_thumbnail_mod import CardThumbnailMod


class CardThumbnailRow(sqlalchemy_conf_impl.Base):
    __tablename__ = "cards"

    data: Mapped[dict[str, Any]] = mapped_column(type_=JSONB, nullable=False)
    image_id = index_property("data", "image_id")

    def to_mod(self) -> CardThumbnailMod:
        return CardThumbnailMod.model_validate(self.data)


@dataclass
class CardThumbnailRep:
    envs_conf = envs_conf_impl

    def get(self, session: Session, image_id: int) -> CardThumbnailMod:
        return (
            session.query(CardThumbnailRow)
            .where(CardThumbnailRow.image_id == image_id)
            .one()
            .to_mod()
        )

    def sync(self, session: Session, image: ImageMod) -> None:
        def build_thumbnail_bytes() -> BytesIO:
            thumbnail = build_thumbnail(image.data, 48 * 3)
            bytesio = BytesIO()
            thumbnail.save(bytesio, "WEBP")  # WEBP bytes
            bytesio.seek(0)
            return bytesio

        def update_row(row: CardThumbnailRow, mod: CardThumbnailMod) -> None:
            row.data = mod.model_dump()

        def insert_row(mod: CardThumbnailMod) -> None:
            row = CardThumbnailRow(data=mod.model_dump())
            session.add(row)

        thumbnail_bytes = build_thumbnail_bytes()
        mod = CardThumbnailMod(thumbnail_bytes=thumbnail_bytes, image_id=image.id)
        row = (
            session.query(CardThumbnailRow)
            .where(CardThumbnailRow.image_id == image.id)
            .one_or_none()
        )
        update_row(row, mod) if row else insert_row(mod)


card_thumbnail_rep_impl = CardThumbnailRep()
