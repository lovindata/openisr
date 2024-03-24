from dataclasses import dataclass
from io import BytesIO
from typing import Any

from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, Session, mapped_column
from v2.commands.images.models.image_mod import ImageMod
from v2.confs.envs_conf import envs_conf_impl
from v2.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from v2.helpers.pil_utils import build_thumbnail
from v2.queries.app.models.card_thumbnail_mod import CardThumbnailMod


class CardThumbnailRow(sqlalchemy_conf_impl.Base):
    __tablename__ = "card_thumbnails"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    data: Mapped[dict[str, Any]] = mapped_column(type_=JSON, nullable=False)
    image_id: Mapped[int] = mapped_column(unique=True, index=True)

    @classmethod
    def insert_with(cls, session: Session, mod: CardThumbnailMod) -> None:
        row = CardThumbnailRow(data=mod.model_dump(), image_id=mod.image_id)
        session.add(row)

    def update_with(self, mod: CardThumbnailMod) -> None:
        self.data = mod.model_dump()
        self.image_id = mod.image_id

    def to_mod(self) -> CardThumbnailMod:
        return CardThumbnailMod.model_validate(self.data)


@dataclass
class CardThumbnailsRep:
    envs_conf = envs_conf_impl

    def get(self, session: Session, image_id: int) -> CardThumbnailMod:
        return (
            session.query(CardThumbnailRow)
            .where(CardThumbnailRow.image_id == image_id)
            .one()
            .to_mod()
        )

    def sync(self, session: Session, image: ImageMod) -> None:
        def build_thumbnail_bytes() -> bytes:
            thumbnail = build_thumbnail(image.data, 48 * 3)
            bytesio = BytesIO()
            thumbnail.save(bytesio, "WEBP")  # WEBP bytes
            bytesio.seek(0)
            return bytesio.getvalue()

        thumbnail_bytes = build_thumbnail_bytes()
        mod = CardThumbnailMod(thumbnail_bytes=thumbnail_bytes, image_id=image.id)
        row = (
            session.query(CardThumbnailRow)
            .where(CardThumbnailRow.image_id == image.id)
            .one_or_none()
        )
        row.update_with(mod) if row else CardThumbnailRow.insert_with(session, mod)

    def clean_sync(self, session: Session, image_id: int) -> None:
        session.query(CardThumbnailRow).where(
            CardThumbnailRow.image_id == image_id
        ).delete()


card_thumbnails_rep_impl = CardThumbnailsRep()
