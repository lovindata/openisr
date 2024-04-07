from dataclasses import dataclass
from io import BytesIO

from sqlalchemy.orm import Mapped, Session, mapped_column

from backend.commands.images.models.image_mod import ImageMod
from backend.confs.envs_conf import envs_conf_impl
from backend.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from backend.helpers.pil_utils import build_thumbnail
from backend.queries.app.models.card_thumbnail_mod import CardThumbnailMod


class CardThumbnailRow(sqlalchemy_conf_impl.Base):
    __tablename__ = "card_thumbnails"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(unique=True, index=True)
    thumbnail_bytes: Mapped[bytes]

    @classmethod
    def insert_with(cls, session: Session, mod: CardThumbnailMod) -> None:
        row = CardThumbnailRow(
            thumbnail_bytes=mod.thumbnail_bytes, image_id=mod.image_id
        )
        session.add(row)

    def update_with(self, mod: CardThumbnailMod) -> None:
        self.thumbnail_bytes = mod.thumbnail_bytes
        self.image_id = mod.image_id

    def to_mod(self) -> CardThumbnailMod:
        return CardThumbnailMod(
            thumbnail_bytes=self.thumbnail_bytes, image_id=self.image_id
        )


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
        mod = self._build_mod(image)
        row = (
            session.query(CardThumbnailRow)
            .where(CardThumbnailRow.image_id == image.id)
            .one_or_none()
        )
        row.update_with(mod) if row else CardThumbnailRow.insert_with(session, mod)

    def clean_sync(self, session: Session, image_id: int) -> None:
        row = (
            session.query(CardThumbnailRow)
            .where(CardThumbnailRow.image_id == image_id)
            .one_or_none()
        )
        if row:
            session.delete(row)

    def _build_mod(self, image: ImageMod) -> CardThumbnailMod:
        def build_thumbnail_bytes() -> bytes:
            thumbnail = build_thumbnail(image.data, 48 * 3)
            bytesio = BytesIO()
            thumbnail.save(bytesio, "WEBP")  # WEBP bytes
            bytesio.seek(0)
            return bytesio.getvalue()

        thumbnail_bytes = build_thumbnail_bytes()
        mod = CardThumbnailMod(thumbnail_bytes=thumbnail_bytes, image_id=image.id)
        return mod


card_thumbnails_rep_impl = CardThumbnailsRep()
