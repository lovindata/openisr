from dataclasses import dataclass
from io import BytesIO
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.indexable import index_property
from sqlalchemy.orm import Mapped, Session, mapped_column
from v2.commands.images.models.image_mod import ImageMod
from v2.confs.envs_conf import envs_conf_impl
from v2.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from v2.queries.app.models.card_download_mod import CardDownloadMod


class CardDownloadRow(sqlalchemy_conf_impl.Base):
    __tablename__ = "card_downloads"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    data: Mapped[dict[str, Any]] = mapped_column(type_=JSONB, nullable=False)
    image_id = index_property("data", "image_id")

    def to_mod(self) -> CardDownloadMod:
        return CardDownloadMod.model_validate(self.data)


@dataclass
class CardDownloadsRep:
    envs_conf = envs_conf_impl

    def get(self, session: Session, image_id: int) -> CardDownloadMod:
        return (
            session.query(CardDownloadRow)
            .where(CardDownloadRow.image_id == image_id)
            .one()
            .to_mod()
        )

    def sync(self, session: Session, image: ImageMod) -> None:
        def build_bytes() -> "bytes":
            bytesio = BytesIO()
            image.data.save(bytesio, image.extension().value)
            bytesio.seek(0)
            return bytesio.getvalue()

        def update_row(row: CardDownloadRow, mod: CardDownloadMod) -> None:
            row.data = mod.model_dump()

        def insert_row(mod: CardDownloadMod) -> None:
            row = CardDownloadRow(data=mod.model_dump())
            session.add(row)

        bytes = build_bytes()
        mod = CardDownloadMod(
            bytes=bytes,
            media_type=image.extension().to_media_type(),
            filename=f"{image.name}.{image.extension().to_file_extension()}",
            image_id=image.id,
        )
        row = (
            session.query(CardDownloadRow)
            .where(CardDownloadRow.image_id == image.id)
            .one_or_none()
        )
        update_row(row, mod) if row else insert_row(mod)


card_downloads_rep_impl = CardDownloadsRep()
