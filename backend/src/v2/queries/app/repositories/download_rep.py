from dataclasses import dataclass
from io import BytesIO
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.indexable import index_property
from sqlalchemy.orm import Mapped, Session, mapped_column
from v2.commands.images.models.image_mod import ImageMod
from v2.commands.shared.models.extension_val import ExtensionVal
from v2.confs.envs_conf import envs_conf_impl
from v2.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from v2.queries.app.models.download_mod import DownloadMod


class DownloadRow(sqlalchemy_conf_impl.Base):
    __tablename__ = "cards"

    data: Mapped[dict[str, Any]] = mapped_column(type_=JSONB, nullable=False)
    image_id = index_property("data", "image_id")

    def to_mod(self) -> DownloadMod:
        return DownloadMod.model_validate(self.data)


@dataclass
class DownloadRep:
    envs_conf = envs_conf_impl

    def get(self, session: Session, image_id: int) -> DownloadMod:
        return (
            session.query(DownloadRow)
            .where(DownloadRow.image_id == image_id)
            .one()
            .to_mod()
        )

    def sync(self, session: Session, image: ImageMod) -> None:
        def build_bytes() -> BytesIO:
            bytesio = BytesIO()
            image.data.save(bytesio, ExtensionVal(image.data.format).value)
            bytesio.seek(0)
            return bytesio

        def update_row(row: DownloadRow, mod: DownloadMod) -> None:
            row.data = mod.model_dump()

        def insert_row(mod: DownloadMod) -> None:
            row = DownloadRow(data=mod.model_dump())
            session.add(row)

        bytes = build_bytes()
        mod = DownloadMod(
            bytes=bytes,
            media_type=ExtensionVal(image.data.format).to_media_type(),
            filename=f"{image.name}.{ExtensionVal(image.data.format).to_file_extension()}",
            image_id=image.id,
        )
        row = (
            session.query(DownloadRow)
            .where(DownloadRow.image_id == image.id)
            .one_or_none()
        )
        update_row(row, mod) if row else insert_row(mod)


download_rep_impl = DownloadRep()
