from dataclasses import dataclass
from typing import List

from PIL.Image import Image
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship

from backend.v2.commands.images.models.image_mod import ImageMod
from backend.v2.commands.processes.repositories.processes_rep.processes_rep import (
    ProcessRow,
)
from backend.v2.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from backend.v2.helpers.exception_utils import BadRequestException
from backend.v2.helpers.pil_utils import extract_bytes, open_from_bytes


class ImageRow(sqlalchemy_conf_impl.Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, index=True)
    name: Mapped[str]
    data: Mapped[bytes]

    processes: Mapped[List[ProcessRow]] = (
        relationship()
    )  # Cascade effect only applies with 'session.delete': https://github.com/sqlalchemy/sqlalchemy/discussions/7974

    def update_with(self, mod: ImageMod) -> "ImageRow":
        self.name = mod.name
        self.data = extract_bytes(mod.data)
        return self

    def to_mod(self) -> ImageMod:
        return ImageMod(self.id, self.name, open_from_bytes(self.data))


@dataclass
class ImagesRep:
    def insert(self, session: Session, name: str, data: Image) -> ImageMod:
        row = ImageRow(name=name, data=extract_bytes(data))
        session.add(row)
        session.flush()
        return row.to_mod()

    def delete(self, session: Session, image_id: int) -> None:
        row = session.query(ImageRow).where(ImageRow.id == image_id).one_or_none()
        if row:
            session.delete(row)

    def update(self, session: Session, mod: ImageMod) -> ImageMod:
        mod = (
            session.query(ImageRow)
            .where(ImageRow.id == mod.id)
            .one()
            .update_with(mod)
            .to_mod()
        )
        return mod

    def get_or_raise(self, session: Session, image_id: int) -> ImageMod:
        row = session.query(ImageRow).where(ImageRow.id == image_id).one_or_none()
        if row is None:
            raise BadRequestException(f"No image with ID={image_id}.")
        return row.to_mod()

    def list(self, session: Session, ids: List[int]) -> List[ImageMod]:
        rows = session.query(ImageRow).where(ImageRow.id.in_(ids)).all()
        mods = [row.to_mod() for row in rows]
        return mods


images_rep_impl = ImagesRep()
