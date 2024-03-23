from dataclasses import dataclass

from PIL.Image import Image
from sqlalchemy import select
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship
from v2.commands.images.models.image_mod import ImageMod
from v2.commands.processes.repositories.processes_rep import ProcessRow
from v2.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from v2.helpers.exception_utils import BadRequestException
from v2.helpers.pil_utils import extract_bytes, open_from_bytes


class ImageRow(sqlalchemy_conf_impl.Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, index=True)
    name: Mapped[str] = mapped_column(nullable=False)
    data: Mapped[bytes] = mapped_column(nullable=False)

    processes = relationship(ProcessRow)

    def set_all_with(self, mod: ImageMod) -> "ImageRow":
        self.name = mod.name
        self.data = extract_bytes(mod.data)
        return self

    def to_mod(self) -> ImageMod:
        return ImageMod(self.id, self.name, open_from_bytes(self.data))


@dataclass
class ImagesRep:
    def insert(self, session: Session, name: str, data: Image) -> None:
        row = ImageRow(name=name, data=extract_bytes(data))
        session.add(row)
        session.flush()

    def delete(self, session: Session, id: int) -> None:
        row = session.query(ImageRow).where(ImageRow.id == id).one()
        session.delete(row)

    def update(self, session: Session, mod: ImageMod) -> ImageMod:
        return (
            session.query(ImageRow)
            .where(ImageRow.id == id)
            .one()
            .set_all_with(mod)
            .to_mod()
        )

    def get(self, session: Session, id: int) -> ImageMod:
        return session.query(ImageRow).where(ImageRow.id == id).one().to_mod()


images_rep_impl = ImagesRep()
