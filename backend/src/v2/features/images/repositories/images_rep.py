from PIL.Image import Image
from sqlalchemy import select
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship
from v2.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from v2.features.images.models.image_mod import ImageMod
from v2.features.processes.repositories.processes_rep import ProcessRow
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


class ImagesRep:
    def insert(self, session: Session, name: str, data: Image) -> None:
        row = ImageRow(name=name, data=extract_bytes(data))
        session.add(row)
        session.flush()

    def delete(self, session: Session, id: int) -> None:
        row = self._get_or_raise_when_image_not_found(session, id)
        session.delete(row)

    def update(self, session: Session, mod: ImageMod) -> ImageMod:
        return (
            self._get_or_raise_when_image_not_found(session, mod.id)
            .set_all_with(mod)
            .to_mod()
        )

    def get(self, session: Session, id: int) -> ImageMod:
        row = self._get_or_raise_when_image_not_found(session, id)
        return row.to_mod()

    def _get_or_raise_when_image_not_found(self, session: Session, id: int) -> ImageRow:
        stmt = select(ImageRow).where(ImageRow.id == id)
        row = session.scalar(stmt)
        if row:
            return row
        else:
            raise BadRequestException(
                f"The image associated with the provided ID ({id}) does not exist."
            )


images_rep_impl = ImagesRep()
