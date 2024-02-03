from io import BytesIO
from typing import List

from adapters.repositories.configs.base import Base
from entities.image_ent import ImageEnt
from helpers.exception_utils import BadRequestException
from helpers.pil_utils import extract_bytes, open_from_bytes
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Mapped, Session, mapped_column
from usecases.repositories.images_rep import ImagesRep


class ImageRow(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(nullable=False)
    data: Mapped[bytes] = mapped_column(nullable=False)

    def set_all_with(self, entity: ImageEnt) -> "ImageRow":
        self.name = entity.name
        self.data = extract_bytes(entity.data)
        return self

    def to_ent(self) -> ImageEnt:
        return ImageEnt(self.id, self.name, open_from_bytes(self.data))


class SqlAlchemyImagesRep(ImagesRep):
    def list(self, session: Session) -> List[ImageEnt]:
        stmt = select(ImageRow)
        rows = session.scalars(stmt).all()
        ents = [x.to_ent() for x in rows]
        return ents

    def insert(self, session: Session, name: str, data: bytes) -> ImageEnt:
        row = ImageRow(name=name, data=data)
        session.add(row)
        session.flush()
        return row.to_ent()

    def get(self, session: Session, id: int) -> ImageEnt:
        row = self._get_or_raise_when_image_not_found(session, id)
        return row.to_ent()

    def delete(self, session: Session, id: int) -> ImageEnt:
        row = self._get_or_raise_when_image_not_found(session, id)
        ent = row.to_ent()
        session.delete(row)
        return ent

    def update(self, session: Session, ent: ImageEnt) -> ImageEnt:
        return (
            self._get_or_raise_when_image_not_found(session, ent.id)
            .set_all_with(ent)
            .to_ent()
        )

    def _get_or_raise_when_image_not_found(self, session: Session, id: int) -> ImageRow:
        stmt = select(ImageRow).where(ImageRow.id == id)
        row = session.scalar(stmt)
        if row:
            return row
        else:
            raise BadRequestException(
                f"The image associated with the provided ID ({id}) does not exist."
            )


sqlalchemy_images_rep_impl = SqlAlchemyImagesRep()
