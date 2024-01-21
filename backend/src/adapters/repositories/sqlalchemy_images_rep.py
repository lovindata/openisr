from io import BytesIO
from typing import List

from adapters.repositories.configs.base import Base
from entities.image_ent import ImageEnt
from helpers.exception_utils import BadRequestException
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Mapped, Session, mapped_column
from usecases.repositories.images_rep import ImagesRep


class ImageRow(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(nullable=False)
    data: Mapped[bytes] = mapped_column(nullable=False)

    @classmethod
    def from_ent(cls, entity: ImageEnt) -> "ImageRow":
        data_bytes = BytesIO()
        entity.data.save(data_bytes)
        data_bytes = data_bytes.getvalue()
        return ImageRow(name=entity.name, data=data_bytes)

    def to_ent(self) -> ImageEnt:
        data_image = BytesIO(self.data)
        data_image = Image.open(data_image)
        return ImageEnt(self.id, self.name, data_image)


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
