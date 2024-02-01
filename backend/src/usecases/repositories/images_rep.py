from abc import ABC, abstractmethod
from typing import List

from entities.image_ent import ImageEnt
from sqlalchemy.orm import Session


class ImagesRep(ABC):
    @abstractmethod
    def list(self, session: Session) -> List[ImageEnt]:
        pass

    @abstractmethod
    def insert(self, session: Session, name: str, data: bytes) -> ImageEnt:
        pass

    @abstractmethod
    def get(self, session: Session, id: int) -> ImageEnt:
        pass

    @abstractmethod
    def delete(self, session: Session, id: int) -> ImageEnt:
        pass

    @abstractmethod
    def update(self, session: Session, ent: ImageEnt) -> ImageEnt:
        pass
