from abc import ABC, abstractmethod
from typing import Literal

from entities.process_ent.process_ent import ProcessEnt
from sqlalchemy.orm import Session


class ProcessesRep(ABC):
    @abstractmethod
    def create_run(
        self,
        session: Session,
        image_id: int,
        extension: Literal["JPEG", "PNG", "WEBP"],
        preserve_ratio: bool,
        target_width: int,
        target_height: int,
        enable_ai: bool,
    ) -> ProcessEnt:
        pass

    @abstractmethod
    def get_latest(self, session: Session, image_id: int) -> ProcessEnt:
        pass
