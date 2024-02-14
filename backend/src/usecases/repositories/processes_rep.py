from abc import ABC, abstractmethod
from typing import Literal

from entities.process_ent import ProcessEnt
from entities.shared.extension_val import ExtensionVal
from sqlalchemy.orm import Session


class ProcessesRep(ABC):
    @abstractmethod
    def create_run(
        self,
        session: Session,
        image_id: int,
        extension: ExtensionVal,
        target_width: int,
        target_height: int,
        enable_ai: bool,
    ) -> ProcessEnt:
        pass

    @abstractmethod
    def update(self, session: Session, ent: ProcessEnt) -> ProcessEnt:
        pass

    @abstractmethod
    def get_latest(self, session: Session, image_id: int) -> ProcessEnt:
        pass
