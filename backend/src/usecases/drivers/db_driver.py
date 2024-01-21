from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generator

from sqlalchemy.orm import Session


class DbDriver(ABC):
    @abstractmethod
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        pass
