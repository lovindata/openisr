from abc import ABC, abstractmethod
from datetime import datetime


class StatusVal:
    class Ended(ABC):
        @abstractmethod
        def __init__(self, at: datetime) -> None:
            self.at = at

    class Successful(Ended):
        def __init__(self, at: datetime) -> None:
            self.at = super().__init__(at)

    class Failed(Ended):
        def __init__(self, at: datetime, error: str) -> None:
            self.at = super().__init__(at)
            self.error = error

    def __init__(self, started_at: datetime | None, ended: Ended | None) -> None:
        self.started_at = started_at
        self.ended = ended
