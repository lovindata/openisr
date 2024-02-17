from abc import ABC
from datetime import datetime


class StatusVal:
    def __init__(
        self, started_at: datetime, ended: "Successful | Failed | None"
    ) -> None:
        self.started_at = started_at
        self.ended = ended

    class Successful:
        def __init__(self, at: datetime) -> None:
            self.at = at

    class Failed:
        def __init__(self, at: datetime, error: str, stacktrace: str | None) -> None:
            self.at = at
            self.error = error
            self.stacktrace = stacktrace
