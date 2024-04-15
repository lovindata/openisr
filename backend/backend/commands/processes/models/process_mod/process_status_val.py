from dataclasses import dataclass
from datetime import datetime


@dataclass
class ProcessStatusVal:
    started_at: datetime
    ended: "Successful | Failed | None"

    @dataclass
    class Successful:
        at: datetime

    @dataclass
    class Failed:
        at: datetime
        error: str
        stacktrace: str | None
