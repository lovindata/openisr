from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class StatusDto(BaseModel):
    class SuccessfulDto(BaseModel):
        kind: Literal["successful"]
        at: datetime

    class FailedDto(BaseModel):
        kind: Literal["failed"]
        at: datetime
        error: str

    started_at: datetime
    ended: SuccessfulDto | FailedDto | None