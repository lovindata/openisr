from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class StatusDto(BaseModel):
    started_at: datetime
    ended: "SuccessfulDto | FailedDto | None"

    class SuccessfulDto(BaseModel):
        kind: Literal["successful"] = "successful"
        at: datetime

    class FailedDto(BaseModel):
        kind: Literal["failed"] = "failed"
        at: datetime
        error: str
