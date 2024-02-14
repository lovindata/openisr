from typing import Literal

from pydantic import BaseModel


class StatusDto(BaseModel):
    class SuccessfulDto(BaseModel):
        kind: Literal["successful"]

    class FailedDto(BaseModel):
        kind: Literal["failed"]
        error: str

    duration: int
    ended: SuccessfulDto | FailedDto | None
