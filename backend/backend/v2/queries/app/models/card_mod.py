from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class CardMod(BaseModel):
    thumbnail_src: str
    name: str
    source: "Dimension"
    target: "Dimension | None"
    status: "Runnable | Stoppable | Errored | Downloadable" = Field(
        discriminator="type"
    )
    extension: Literal["JPEG", "PNG", "WEBP"]
    preserve_ratio: bool
    enable_ai: bool
    image_id: int

    class Dimension(BaseModel):
        width: int
        height: int

    class Runnable(BaseModel):
        type: Literal["Runnable"]

    class Stoppable(BaseModel):
        type: Literal["Stoppable"]

        @computed_field
        @property
        def duration(self) -> int:
            return round((datetime.now() - self.started_at).total_seconds())

        started_at: datetime

    class Errored(BaseModel):
        type: Literal["Errored"]
        duration: int
        error: str

    class Downloadable(BaseModel):
        type: Literal["Downloadable"]
        image_src: str
