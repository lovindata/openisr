from typing import Literal

from pydantic import BaseModel


class CardMod(BaseModel):
    thumbnail_src: str
    name: str
    source: "Dimension"
    target: "Dimension | None"
    status: "Runnable | Stoppable | Errored | Downloadable"
    error: str | None
    extension: Literal["JPEG", "PNG", "WEBP"]
    preserve_ratio: bool
    enable_ai: bool
    image_id: int

    class Dimension(BaseModel):
        width: int
        height: int

    class Runnable(BaseModel):
        pass

    class Stoppable(BaseModel):
        duration: int

    class Errored(BaseModel):
        duration: int

    class Downloadable(BaseModel):
        pass
