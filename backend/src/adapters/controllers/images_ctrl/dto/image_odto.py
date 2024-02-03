from typing import Literal

from adapters.controllers.common.dto.image_size_dto import ImageSizeDto
from pydantic import BaseModel


class ImageODto(BaseModel):
    id: int
    src: str
    name: str
    extension: Literal["JPEG", "PNG", "WEBP"]
    source: ImageSizeDto
