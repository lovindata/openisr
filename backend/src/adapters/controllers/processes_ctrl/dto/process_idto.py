from typing import Literal

from adapters.controllers.shared.dto.image_size_dto import ImageSizeDto
from pydantic import BaseModel


class ProcessIDto(BaseModel):
    extension: Literal["JPEG", "PNG", "WEBP"]
    target: ImageSizeDto
    enable_ai: bool
