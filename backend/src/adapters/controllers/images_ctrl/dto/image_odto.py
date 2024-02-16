from typing import Literal

from adapters.controllers.images_ctrl.dto.src_dto import SrcDto
from adapters.controllers.shared.dto.image_size_dto import ImageSizeDto
from pydantic import BaseModel


class ImageODto(BaseModel):
    id: int
    src: SrcDto
    name: str
    extension: Literal["JPEG", "PNG", "WEBP"]
    source: ImageSizeDto
