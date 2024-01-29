from adapters.controllers.images_ctrl.dto.image_size_dto import ImageSizeDto
from pydantic import BaseModel


class ImageODto(BaseModel):
    id: int
    src: str
    name: str
    source: ImageSizeDto
