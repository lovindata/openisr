from typing import Literal

from adapters.controllers.processes_ctrl.dto.image_size_dto import ImageSizeDto
from pydantic import BaseModel


class ProcessIDto(BaseModel):
    extension: Literal["JPEG", "PNG", "WEBP"]
    preserve_ratio: bool
    target: ImageSizeDto
    enable_ai: bool
