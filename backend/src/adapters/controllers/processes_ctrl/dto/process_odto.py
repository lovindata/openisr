from typing import Literal

from adapters.controllers.processes_ctrl.dto.image_size_dto import ImageSizeDto
from adapters.controllers.processes_ctrl.dto.status_dto import StatusDto
from pydantic import BaseModel


class ProcessODto(BaseModel):
    id: int
    target: ImageSizeDto
    status: StatusDto
    extension: Literal["JPEG", "PNG", "WEBP"]
    preserve_ratio: bool
    enable_ai: bool
