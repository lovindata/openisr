from typing import Literal

from adapters.controllers.processes_ctrl.dto.status_dto import StatusDto
from adapters.controllers.shared.dto.image_size_dto import ImageSizeDto
from pydantic import BaseModel


class ProcessODto(BaseModel):
    id: int
    target: ImageSizeDto
    status: StatusDto
    extension: Literal["JPEG", "PNG", "WEBP"]
    enable_ai: bool
