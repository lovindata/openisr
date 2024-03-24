from typing import Literal

from pydantic import BaseModel
from v2.commands.processes.controllers.processes_ctrl.image_size_dto import ImageSizeDto


class ProcessDto(BaseModel):
    extension: Literal["JPEG", "PNG", "WEBP"]
    target: ImageSizeDto
    enable_ai: bool
