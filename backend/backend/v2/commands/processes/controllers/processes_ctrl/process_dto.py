from pydantic import BaseModel

from backend.v2.commands.processes.controllers.processes_ctrl.image_size_dto import (
    ImageSizeDto,
)
from backend.v2.commands.shared.models.extension_val import ExtensionVal


class ProcessDto(BaseModel):
    extension: ExtensionVal
    target: ImageSizeDto
    enable_ai: bool
