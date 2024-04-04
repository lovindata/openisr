from pydantic import BaseModel

from backend.v2.commands.processes.models.process_mod.process_ai_val import ProcessAIVal
from backend.v2.commands.processes.models.process_mod.process_bicubic_val import (
    ProcessBicubicVal,
)
from backend.v2.commands.shared.models.extension_val import ExtensionVal


class ProcessDto(BaseModel):
    extension: ExtensionVal
    scaling: ProcessBicubicVal | ProcessAIVal
