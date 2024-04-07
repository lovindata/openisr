from typing import Literal

from pydantic import BaseModel, Field

from backend.v2.commands.processes.models.process_mod.process_ai_val import ProcessAIVal
from backend.v2.commands.processes.models.process_mod.process_bicubic_val import (
    ProcessBicubicVal,
)
from backend.v2.commands.processes.models.process_mod.process_resolution_val import (
    ProcessResolutionVal,
)
from backend.v2.commands.shared.models.extension_val import ExtensionVal


class ProcessDto(BaseModel):
    extension: ExtensionVal
    scaling: "Bicubic | AI" = Field(discriminator="type")

    class Dimension(BaseModel):
        width: int
        height: int

    class Bicubic(BaseModel):
        type: Literal["Bicubic"]
        width: int
        height: int

        def to_val(self) -> ProcessBicubicVal:
            return ProcessBicubicVal(ProcessResolutionVal(self.width, self.height))

    class AI(BaseModel):
        type: Literal["AI"]
        scale: Literal[2, 3, 4]

        def to_val(self) -> ProcessAIVal:
            return ProcessAIVal(self.scale)
