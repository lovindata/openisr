from dataclasses import dataclass
from typing import Tuple

from backend.commands.processes.models.process_mod.process_resolution_val import (
    ProcessResolutionVal,
)


@dataclass
class ProcessSourceComp:
    value: ProcessResolutionVal

    @classmethod
    def _generate(cls, source_width: int, source_height: int) -> "ProcessSourceComp":
        return ProcessSourceComp(ProcessResolutionVal(source_width, source_height))

    def __composite_values__(self) -> Tuple[int, int]:
        return self.value.width, self.value.height
