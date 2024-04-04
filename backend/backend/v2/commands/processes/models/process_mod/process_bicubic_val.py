from dataclasses import dataclass

from backend.v2.commands.processes.models.process_mod.process_resolution_val import (
    ProcessResolutionVal,
)


@dataclass
class ProcessBicubicVal:
    target: ProcessResolutionVal
