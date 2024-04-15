from dataclasses import dataclass

from backend.commands.processes.models.process_mod.process_resolution_val import (
    ProcessResolutionVal,
)


@dataclass
class ProcessBicubicVal:
    target: ProcessResolutionVal
