from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta

from backend.commands.processes.models.process_mod.process_ai_val import ProcessAIVal
from backend.commands.processes.models.process_mod.process_bicubic_val import (
    ProcessBicubicVal,
)
from backend.commands.processes.models.process_mod.process_resolution_val import (
    ProcessResolutionVal,
)
from backend.commands.processes.models.process_mod.process_status_val import (
    ProcessStatusVal,
)
from backend.commands.shared.models.extension_val import ExtensionVal
from backend.helpers.exception_utils import ServerInternalErrorException


@dataclass
class ProcessMod:
    id: int
    image_id: int | None
    extension: ExtensionVal
    source: ProcessResolutionVal
    scaling: ProcessBicubicVal | ProcessAIVal
    status: ProcessStatusVal

    def terminate_success(self) -> "ProcessMod":
        self._raise_when_terminating_already_ended()
        output = deepcopy(self)
        output.status.ended = ProcessStatusVal.Successful(at=datetime.now())
        return output

    def terminate_failed(
        self, error: str, stacktrace: str | None = None
    ) -> "ProcessMod":
        self._raise_when_terminating_already_ended()
        output = deepcopy(self)
        output.status.ended = ProcessStatusVal.Failed(
            at=datetime.now(), error=error, stacktrace=stacktrace
        )
        return output

    def terminate_failed_timed_out(self, timeout_in_seconds: int) -> "ProcessMod":
        output = deepcopy(self)
        output.status.ended = ProcessStatusVal.Failed(
            at=output.status.started_at + timedelta(seconds=timeout_in_seconds),
            error="Process timeout.",
            stacktrace=None,
        )
        return output

    def resolve_timeout(self, timeout_in_seconds: int) -> "ProcessMod":
        output = deepcopy(self)
        diff = round((datetime.now() - output.status.started_at).total_seconds())
        if not output.status.ended and diff > timeout_in_seconds:
            output = self.terminate_failed_timed_out(timeout_in_seconds)
        return output

    def _raise_when_terminating_already_ended(self) -> None:
        if self.status.ended:
            raise ServerInternalErrorException(
                "Cannot terminate an already ended process."
            )
