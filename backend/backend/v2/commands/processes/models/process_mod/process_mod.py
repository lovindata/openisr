from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Tuple

from backend.v2.commands.processes.models.process_mod.image_size_val import ImageSizeVal
from backend.v2.commands.processes.models.process_mod.status_val import StatusVal
from backend.v2.commands.shared.models.extension_val import ExtensionVal
from backend.v2.helpers.exception_utils import ServerInternalErrorException


@dataclass
class ProcessMod:
    id: int
    image_id: int | None
    extension: ExtensionVal
    source: ImageSizeVal
    target: ImageSizeVal
    enable_ai: bool
    status: StatusVal

    def terminate_success(self) -> "ProcessMod":
        self._raise_when_terminating_already_ended()
        output = deepcopy(self)
        output.status.ended = StatusVal.Successful(datetime.now())
        return output

    def terminate_failed(
        self, error: str, stacktrace: str | None = None
    ) -> "ProcessMod":
        self._raise_when_terminating_already_ended()
        output = deepcopy(self)
        output.status.ended = StatusVal.Failed(datetime.now(), error, stacktrace)
        return output

    def terminate_failed_timed_out(self, timeout_in_seconds: int) -> "ProcessMod":
        output = deepcopy(self)
        output.status.ended = StatusVal.Failed(
            output.status.started_at + timedelta(seconds=timeout_in_seconds),
            "Process timeout.",
            None,
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
