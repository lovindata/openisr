from dataclasses import dataclass
from datetime import datetime, timedelta

from backend.v2.commands.processes.models.process_mod.image_size_val import ImageSizeVal
from backend.v2.commands.processes.models.process_mod.status_val import StatusVal
from backend.v2.commands.shared.models.extension_val import ExtensionVal
from backend.v2.helpers.exception_utils import ServerInternalErrorException


@dataclass
class ProcessMod:
    id: int
    image_id: int
    extension: ExtensionVal
    source: ImageSizeVal
    target: ImageSizeVal
    enable_ai: bool
    status: StatusVal

    def terminate_success(self) -> "ProcessMod":
        self._raise_when_terminating_already_ended()
        self.status.ended = StatusVal.Successful(datetime.now())
        return self

    def terminate_failed(
        self, error: str, stacktrace: str | None = None
    ) -> "ProcessMod":
        self._raise_when_terminating_already_ended()
        self.status.ended = StatusVal.Failed(datetime.now(), error, stacktrace)
        return self

    def resolve_timeout(self, timeout_in_seconds: int) -> "ProcessMod":
        diff = round((datetime.now() - self.status.started_at).total_seconds())
        if not self.status.ended and diff > timeout_in_seconds:
            self.status.ended = StatusVal.Failed(
                self.status.started_at + timedelta(seconds=timeout_in_seconds),
                "Process timeout.",
                None,
            )
        return self

    def _raise_when_terminating_already_ended(self) -> None:
        if self.status.ended:
            raise ServerInternalErrorException(
                "Cannot terminate an already ended process."
            )
