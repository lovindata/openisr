from dataclasses import dataclass
from datetime import datetime

from v2.features.processes.models.process_mod.image_size_val import ImageSizeVal
from v2.features.processes.models.process_mod.status_val import StatusVal
from v2.features.shared.models.extension_val import ExtensionVal
from v2.helpers.exception_utils import ServerInternalErrorException


@dataclass
class ProcessMod:
    id: int
    image_id: int | None
    extension: ExtensionVal
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
        return (
            self.terminate_failed("Process timeout.")
            if (self.status.ended is None and diff > timeout_in_seconds)
            else self
        )

    def _raise_when_terminating_already_ended(self) -> None:
        if self.status.ended is not None:
            raise ServerInternalErrorException(
                "Cannot terminate an already ended process."
            )
