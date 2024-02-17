from datetime import datetime

from entities.process_ent.image_size_val import ImageSizeVal
from entities.process_ent.status_val import StatusVal
from entities.shared.extension_val import ExtensionVal
from helpers.exception_utils import ServerInternalErrorException


class ProcessEnt:
    def __init__(
        self,
        id: int,
        image_id: int | None,
        extension: ExtensionVal,
        target: ImageSizeVal,
        enable_ai: bool,
        status: StatusVal,
    ) -> None:
        self.id = id
        self.image_id = image_id
        self.extension = extension
        self.target = target
        self.enable_ai = enable_ai
        self.status = status

    def terminate_success(self) -> "ProcessEnt":
        self._raise_when_terminating_already_ended()
        self.status.ended = StatusVal.Successful(datetime.now())
        return self

    def terminate_failed(
        self, error: str, stacktrace: str | None = None
    ) -> "ProcessEnt":
        self._raise_when_terminating_already_ended()
        self.status.ended = StatusVal.Failed(datetime.now(), error, stacktrace)
        return self

    def resolve_timeout(self, timeout_in_seconds: int) -> "ProcessEnt":
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
