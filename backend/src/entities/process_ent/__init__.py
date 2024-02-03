from datetime import datetime

from entities.common.extension_val import ExtensionVal
from entities.process_ent.image_size_val import ImageSizeVal
from entities.process_ent.status_val import StatusVal


class ProcessEnt:
    def __init__(
        self,
        id: int,
        image_id: int,
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
        self.status.ended = StatusVal.Successful(datetime.now())
        return self

    def terminate_failed(self, stacktrace_error: str) -> "ProcessEnt":
        self.status.ended = StatusVal.Failed(datetime.now(), stacktrace_error)
        return self
