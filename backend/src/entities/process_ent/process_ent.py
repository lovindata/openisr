from entities.process_ent.extension_val import ExtensionVal
from entities.process_ent.image_size_val import ImageSizeVal
from entities.process_ent.status_val import StatusVal


class ProcessEnt:
    def __init__(
        self,
        id: int,
        source_image_id: int,
        extension: ExtensionVal,
        preserve_ratio: bool,
        target: ImageSizeVal,
        enable_ai: bool,
        status: StatusVal,
    ) -> None:
        self.id = id
        self.source_image_id = source_image_id
        self.extension = extension
        self.preserve_ratio = preserve_ratio
        self.target = target
        self.enable_ai = enable_ai
        self.status = status
