from enum import Enum
from typing import Literal


class ExtensionVal(Enum):
    JPEG = "JPEG"
    PNG = "PNG"
    WEBP = "WEBP"

    def to_media_type(self) -> Literal["image/jpeg", "image/png", "image/webp"]:
        match self:
            case ExtensionVal.JPEG:
                return "image/jpeg"
            case ExtensionVal.PNG:
                return "image/png"
            case ExtensionVal.WEBP:
                return "image/webp"

    def to_file_extension(self) -> Literal["jpg", "png", "webp"]:
        match self:
            case ExtensionVal.JPEG:
                return "jpg"
            case ExtensionVal.PNG:
                return "png"
            case ExtensionVal.WEBP:
                return "webp"
