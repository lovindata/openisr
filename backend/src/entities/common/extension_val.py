from enum import Enum


class ExtensionVal(Enum):
    JPEG = "JPEG"
    PNG = "PNG"
    WEBP = "WEBP"

    def to_file_extension(self) -> str:
        match self:
            case ExtensionVal.JPEG:
                return "jpg"
            case _:
                return self.value.lower()
