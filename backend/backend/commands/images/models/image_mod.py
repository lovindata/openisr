from copy import deepcopy
from dataclasses import dataclass

from PIL.Image import Image

from backend.commands.shared.models.extension_val import ExtensionVal


@dataclass
class ImageMod:
    id: int
    name: str
    data: Image

    def extension(self) -> ExtensionVal:
        return ExtensionVal(self.data.format)

    def update_data(self, data: Image) -> "ImageMod":
        output = deepcopy(self)
        output.data = data
        return output