from entities.common.extension_val import ExtensionVal
from PIL.Image import Image


class ImageEnt:
    def __init__(self, id: int, name: str, data: Image) -> None:
        self.id = id
        self.name = name
        self.data = data

    def update_data(self, data: Image) -> "ImageEnt":
        self.data = data
        return self
