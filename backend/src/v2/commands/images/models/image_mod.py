from dataclasses import dataclass

from PIL.Image import Image


@dataclass
class ImageMod:
    id: int
    name: str
    data: Image

    def update_data(self, data: Image) -> "ImageMod":
        self.data = data
        return self
