from dataclasses import dataclass


@dataclass
class ImageSizeVal:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
