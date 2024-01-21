from PIL.Image import Image


class ImageEnt:
    def __init__(self, id: int, name: str, data: Image) -> None:
        self.id = id
        self.name = name
        self.data = data
