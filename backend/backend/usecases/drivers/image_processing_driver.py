from abc import ABC, abstractmethod

from entities.process_ent import ProcessEnt
from PIL.Image import Image


class ImageProcessingDriver(ABC):
    @abstractmethod
    def process_image(self, data: Image, process: ProcessEnt) -> Image:
        pass
