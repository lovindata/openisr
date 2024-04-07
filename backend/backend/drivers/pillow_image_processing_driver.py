from io import BytesIO

from entities.process_ent import ProcessEnt
from entities.process_ent.image_size_val import ImageSizeVal
from entities.shared.extension_val import ExtensionVal
from PIL.Image import Image, Resampling, open
from PIL_DAT.dat_light import DATLight
from usecases.drivers.image_processing_driver import ImageProcessingDriver


class PillowImageProcessingDriver(ImageProcessingDriver):
    def process_image(self, data: Image, process: ProcessEnt) -> Image:
        def resize(data: Image, target: ImageSizeVal, enable_ai: bool) -> Image:
            if enable_ai:
                model = DATLight(4)
                while (
                    data.width < process.target.width
                    or data.height < process.target.height
                ):
                    data = model.upscale(data)
            return data.resize((target.width, target.height), Resampling.BICUBIC)

        def change_extension(data: Image, extension: ExtensionVal) -> Image:
            out_bytes = BytesIO()
            data.save(out_bytes, format=extension.value)
            return open(out_bytes)

        data = resize(data, process.target, process.enable_ai)
        data = change_extension(data, process.extension)
        return data


pillow_image_processing_driver_impl = PillowImageProcessingDriver()
