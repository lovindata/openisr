from io import BytesIO

from PIL.Image import BICUBIC, Image, open
from PIL_DAT.dat_light import DATLight
from v2.features.processes.models.process_mod import ProcessMod


class ImageProcessingSvc:
    def process_image(self, data: Image, process: ProcessMod) -> Image:
        def resize(data: Image) -> Image:
            if process.enable_ai:
                model = DATLight(4)
                while (
                    data.width < process.target.width
                    or data.height < process.target.height
                ):
                    data = model.upscale(data)
            return data.resize((process.target.width, process.target.height), BICUBIC)

        def change_extension(data: Image) -> Image:
            out_bytes = BytesIO()
            data.save(out_bytes, format=process.extension.value)
            return open(out_bytes)

        data = resize(data)
        data = change_extension(data)
        return data


image_processing_svc_impl = ImageProcessingSvc()
