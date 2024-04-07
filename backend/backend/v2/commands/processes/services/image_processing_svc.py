from dataclasses import dataclass
from io import BytesIO

from PIL.Image import Image, Resampling, open
from PIL_DAT.Image import upscale

from backend.v2.commands.processes.models.process_mod.process_ai_val import ProcessAIVal
from backend.v2.commands.processes.models.process_mod.process_bicubic_val import (
    ProcessBicubicVal,
)
from backend.v2.commands.processes.models.process_mod.process_mod import ProcessMod
from backend.v2.commands.shared.models.extension_val import ExtensionVal


@dataclass
class ImageProcessingSvc:
    def process_image(self, data: Image, process: ProcessMod) -> Image:
        def resize() -> Image:
            match process.scaling:
                case ProcessBicubicVal(target=target):
                    return data.resize(
                        (target.width, target.height),
                        Resampling.BICUBIC,
                    )
                case ProcessAIVal(scale=scale):
                    return upscale(data, scale)

        def change_extension(data: Image) -> Image:
            out_bytes = BytesIO()
            if process.extension == ExtensionVal.JPEG:
                data = data.convert("RGB")
            data.save(out_bytes, format=process.extension.value)
            return open(out_bytes)

        data = resize()
        data = change_extension(data)
        return data


image_processing_svc_impl = ImageProcessingSvc()
