from io import BytesIO
from typing import Tuple

import numpy as np
from cv2.dnn_superres import DnnSuperResImpl
from entities.process_ent import ProcessEnt
from entities.process_ent.image_size_val import ImageSizeVal
from entities.shared.extension_val import ExtensionVal
from PIL import Image as PILImg
from PIL.Image import LANCZOS, Image, open
from usecases.drivers.image_processing_driver import ImageProcessingDriver


class OpcvPillowImageProcessingDriver(ImageProcessingDriver):
    def process_image(self, data: Image, process: ProcessEnt) -> Image:
        def resize(data: Image, target: ImageSizeVal, enable_ai: bool) -> Image:
            if enable_ai:
                data = self._upsample_with_edsr_until_enough(data, process)
            return data.resize((target.width, target.height), LANCZOS)

        def change_extension(data: Image, extension: ExtensionVal) -> Image:
            out_bytes = BytesIO()
            data.save(out_bytes, format=extension.value)
            return open(out_bytes)

        data = resize(data, process.target, process.enable_ai)
        data = change_extension(data, process.extension)
        return data

    def _upsample_with_edsr_until_enough(
        self, data: Image, process: ProcessEnt
    ) -> Image:
        def load_edsr() -> DnnSuperResImpl:
            edsr = DnnSuperResImpl.create()
            edsr.readModel(
                "./src/drivers/opcv_pillow_image_processing_driver/EDSR_x4.pb"
            )
            edsr.setModel("edsr", 4)
            return edsr

        def extract_opac_and_alpha(data: Image) -> Tuple[Image, Image]:
            data = data.convert("RGBA")
            opac_img = PILImg.new("RGB", data.size)
            alpha_img = PILImg.new("RGB", data.size)
            for x in range(data.width):
                for y in range(data.height):
                    r, g, b, a = data.getpixel((x, y))
                    (
                        opac_img.putpixel((x, y), (255, 255, 255))
                        if a == 0
                        else opac_img.putpixel((x, y), (r, g, b))
                    )
                    alpha_img.putpixel((x, y), (a, a, a))
            return opac_img, alpha_img

        def upsample_opac_and_alpha(
            edsr: DnnSuperResImpl,
            opac_img: Image,
            alpha_img: Image,
            process: ProcessEnt,
        ) -> Tuple[Image, Image]:
            opac_img_ndarray = np.asarray(opac_img)
            alpha_img_ndarray = np.asarray(alpha_img)
            while (
                opac_img_ndarray.shape[0] < process.target.width
                or opac_img_ndarray.shape[1] < process.target.height
            ):
                opac_img_ndarray = edsr.upsample(opac_img_ndarray)
                alpha_img_ndarray = edsr.upsample(alpha_img_ndarray)
            opac_img = PILImg.fromarray(opac_img_ndarray)
            alpha_img = PILImg.fromarray(alpha_img_ndarray)
            return opac_img, alpha_img

        def merge_opac_and_alpha(opac_img: Image, alpha_img: Image) -> Image:
            alpha_img = alpha_img.convert("L")
            merged_img = PILImg.new("RGBA", opac_img.size)
            for x in range(opac_img.width):
                for y in range(opac_img.height):
                    r, g, b = opac_img.getpixel((x, y))
                    a = alpha_img.getpixel((x, y))
                    merged_img.putpixel((x, y), (r, g, b, a))
            return merged_img

        if data.width < process.target.width or data.height < process.target.height:
            edsr = load_edsr()
            opac_img, alpha_img = extract_opac_and_alpha(data)
            opac_img, alpha_img = upsample_opac_and_alpha(
                edsr, opac_img, alpha_img, process
            )
            data = merge_opac_and_alpha(opac_img, alpha_img)
        return data


opcv_pillow_image_processing_driver_impl = OpcvPillowImageProcessingDriver()
