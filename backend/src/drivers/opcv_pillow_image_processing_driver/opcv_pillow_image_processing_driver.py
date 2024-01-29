from io import BytesIO

from cv2.dnn_superres import DnnSuperResImpl
from entities.process_ent.extension_val import ExtensionVal
from entities.process_ent.image_size_val import ImageSizeVal
from entities.process_ent.process_ent import ProcessEnt
from helpers.exception_utils import ServerInternalErrorException
from loguru import logger
from PIL.Image import LANCZOS, Image, open
from usecases.drivers.image_processing_driver import ImageProcessingDriver


class OpcvPillowImageProcessingDriver(ImageProcessingDriver):
    def __init__(self) -> None:
        def load_edsr() -> DnnSuperResImpl:
            logger.info("Loading EDSR model into RAM.")
            edsr = DnnSuperResImpl.create()
            edsr.readModel(
                "./src/drivers/opcv_pillow_image_processing_driver/EDSR_x4.pb"
            )
            edsr.setModel("edsr", 4)
            return edsr

        self._edsr = load_edsr()

    def process_image(self, data: Image, process: ProcessEnt) -> Image | Exception:
        def resize(data: Image, target: ImageSizeVal, enable_ai: bool) -> Image:
            if enable_ai:
                raise ServerInternalErrorException("Enable AI not implemented yet.")
            return data.resize((target.width, target.height), LANCZOS)

        def change_extension(data: Image, extension: ExtensionVal) -> Image:
            out_bytes = BytesIO()
            data.save(out_bytes, format=extension.value)
            return open(out_bytes)

        data = resize(data, process.target, process.enable_ai)
        data = change_extension(data, process.extension)
        return data


opcv_pillow_image_processing_driver_impl = OpcvPillowImageProcessingDriver()
