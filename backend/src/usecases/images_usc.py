from io import BytesIO
from typing import List, Tuple

from adapters.repositories.sqlalchemy_images_rep import sqlalchemy_images_rep_impl
from drivers.os_env_loader_driver import os_env_laoder_driver_impl
from drivers.sqlalchemy_db_driver import sqlalchemy_db_driver_impl
from entities.image_ent import ImageEnt
from entities.shared.extension_val import ExtensionVal
from fastapi import UploadFile
from helpers.exception_utils import BadRequestException, ServerInternalErrorException
from helpers.pil_utils import build_thumbnail, open_from_bytes
from usecases.drivers.db_driver import DbDriver
from usecases.drivers.env_loader_driver import EnvLoaderDriver
from usecases.repositories.images_rep import ImagesRep


class ImagesUsc:
    def __init__(
        self,
        env_loader_driver: EnvLoaderDriver,
        db_driver: DbDriver,
        images_rep: ImagesRep,
    ) -> None:
        self.env_loader_driver = env_loader_driver
        self.db_driver = db_driver
        self.images_rep = images_rep

    def list_images(self) -> List[ImageEnt]:
        with self.db_driver.get_session() as session:
            images = self.images_rep.list(session)
        return images

    def delete_image(self, id: int) -> ImageEnt:
        with self.db_driver.get_session() as session:
            image = self.images_rep.delete(session, id)
        return image

    def upload_images(self, files: List[UploadFile]) -> List[ImageEnt]:
        def extract_name_and_data(file: UploadFile) -> Tuple[str, bytes]:
            if file.filename is None:
                raise BadRequestException("Cannot extract a file name.")
            name = "".join(file.filename.split(".")[:-1])
            data = file.file.read()
            if (img_format := open_from_bytes(data).format) not in [
                x.value for x in list(ExtensionVal)
            ]:
                raise BadRequestException(f"Not supported image format: {img_format}.")
            return name, data

        images = [extract_name_and_data(file) for file in files]
        with self.db_driver.get_session() as session:
            images = [
                self.images_rep.insert(session, name, data) for (name, data) in images
            ]
        return images

    def get_image_thumbnail_as_webp(self, id: int) -> BytesIO:
        with self.db_driver.get_session() as session:
            image = self.images_rep.get(session, id).data
        image = build_thumbnail(image, 48 * 3)
        bytesio = BytesIO()
        image.save(bytesio, "WEBP")
        bytesio.seek(0)
        return bytesio

    def download_image(self, id: int) -> Tuple[BytesIO, str, ExtensionVal]:
        def build_image_extension() -> ExtensionVal:
            if image.data.format is None:
                raise ServerInternalErrorException(
                    f"Cannot retrieve image id={id} format: {image.data.format}."
                )
            else:
                try:
                    return ExtensionVal(image.data.format)
                except:
                    raise ServerInternalErrorException(
                        f"Unknown image id={id} format: {image.data.format}."
                    )

        with self.db_driver.get_session() as session:
            image = self.images_rep.get(session, id)
        bytesio = BytesIO()
        image.data.save(bytesio, format=image.data.format)
        bytesio.seek(0)
        extension = build_image_extension()
        return bytesio, image.name, extension

    def build_image_src_thumbnail(self, id: int) -> str:
        src = f"/images/thumbnail/{id}.webp"
        if not self.env_loader_driver.prod_mode:
            src = f"http://localhost:{self.env_loader_driver.api_port}" + src
        return src

    def build_image_src_download(self, id: int) -> str:
        src = f"/images/{id}/download"
        if not self.env_loader_driver.prod_mode:
            src = f"http://localhost:{self.env_loader_driver.api_port}" + src
        return src


images_usc_impl = ImagesUsc(
    os_env_laoder_driver_impl, sqlalchemy_db_driver_impl, sqlalchemy_images_rep_impl
)
