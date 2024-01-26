from io import BytesIO
from typing import List, Tuple

from adapters.repositories.sqlalchemy_images_rep import sqlalchemy_images_rep_impl
from drivers.os_env_loader_driver import os_env_laoder_driver_impl
from drivers.sqlalchemy_db_driver import sqlalchemy_db_driver_impl
from entities.image_ent import ImageEnt
from fastapi import UploadFile
from helpers.exception_utils import BadRequestException
from helpers.pil_utils import build_thumbnail
from usecases.drivers.db_driver import DbDriver
from usecases.drivers.env_loader_driver import EnvLoaderDriver
from usecases.repositories.images_rep import ImagesRep


class ImageUsc:
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
            name = file.filename
            data = file.file.read()
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

    def build_image_src(self, id: int) -> str:
        src = f"/images/thumbnail/{id}.webp"
        if not self.env_loader_driver.prod_mode:
            src = f"http://localhost:{self.env_loader_driver.api_port}" + src
        return src


image_usc_impl = ImageUsc(
    os_env_laoder_driver_impl, sqlalchemy_db_driver_impl, sqlalchemy_images_rep_impl
)
