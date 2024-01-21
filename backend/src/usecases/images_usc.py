from io import BytesIO
from typing import List, Tuple

from adapters.repositories.sqlalchemy_images_rep import sqlalchemy_images_rep_impl
from drivers.sqlalchemy_db_driver import sqlalchemy_db_driver_impl
from entities.image_ent import ImageEnt
from fastapi import UploadFile
from helpers.exception_utils import BadRequestException
from helpers.pil_utils import build_thumbnail
from usecases.drivers.db_driver import DbDriver
from usecases.repositories.images_rep import ImagesRep


class ImageUsc:
    def __init__(self, db_driver: DbDriver, images_rep: ImagesRep) -> None:
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


image_usc_impl = ImageUsc(sqlalchemy_db_driver_impl, sqlalchemy_images_rep_impl)
