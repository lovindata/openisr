from dataclasses import dataclass
from typing import List, Tuple

from fastapi import UploadFile
from PIL.Image import Image

from backend.v2.commands.images.repositories.images_rep import images_rep_impl
from backend.v2.commands.shared.models.extension_val import ExtensionVal
from backend.v2.confs.envs_conf import envs_conf_impl
from backend.v2.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from backend.v2.helpers.exception_utils import BadRequestException
from backend.v2.helpers.pil_utils import open_from_bytes
from backend.v2.queries.app.repositories.card_downloads_rep import (
    card_downloads_rep_impl,
)
from backend.v2.queries.app.repositories.card_thumbnails_rep import (
    card_thumbnails_rep_impl,
)
from backend.v2.queries.app.repositories.cards_rep import cards_rep_impl


@dataclass
class ImagesSvc:
    envs_conf = envs_conf_impl
    sqlalchemy_conf = sqlalchemy_conf_impl
    images_rep = images_rep_impl
    cards_rep = cards_rep_impl
    card_thumbnails_rep = card_thumbnails_rep_impl
    card_download_rep = card_downloads_rep_impl

    def delete_image(self, image_id: int) -> None:
        with self.sqlalchemy_conf.get_session() as session:
            self.images_rep.delete(session, image_id)
            self.cards_rep.clean_sync(session, image_id)
            self.card_thumbnails_rep.clean_sync(session, image_id)
            self.card_download_rep.clean_sync(session, image_id)

    def upload_images(self, files: List[UploadFile]) -> None:
        def extract_name_and_data(
            file: UploadFile,
        ) -> Tuple[str, Image]:
            if file.filename is None:
                raise BadRequestException("Unable to extract the file name.")
            name = "".join(file.filename.split(".")[:-1])
            data = open_from_bytes(file.file.read())
            if (img_format := data.format) not in [x.value for x in list(ExtensionVal)]:
                raise BadRequestException(f"Not supported image format: {img_format}.")
            return name, data

        def upload_transactionally(images: List[Tuple[str, Image]]) -> None:
            with self.sqlalchemy_conf.get_session() as session:
                for name, data in images:
                    image = self.images_rep.insert(session, name, data)
                    self.cards_rep.sync(session, image, None)
                    self.card_thumbnails_rep.sync(session, image)

        images = [extract_name_and_data(file) for file in files]
        upload_transactionally(images)


images_svc_impl = ImagesSvc()
