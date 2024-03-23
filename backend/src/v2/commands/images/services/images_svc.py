from dataclasses import dataclass
from typing import List, Tuple

from fastapi import UploadFile
from PIL.Image import Image
from v2.commands.images.repositories.images_rep import images_rep_impl
from v2.commands.shared.models.extension_val import ExtensionVal
from v2.confs.envs_conf import envs_conf_impl
from v2.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from v2.helpers.exception_utils import BadRequestException
from v2.helpers.pil_utils import open_from_bytes


@dataclass
class ImagesSvc:
    envs_conf = envs_conf_impl
    sqlalchemy_conf = sqlalchemy_conf_impl
    images_rep = images_rep_impl

    def delete_image(self, id: int) -> None:
        with self.sqlalchemy_conf.get_session() as session:
            self.images_rep.delete(session, id)

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
                    self.images_rep.insert(session, name, data)

        images = [extract_name_and_data(file) for file in files]
        upload_transactionally(images)


images_svc_impl = ImagesSvc()
