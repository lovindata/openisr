from dataclasses import dataclass
from typing import List

from fastapi import APIRouter, FastAPI, UploadFile
from v2.commands.images.services.images_svc import images_svc_impl


@dataclass
class ImagesCmd:
    images_svc = images_svc_impl

    def router(self) -> APIRouter:
        app = FastAPI()

        @app.delete(summary="Delete image", path="/images/{id}", response_model=None)
        def _(id: int) -> None:
            self.images_svc.delete_image(id)

        @app.post(
            summary="Upload local images",
            path="/images/upload-local",
            response_model=None,
        )
        def _(files: List[UploadFile]) -> None:
            self.images_svc.upload_images(files)

        return app.router


images_cmd_impl = ImagesCmd()
