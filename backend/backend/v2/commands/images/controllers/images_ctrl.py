from dataclasses import dataclass
from typing import List

from fastapi import APIRouter, FastAPI, UploadFile

from backend.v2 import commands
from backend.v2.commands.images.services.images_svc import images_svc_impl


@dataclass
class ImagesCtrl:
    images_svc = images_svc_impl

    def router(self) -> APIRouter:
        app = FastAPI()

        @app.post(
            tags=[commands.__name__],
            summary="Upload local images",
            path="/command/v1/images/upload-local",
            response_model=None,
            status_code=204,
        )
        def _(files: List[UploadFile]) -> None:
            self.images_svc.upload_images(files)

        @app.delete(
            tags=[commands.__name__],
            summary="Delete image",
            path="/command/v1/images/{id}/delete",
            status_code=204,
        )
        def _(id: int) -> None:
            self.images_svc.delete_image(id)

        return app.router


images_ctrl_impl = ImagesCtrl()
