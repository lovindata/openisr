from dataclasses import dataclass
from typing import List

from fastapi import APIRouter, FastAPI, UploadFile

from backend import commands
from backend.commands.images.services.images_svc import images_svc_impl


@dataclass
class ImagesCtrl:
    images_svc = images_svc_impl

    def router(self) -> APIRouter:
        app = FastAPI()

        @app.post(
            tags=[commands.__name__],
            summary="Upload local images",
            path="/commands/v1/images/upload-local",
            response_model=None,
            status_code=204,
        )
        def _(files: List[UploadFile]) -> None:
            self.images_svc.upload_images(files)

        @app.delete(
            tags=[commands.__name__],
            summary="Delete image",
            path="/commands/v1/images/{image_id}/delete",
            status_code=204,
        )
        def _(image_id: int) -> None:
            self.images_svc.delete_image(image_id)

        return app.router


images_ctrl_impl = ImagesCtrl()
