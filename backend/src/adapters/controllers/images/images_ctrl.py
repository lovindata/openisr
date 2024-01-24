from typing import List

from adapters.controllers.images.dto.image_odto import ImageODto
from fastapi import APIRouter, FastAPI, UploadFile
from fastapi.responses import StreamingResponse
from usecases.images_usc import ImageUsc, image_usc_impl


class ImagesCtrl:
    def __init__(self, image_usc: ImageUsc) -> None:
        self.image_usc = image_usc
        self._app = FastAPI()

        @self._app.get(
            summary="List images",
            path="/images",
            response_model=List[ImageODto],
        )
        def _() -> List[ImageODto]:
            images = self.image_usc.list_images()
            return [ImageODto.build_from_ent(image) for image in images]

        @self._app.delete(
            summary="Delete image", path="/images/{id}", response_model=ImageODto
        )
        def _(id: int) -> ImageODto:
            image = self.image_usc.delete_image(id)
            return ImageODto.build_from_ent(image)

        @self._app.post(
            summary="Upload local images",
            path="/images/upload-local",
            response_model=List[ImageODto],
        )
        def _(files: List[UploadFile]) -> List[ImageODto]:
            images = self.image_usc.upload_images(files)
            return [ImageODto.build_from_ent(image) for image in images]

        @self._app.get(
            summary="Image thumbnail (144x144)",
            path="/images/thumbnail/{id}.webp",
            response_class=StreamingResponse,
        )
        def _(id: int) -> StreamingResponse:
            bytesio = self.image_usc.get_image_thumbnail_as_webp(id)
            return StreamingResponse(content=bytesio, media_type="image/webp")

    def router(self) -> APIRouter:
        return self._app.router


images_ctrl_impl = ImagesCtrl(image_usc_impl)