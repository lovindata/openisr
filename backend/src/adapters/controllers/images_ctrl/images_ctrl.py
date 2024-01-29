from typing import List

from adapters.controllers.images_ctrl.dto.image_odto import ImageODto
from adapters.controllers.images_ctrl.dto.image_size_dto import ImageSizeDto
from entities.image_ent import ImageEnt
from fastapi import APIRouter, FastAPI, UploadFile
from fastapi.responses import StreamingResponse
from usecases.images_usc import ImagesUsc, images_usc_impl


class ImagesCtrl:
    def __init__(self, images_usc: ImagesUsc) -> None:
        self.images_usc = images_usc
        self._app = FastAPI()

        @self._app.get(
            summary="List images",
            path="/images",
            response_model=List[ImageODto],
        )
        def _() -> List[ImageODto]:
            images = self.images_usc.list_images()
            return [self._build_from_ent(image) for image in images]

        @self._app.delete(
            summary="Delete image", path="/images/{id}", response_model=ImageODto
        )
        def _(id: int) -> ImageODto:
            image = self.images_usc.delete_image(id)
            return self._build_from_ent(image)

        @self._app.post(
            summary="Upload local images",
            path="/images/upload-local",
            response_model=List[ImageODto],
        )
        def _(files: List[UploadFile]) -> List[ImageODto]:
            images = self.images_usc.upload_images(files)
            return [self._build_from_ent(image) for image in images]

        @self._app.get(
            summary="Image thumbnail (144x144)",
            path="/images/thumbnail/{id}.webp",
            response_class=StreamingResponse,
        )
        def _(id: int) -> StreamingResponse:
            bytesio = self.images_usc.get_image_thumbnail_as_webp(id)
            return StreamingResponse(content=bytesio, media_type="image/webp")

    def router(self) -> APIRouter:
        return self._app.router

    def _build_from_ent(self, ent: ImageEnt) -> ImageODto:
        src = self.images_usc.build_image_src(ent.id)
        return ImageODto(
            id=ent.id,
            src=src,
            name=ent.name,
            source=ImageSizeDto(width=ent.data.size[0], height=ent.data.size[1]),
        )


images_ctrl_impl = ImagesCtrl(images_usc_impl)
