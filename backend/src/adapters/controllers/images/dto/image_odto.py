from adapters.controllers.images.dto.image_size_dto import ImageSizeDto
from adapters.controllers.processes.dto.process_odto import ProcessODto
from entities.image_ent import ImageEnt
from pydantic import BaseModel


class ImageODto(BaseModel):
    id: int
    src: str
    name: str
    source: ImageSizeDto

    @classmethod
    def build_from_ent(cls, ent: ImageEnt) -> "ImageODto":
        return ImageODto(
            id=ent.id,
            src=f"/images/thumbnail/{ent.id}.webp",
            name=ent.name,
            source=ImageSizeDto(width=ent.data.size[0], height=ent.data.size[1]),
        )
