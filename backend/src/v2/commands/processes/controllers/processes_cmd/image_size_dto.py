from pydantic import BaseModel


class ImageSizeDto(BaseModel):
    width: int
    height: int
