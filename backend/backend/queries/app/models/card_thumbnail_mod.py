from pydantic import BaseModel


class CardThumbnailMod(BaseModel):
    image_id: int
    thumbnail_bytes: bytes
