from pydantic import BaseModel


class CardThumbnailMod(BaseModel):
    thumbnail_bytes: bytes
    image_id: int
