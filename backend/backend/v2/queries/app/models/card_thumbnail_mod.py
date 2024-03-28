from pydantic import BaseModel, Field


class CardThumbnailMod(BaseModel):
    thumbnail_bytes: bytes
    image_id: int
