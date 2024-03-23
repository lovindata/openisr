from io import BytesIO

from pydantic import BaseModel


class CardThumbnailMod(BaseModel):
    thumbnail_bytes: BytesIO
    image_id: int
