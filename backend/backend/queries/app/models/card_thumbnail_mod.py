from datetime import datetime

from pydantic import BaseModel


class CardThumbnailMod(BaseModel):
    thumbnail_bytes: bytes
    image_id: int
    updated_at: datetime
