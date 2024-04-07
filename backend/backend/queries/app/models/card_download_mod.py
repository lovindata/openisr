from pydantic import BaseModel, Field


class CardDownloadMod(BaseModel):
    image_bytes: bytes
    media_type: str
    filename: str
    image_id: int
