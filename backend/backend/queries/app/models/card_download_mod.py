from pydantic import BaseModel


class CardDownloadMod(BaseModel):
    image_id: int
    image_bytes: bytes
    media_type: str
    filename: str
