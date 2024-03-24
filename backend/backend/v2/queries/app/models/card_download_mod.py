from pydantic import BaseModel


class CardDownloadMod(BaseModel):
    bytes: bytes
    media_type: str
    filename: str
    image_id: int
