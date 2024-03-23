from io import BytesIO

from pydantic import BaseModel


class DownloadMod(BaseModel):
    bytes: BytesIO
    media_type: str
    filename: str
    image_id: int
