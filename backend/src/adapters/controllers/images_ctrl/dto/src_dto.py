from pydantic import BaseModel


class SrcDto(BaseModel):
    thumbnail: str
    download: str
