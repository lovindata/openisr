from dataclasses import dataclass
from typing import List

from fastapi import APIRouter, FastAPI
from fastapi.responses import StreamingResponse
from v2.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from v2.views.app.models.card_mod import CardMod
from v2.views.app.repositories.card_rep import card_rep_impl
from v2.views.app.repositories.card_thumbnail_rep import card_thumbnail_rep_impl
from v2.views.app.repositories.download_rep import download_rep_impl


@dataclass
class AppQry:
    sqlalchemy_conf = sqlalchemy_conf_impl
    card_rep = card_rep_impl
    card_thumbnail_rep = card_thumbnail_rep_impl
    download_rep = download_rep_impl

    def router(self) -> APIRouter:
        app = FastAPI()

        @app.get(summary="Get cards", path="/app/cards", response_model=List[CardMod])
        def _() -> List[CardMod]:
            with self.sqlalchemy_conf.get_session() as session:
                return self.card_rep.get_all(session)

        @app.get(
            summary="Card thumbnail (144x144)",
            path="/app/cards/thumbnail/{image_id}.webp",
            response_class=StreamingResponse,
        )
        def _(image_id: int) -> StreamingResponse:
            with self.sqlalchemy_conf.get_session() as session:
                bytes = self.card_thumbnail_rep.get(session, image_id).thumbnail_bytes
                return StreamingResponse(content=bytes, media_type="image/webp")

        @app.get(
            summary="Download image",
            path="/app/download",
            response_class=StreamingResponse,
        )
        def _(image_id: int) -> StreamingResponse:
            with self.sqlalchemy_conf.get_session() as session:
                download = self.download_rep.get(session, image_id)
                return StreamingResponse(
                    content=download.bytes,
                    media_type=download.media_type,
                    headers={
                        "Content-Disposition": f"attachment; filename={download.filename}"
                    },
                )

        return app.router


app_qry_impl = AppQry()
