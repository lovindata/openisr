from dataclasses import dataclass
from io import BytesIO
from typing import List

from fastapi import APIRouter, FastAPI
from fastapi.responses import StreamingResponse

from backend.v2 import queries
from backend.v2.commands.processes.services.processes_svc import processes_svc_impl
from backend.v2.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from backend.v2.queries.app.models.card_mod import CardMod
from backend.v2.queries.app.repositories.card_download_rep import (
    card_downloads_rep_impl,
)
from backend.v2.queries.app.repositories.card_thumbnails_rep import (
    card_thumbnails_rep_impl,
)
from backend.v2.queries.app.repositories.cards_rep import cards_rep_impl


@dataclass
class AppCtrl:
    sqlalchemy_conf = sqlalchemy_conf_impl
    card_rep = cards_rep_impl
    card_thumbnail_rep = card_thumbnails_rep_impl
    download_rep = card_downloads_rep_impl
    processes_svc = processes_svc_impl

    def router(self) -> APIRouter:
        app = FastAPI()

        @app.get(
            tags=[queries.__name__],
            summary="Get cards",
            path="/query/v1/app/cards",
            response_model=List[CardMod],
        )
        def _() -> List[CardMod]:
            with self.sqlalchemy_conf.get_session() as session:
                return self.card_rep.list(session)

        @app.get(
            tags=[queries.__name__],
            summary="Card thumbnail (144x144)",
            path="/query/v1/app/cards/thumbnail/{image_id}.webp",
            response_class=StreamingResponse,
        )
        def _(image_id: int) -> StreamingResponse:
            with self.sqlalchemy_conf.get_session() as session:
                bytes = self.card_thumbnail_rep.get(session, image_id).thumbnail_bytes
                return StreamingResponse(
                    content=BytesIO(bytes), media_type="image/webp"
                )

        @app.get(
            tags=[queries.__name__],
            summary="Download image",
            path="/query/v1/app/cards/download",
            response_class=StreamingResponse,
        )
        def _(image_id: int) -> StreamingResponse:
            with self.sqlalchemy_conf.get_session() as session:
                download = self.download_rep.get(session, image_id)
                return StreamingResponse(
                    content=BytesIO(download.image_bytes),
                    media_type=download.media_type,
                    headers={
                        "Content-Disposition": f"attachment; filename={download.filename}"
                    },
                )

        return app.router


app_ctrl_impl = AppCtrl()
