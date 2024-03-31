import json
from dataclasses import dataclass
from typing import Any, List, Tuple

from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, Session, mapped_column

from backend.v2.commands.images.models.image_mod import ImageMod
from backend.v2.commands.processes.models.process_mod.process_mod import ProcessMod
from backend.v2.commands.processes.models.process_mod.status_val import StatusVal
from backend.v2.confs.envs_conf import envs_conf_impl
from backend.v2.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from backend.v2.queries.app.models.card_mod import CardMod


class CardRow(sqlalchemy_conf_impl.Base):
    __tablename__ = "cards"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    data: Mapped[dict[str, Any]] = mapped_column(type_=JSON, nullable=False)
    image_id: Mapped[int] = mapped_column(unique=True, index=True)

    @classmethod
    def insert_with(cls, session: Session, mod: CardMod) -> None:
        row = CardRow(data=json.loads(mod.model_dump_json()), image_id=mod.image_id)
        session.add(row)

    def update_with(self, mod: CardMod) -> None:
        self.data = json.loads(mod.model_dump_json())
        self.image_id = mod.image_id

    def to_mod(self) -> CardMod:
        return CardMod.model_validate_json(json.dumps(self.data))


@dataclass
class CardsRep:
    envs_conf = envs_conf_impl

    def list(self, session: Session) -> List[CardMod]:
        return [card.to_mod() for card in session.query(CardRow).all()]

    def sync(
        self, session: Session, image: ImageMod, process: ProcessMod | None
    ) -> None:
        mod = self._build_mod(image, process)
        row = session.query(CardRow).where(CardRow.image_id == image.id).one_or_none()
        row.update_with(mod) if row else CardRow.insert_with(session, mod)

    def bulk_sync(
        self,
        session: Session,
        image_process_couples: List[Tuple[ImageMod, ProcessMod | None]],
    ) -> None:
        mods_dict = {
            image.id: self._build_mod(image, process)
            for image, process in image_process_couples
        }
        rows_dict = {
            row.image_id: row
            for row in session.query(CardRow)
            .where(CardRow.image_id.in_(mods_dict.keys()))
            .all()
        }
        for mod_image_id in mods_dict:
            row = rows_dict.get(mod_image_id, None)
            if row:
                row.update_with(mods_dict[mod_image_id])
            else:
                CardRow.insert_with(session, mods_dict[mod_image_id])

    def clean_sync(self, session: Session, image_id: int) -> None:
        row = session.query(CardRow).where(CardRow.image_id == image_id).one_or_none()
        if row:
            session.delete(row)

    def _build_mod(self, image: ImageMod, process: ProcessMod | None) -> CardMod:
        def build_thumbnail_src() -> str:
            src = f"/app/cards/thumbnail/{image.id}.webp"
            if not self.envs_conf.prod_mode:
                src = f"http://localhost:{self.envs_conf.api_port}" + src
            return src

        def parse_process_status_ended(
            process: ProcessMod,
        ) -> CardMod.Stoppable | CardMod.Errored | CardMod.Downloadable:
            match process.status.ended:
                case StatusVal.Successful():
                    return CardMod.Downloadable(type="Downloadable")
                case StatusVal.Failed():
                    return CardMod.Errored(
                        type="Errored",
                        duration=round(
                            (
                                process.status.ended.at - process.status.started_at
                            ).total_seconds()
                        ),
                    )
                case None:
                    return CardMod.Stoppable(
                        type="Stoppable", started_at=process.status.started_at
                    )

        thumbnail_src = build_thumbnail_src()
        source = (
            CardMod.Dimension(width=process.source.width, height=process.source.height)
            if process
            else CardMod.Dimension(width=image.data.size[0], height=image.data.size[1])
        )
        target = (
            CardMod.Dimension(width=process.target.width, height=process.target.height)
            if process
            else None
        )
        status = (
            parse_process_status_ended(process)
            if process
            else CardMod.Runnable(type="Runnable")
        )
        error = (
            process.status.ended.error
            if process and (type(process.status.ended) is StatusVal.Failed)
            else None
        )
        extension = (process.extension if process else image.extension()).value
        enable_ai = process.enable_ai if process else False
        mod = CardMod(
            thumbnail_src=thumbnail_src,
            name=image.name,
            source=source,
            target=target,
            status=status,
            image_id=image.id,
            error=error,
            extension=extension,
            preserve_ratio=True,
            enable_ai=enable_ai,
        )
        return mod


cards_rep_impl = CardsRep()
