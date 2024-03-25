from dataclasses import dataclass
from datetime import datetime
from typing import Any, List

from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, Session, mapped_column
from v2.commands.images.models.image_mod import ImageMod
from v2.commands.processes.models.process_mod.process_mod import ProcessMod
from v2.commands.processes.models.process_mod.status_val import StatusVal
from v2.confs.envs_conf import envs_conf_impl
from v2.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from v2.queries.app.models.card_mod import CardMod


class CardRow(sqlalchemy_conf_impl.Base):
    __tablename__ = "cards"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    data: Mapped[dict[str, Any]] = mapped_column(type_=JSON, nullable=False)
    image_id: Mapped[int] = mapped_column(unique=True, index=True)

    @classmethod
    def insert_with(cls, session: Session, mod: CardMod) -> None:
        row = CardRow(data=mod.model_dump(), image_id=mod.image_id)
        session.add(row)

    def update_with(self, mod: CardMod) -> None:
        self.data = mod.model_dump()
        self.image_id = mod.image_id

    def to_mod(self) -> CardMod:
        return CardMod(**self.data)


@dataclass
class CardsRep:
    envs_conf = envs_conf_impl

    def get_all(self, session: Session) -> List[CardMod]:
        return [card.to_mod() for card in session.query(CardRow).all()]

    def sync(
        self, session: Session, image: ImageMod, process: ProcessMod | None
    ) -> None:
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
                    return CardMod.Downloadable()
                case StatusVal.Failed(at=at):
                    duration = round((at - process.status.started_at).total_seconds())
                    return CardMod.Errored(duration=duration)
                case None:
                    duration = round(
                        (datetime.now() - process.status.started_at).total_seconds()
                    )
                    return CardMod.Stoppable(duration=duration)

        thumbnail_src = build_thumbnail_src()
        source = CardMod.Dimension(width=image.data.size[0], height=image.data.size[1])
        target = (
            CardMod.Dimension(width=process.target.width, height=process.target.height)
            if process
            else None
        )
        status = parse_process_status_ended(process) if process else CardMod.Runnable()
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
        row = session.query(CardRow).where(CardRow.image_id == image.id).one_or_none()
        row.update_with(mod) if row else CardRow.insert_with(session, mod)

    def clean_sync(self, session: Session, image_id: int) -> None:
        session.query(CardRow).where(CardRow.image_id == image_id).delete()


cards_rep_impl = CardsRep()
