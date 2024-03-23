from datetime import datetime
from typing import Any, List

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.indexable import index_property
from sqlalchemy.orm import Mapped, Session, mapped_column
from v2.confs.envs_conf import envs_conf_impl
from v2.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from v2.features.images.models.image_mod import ImageMod
from v2.features.processes.models.process_mod import ProcessMod
from v2.features.processes.models.process_mod.status_val import StatusVal
from v2.features.shared.models.extension_val import ExtensionVal
from v2.views.app.models.card_mod import CardMod


class CardRow(sqlalchemy_conf_impl.Base):
    __tablename__ = "cards"

    data: Mapped[dict[str, Any]] = mapped_column(type_=JSONB, nullable=False)
    image_id = index_property("data", "image_id")

    def to_mod(self) -> CardMod:
        return CardMod.model_validate(self.data)


class CardRep:
    envs_conf = envs_conf_impl

    def get_all(self, session: Session) -> List[CardMod]:
        stmt = select(CardRow)
        cards = [card.to_mod() for card in session.scalars(stmt).all()]
        return cards

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

        def update_row(row: CardRow, mod: CardMod) -> None:
            row.data = mod.model_dump()

        def insert_row(mod: CardMod) -> None:
            row = CardRow(data=mod.model_dump())
            session.add(row)

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
        extension = (
            process.extension if process else ExtensionVal(image.data.format)
        ).value
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
        update_row(row, mod) if row else insert_row(mod)


card_rep_impl = CardRep()
