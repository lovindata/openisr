import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Literal

from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.sql import func

from backend.commands.images.models.image_mod import ImageMod
from backend.commands.processes.models.process_mod.process_ai_val import ProcessAIVal
from backend.commands.processes.models.process_mod.process_bicubic_val import (
    ProcessBicubicVal,
)
from backend.commands.processes.models.process_mod.process_mod import ProcessMod
from backend.commands.processes.models.process_mod.process_status_val import (
    ProcessStatusVal,
)
from backend.commands.shared.models.extension_val import ExtensionVal
from backend.confs.envs_conf import envs_conf_impl
from backend.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from backend.queries.app.models.card_mod import CardMod


class CardRow(sqlalchemy_conf_impl.Base):
    __tablename__ = "cards"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(unique=True, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        insert_default=func.now(), onupdate=func.now()
    )
    data: Mapped[dict[str, Any]] = mapped_column(type_=JSON)

    @classmethod
    def insert_with(cls, session: Session, mod: CardMod) -> None:
        row = CardRow(image_id=mod.image_id, data=json.loads(mod.model_dump_json()))
        session.add(row)
        session.flush()

    def update_with(self, mod: CardMod) -> None:
        self.data = json.loads(mod.model_dump_json())
        self.image_id = mod.image_id

    def to_mod(self) -> CardMod:
        return CardMod.model_validate_json(json.dumps(self.data))


@dataclass
class CardsRep:
    envs_conf = envs_conf_impl

    def list(self, session: Session) -> List[CardMod]:
        return [
            card.to_mod()
            for card in session.query(CardRow).order_by(CardRow.updated_at.desc()).all()
        ]

    def sync(
        self, session: Session, image: ImageMod, process: ProcessMod | None
    ) -> None:
        mod = self._build_mod(image, process)
        row = session.query(CardRow).where(CardRow.image_id == image.id).one_or_none()
        row.update_with(mod) if row else CardRow.insert_with(session, mod)

    def clean_sync(self, session: Session, image_id: int) -> None:
        row = session.query(CardRow).where(CardRow.image_id == image_id).one_or_none()
        if row:
            session.delete(row)

    def _build_mod(self, image: ImageMod, process: ProcessMod | None) -> CardMod:
        def build_thumbnail_src() -> str:
            src = f"/queries/v1/app/cards/thumbnail/{image.id}.webp"
            if not self.envs_conf.prod_mode:
                src = f"http://localhost:{self.envs_conf.api_port}" + src
            return src

        def build_source() -> CardMod.Dimension:
            if process:
                return CardMod.Dimension(
                    width=process.source.width, height=process.source.height
                )
            else:
                return CardMod.Dimension(
                    width=image.data.size[0], height=image.data.size[1]
                )

        def build_target() -> CardMod.Dimension | None:
            if not process:
                return None
            else:
                match process.scaling:
                    case ProcessBicubicVal(target=target):
                        return CardMod.Dimension(
                            width=target.width, height=target.height
                        )
                    case ProcessAIVal(scale=scale):
                        return CardMod.Dimension(
                            width=process.source.width * scale,
                            height=process.source.height * scale,
                        )

        def build_status() -> (
            CardMod.Runnable
            | CardMod.Stoppable
            | CardMod.Errored
            | CardMod.Downloadable
        ):
            if process is None:
                return CardMod.Runnable(type="Runnable")
            else:
                match process.status.ended:
                    case ProcessStatusVal.Successful():
                        image_src = (
                            f"/queries/v1/app/cards/download?image_id={image.id}"
                        )
                        if not self.envs_conf.prod_mode:
                            image_src = (
                                f"http://localhost:{self.envs_conf.api_port}"
                                + image_src
                            )
                        return CardMod.Downloadable(
                            type="Downloadable", image_src=image_src
                        )
                    case ProcessStatusVal.Failed(error=error):
                        return CardMod.Errored(
                            type="Errored",
                            duration=round(
                                (
                                    process.status.ended.at - process.status.started_at
                                ).total_seconds()
                            ),
                            error=error,
                        )
                    case None:
                        return CardMod.Stoppable(
                            type="Stoppable", started_at=process.status.started_at
                        )

        def build_default_extension() -> ExtensionVal:
            if process:
                return process.extension
            else:
                return image.extension()

        def build_default_scaling_type() -> Literal["Bicubic", "AI"]:
            if not process:
                return "Bicubic"
            match process.scaling:
                case ProcessBicubicVal():
                    return "Bicubic"
                case ProcessAIVal():
                    return "AI"

        def build_default_scaling_bicubic() -> CardMod.Bicubic:
            if not process:
                return CardMod.Bicubic(
                    type="Bicubic",
                    preserve_ratio=True,
                    target=CardMod.Dimension(
                        width=image.data.size[0], height=image.data.size[1]
                    ),
                )
            match process.scaling:
                case ProcessBicubicVal(target=target):
                    return CardMod.Bicubic(
                        type="Bicubic",
                        preserve_ratio=True,
                        target=CardMod.Dimension(
                            width=target.width, height=target.height
                        ),
                    )
                case ProcessAIVal(scale=scale):
                    return CardMod.Bicubic(
                        type="Bicubic",
                        preserve_ratio=True,
                        target=CardMod.Dimension(
                            width=process.source.width * scale,
                            height=process.source.height * scale,
                        ),
                    )

        def build_default_scaling_ai() -> CardMod.AI:
            if not process:
                return CardMod.AI(type="AI", scale=2)
            match process.scaling:
                case ProcessBicubicVal():
                    return CardMod.AI(type="AI", scale=2)
                case ProcessAIVal():
                    return CardMod.AI(type="AI", scale=process.scaling.scale)

        return CardMod(
            thumbnail_src=build_thumbnail_src(),
            name=image.name,
            source=build_source(),
            target=build_target(),
            status=build_status(),
            default_extension=build_default_extension(),
            default_scaling_type=build_default_scaling_type(),
            default_scaling_bicubic=build_default_scaling_bicubic(),
            default_scaling_ai=build_default_scaling_ai(),
            image_id=image.id,
        )


cards_rep_impl = CardsRep()
