from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from sqlalchemy import ForeignKey, and_
from sqlalchemy.orm import Mapped, Session, composite, mapped_column

from backend.commands.images.models.image_mod import ImageMod
from backend.commands.processes.controllers.processes_ctrl.process_dto import ProcessDto
from backend.commands.processes.models.process_mod.process_mod import ProcessMod
from backend.commands.processes.repositories.processes_rep.composites.process_scaling_comp import (
    ProcessScalingComp,
)
from backend.commands.processes.repositories.processes_rep.composites.process_source_comp import (
    ProcessSourceComp,
)
from backend.commands.processes.repositories.processes_rep.composites.process_status_comp import (
    ProcessStatusComp,
)
from backend.commands.shared.models.extension_val import ExtensionVal
from backend.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from backend.helpers.exception_utils import BadRequestException


class ProcessRow(sqlalchemy_conf_impl.Base):
    __tablename__ = "processes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, index=True)
    image_id: Mapped[Optional[int]] = mapped_column(ForeignKey("images.id"), index=True)
    extension: Mapped[ExtensionVal]

    source_width: Mapped[int]
    source_height: Mapped[int]
    source: Mapped[ProcessSourceComp] = composite(
        ProcessSourceComp._generate, "source_width", "source_height"
    )

    scaling_bicubic_target_width: Mapped[Optional[int]]
    scaling_bicubic_target_height: Mapped[Optional[int]]
    scaling_ai_scale: Mapped[Optional[int]]
    scaling: Mapped[ProcessScalingComp] = composite(
        ProcessScalingComp._generate,
        "scaling_bicubic_target_width",
        "scaling_bicubic_target_height",
        "scaling_ai_scale",
    )

    status_started_at: Mapped[datetime]
    status_ended_successful_at: Mapped[Optional[datetime]]
    status_ended_failed_at: Mapped[Optional[datetime]]
    status_ended_failed_error: Mapped[Optional[str]]
    status_ended_failed_stacktrace: Mapped[Optional[str]]
    status: Mapped[ProcessStatusComp] = composite(
        ProcessStatusComp._generate,
        "status_started_at",
        "status_ended_successful_at",
        "status_ended_failed_at",
        "status_ended_failed_error",
        "status_ended_failed_stacktrace",
    )

    def update_with(self, mod: ProcessMod) -> "ProcessRow":
        self.image_id = mod.image_id
        self.extension = mod.extension
        self.source = ProcessSourceComp(mod.source)
        self.scaling = ProcessScalingComp(mod.scaling)
        self.status = ProcessStatusComp(mod.status)
        return self

    def to_mod(self) -> ProcessMod:
        return ProcessMod(
            id=self.id,
            image_id=self.image_id,
            extension=self.extension,
            source=self.source.value,
            scaling=self.scaling.value,
            status=self.status.value,
        )


@dataclass
class ProcessesRep:
    def create_run_with_dto(
        self, session: Session, image: ImageMod, dto: ProcessDto
    ) -> ProcessMod:
        row = ProcessRow(
            image_id=image.id,
            extension=dto.extension,
            source_width=image.data.size[0],
            source_height=image.data.size[1],
            scaling=ProcessScalingComp(dto.scaling.to_val()),
            status_started_at=datetime.now(),
        )
        session.add(row)
        session.flush()
        return row.to_mod()

    def create_run_with_mod(
        self, session: Session, image: ImageMod, mod: ProcessMod
    ) -> ProcessMod:
        row = ProcessRow(
            image_id=image.id,
            extension=mod.extension,
            source_width=image.data.size[0],
            source_height=image.data.size[1],
            scaling=ProcessScalingComp(mod.scaling),
            status_started_at=datetime.now(),
        )
        session.add(row)
        session.flush()
        return row.to_mod()

    def update(self, session: Session, mod: ProcessMod) -> None:
        session.query(ProcessRow).where(ProcessRow.id == mod.id).one().update_with(mod)

    def get_latest(self, session: Session, image_id: int) -> ProcessMod | None:
        row = (
            session.query(ProcessRow)
            .where(ProcessRow.image_id == image_id)
            .order_by(ProcessRow.status_started_at.desc())
            .limit(1)
            .one_or_none()
        )
        return row.to_mod() if row else None

    def get_latest_or_raise(self, session: Session, image_id: int) -> ProcessMod:
        process_latest = self.get_latest(session, image_id)
        if process_latest is None:
            raise BadRequestException(
                f"No latest process found for image with ID={image_id}."
            )
        return process_latest

    def list_running(self, session: Session) -> List[ProcessMod]:
        rows = (
            session.query(ProcessRow)
            .where(
                and_(
                    ProcessRow.status_ended_successful_at == None,
                    ProcessRow.status_ended_failed_at == None,
                )
            )
            .all()
        )
        mods = [row.to_mod() for row in rows]
        return mods

    def delete(self, session: Session, process_id: int) -> None:
        row = session.query(ProcessRow).where(ProcessRow.id == process_id).one_or_none()
        if row:
            session.delete(row)


processes_rep_impl = ProcessesRep()
