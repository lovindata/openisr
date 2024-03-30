from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from sqlalchemy import ForeignKey, func
from sqlalchemy.orm import Mapped, Session, mapped_column

from backend.v2.commands.processes.controllers.processes_ctrl.process_dto import (
    ProcessDto,
)
from backend.v2.commands.processes.models.process_mod.image_size_val import ImageSizeVal
from backend.v2.commands.processes.models.process_mod.process_mod import ProcessMod
from backend.v2.commands.processes.models.process_mod.status_val import StatusVal
from backend.v2.commands.shared.models.extension_val import ExtensionVal
from backend.v2.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from backend.v2.helpers.exception_utils import BadRequestException


class ProcessRow(sqlalchemy_conf_impl.Base):
    __tablename__ = "processes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, index=True)
    image_id: Mapped[Optional[int]] = mapped_column(ForeignKey("images.id"))
    extension: Mapped[ExtensionVal]
    target_width: Mapped[int]
    target_height: Mapped[int]
    enable_ai: Mapped[bool]
    status_started_at: Mapped[datetime]
    status_ended_successful_at: Mapped[Optional[datetime]]
    status_ended_failed_at: Mapped[Optional[datetime]]
    status_ended_failed_error: Mapped[Optional[str]]
    status_ended_failed_stacktrace: Mapped[Optional[str]]

    def update_with(self, model: ProcessMod) -> "ProcessRow":
        self.image_id = model.image_id
        self.extension = model.extension
        self.target_width = model.target.width
        self.target_height = model.target.height
        self.enable_ai = model.enable_ai
        self.status_started_at = model.status.started_at
        self.status_ended_successful_at = (
            ended.at
            if type(ended := model.status.ended) is StatusVal.Successful
            else None
        )
        self.status_ended_failed_at = (
            ended.at if type(ended := model.status.ended) is StatusVal.Failed else None
        )
        self.status_ended_failed_error = (
            ended.error
            if type(ended := model.status.ended) is StatusVal.Failed
            else None
        )
        self.status_ended_failed_stacktrace = (
            ended.stacktrace
            if type(ended := model.status.ended) is StatusVal.Failed
            else None
        )
        return self

    def to_mod(self) -> ProcessMod:
        def parse_status_ended_columns() -> (
            StatusVal.Successful | StatusVal.Failed | None
        ):
            if self.status_ended_successful_at:
                return StatusVal.Successful(self.status_ended_successful_at)
            elif self.status_ended_failed_at and self.status_ended_failed_error:
                return StatusVal.Failed(
                    self.status_ended_failed_at,
                    self.status_ended_failed_error,
                    self.status_ended_failed_stacktrace,
                )
            else:
                return None

        ended = parse_status_ended_columns()
        return ProcessMod(
            self.id,
            self.image_id,
            self.extension,
            ImageSizeVal(self.target_width, self.target_height),
            self.enable_ai,
            StatusVal(self.status_started_at, ended),
        )


@dataclass
class ProcessesRep:
    def create_run_with_dto(
        self, session: Session, image_id: int, dto: ProcessDto
    ) -> ProcessMod:
        row = ProcessRow(
            image_id=image_id,
            extension=dto.extension,
            target_width=dto.target.width,
            target_height=dto.target.height,
            enable_ai=dto.enable_ai,
            status_started_at=datetime.now(),
        )
        session.add(row)
        session.flush()
        return row.to_mod()

    def create_run_with_mod(
        self, session: Session, image_id: int, mod: ProcessMod
    ) -> ProcessMod:
        row = ProcessRow(
            image_id=image_id,
            extension=mod.extension,
            target_width=mod.target.width,
            target_height=mod.target.height,
            enable_ai=mod.enable_ai,
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

    def get_latest_or_throw(self, session: Session, image_id: int) -> ProcessMod:
        process_latest = self.get_latest(session, image_id)
        if process_latest is None:
            raise BadRequestException(
                f"No latest process found for image with ID={image_id}."
            )
        return process_latest

    def list_latest(self, session: Session, image_ids: List[int]) -> List[ProcessMod]:
        subquery = (
            session.query(
                ProcessRow.image_id,
                func.max(ProcessRow.status_started_at).label("status_started_at"),
            )
            .where(ProcessRow.image_id.in_(image_ids))
            .group_by(ProcessRow.image_id)
        ).subquery()
        query = (
            session.query(ProcessRow)
            .where(ProcessRow.image_id.in_(image_ids))
            .join(
                subquery,
                (ProcessRow.image_id == subquery.c.image_id)
                & (ProcessRow.status_started_at == subquery.c.status_started_at),
            )
        )
        mods = [row.to_mod() for row in query.all()]
        return mods

    def delete(self, session: Session, id: int) -> None:
        row = session.query(ProcessRow).where(ProcessRow.id == id).one()
        session.delete(row)


processes_rep_impl = ProcessesRep()
