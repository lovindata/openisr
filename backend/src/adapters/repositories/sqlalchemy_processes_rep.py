from datetime import datetime
from typing import Optional

from adapters.repositories.configs.base import Base
from entities.process_ent import ProcessEnt
from entities.process_ent.image_size_val import ImageSizeVal
from entities.process_ent.status_val import StatusVal
from entities.shared.extension_val import ExtensionVal
from helpers.exception_utils import BadRequestException
from sqlalchemy import ForeignKey, select
from sqlalchemy.orm import Mapped, Session, mapped_column
from usecases.repositories.processes_rep import ProcessesRep


class ProcessRow(Base):
    __tablename__ = "processes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
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

    def set_all_with(self, entity: ProcessEnt) -> "ProcessRow":
        self.image_id = entity.image_id
        self.extension = entity.extension
        self.target_width = entity.target.width
        self.target_height = entity.target.height
        self.enable_ai = entity.enable_ai
        self.status_started_at = entity.status.started_at
        self.status_ended_successful_at = (
            ended.at
            if type(ended := entity.status.ended) is StatusVal.Successful
            else None
        )
        self.status_ended_failed_at = (
            ended.at if type(ended := entity.status.ended) is StatusVal.Failed else None
        )
        self.status_ended_failed_error = (
            ended.error
            if type(ended := entity.status.ended) is StatusVal.Failed
            else None
        )
        self.status_ended_failed_stacktrace = (
            ended.stacktrace
            if type(ended := entity.status.ended) is StatusVal.Failed
            else None
        )
        return self

    def to_ent(self) -> ProcessEnt:
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
        return ProcessEnt(
            self.id,
            self.image_id,
            self.extension,
            ImageSizeVal(self.target_width, self.target_height),
            self.enable_ai,
            StatusVal(self.status_started_at, ended),
        )


class SqlAlchemyProcessesRep(ProcessesRep):
    def create_run(
        self,
        session: Session,
        image_id: int,
        extension: ExtensionVal,
        target_width: int,
        target_height: int,
        enable_ai: bool,
    ) -> ProcessEnt:
        row = ProcessRow(
            image_id=image_id,
            extension=ExtensionVal(extension),
            target_width=target_width,
            target_height=target_height,
            enable_ai=enable_ai,
            status_started_at=datetime.now(),
        )
        session.add(row)
        session.flush()
        return row.to_ent()

    def update(self, session: Session, ent: ProcessEnt) -> ProcessEnt:
        return (
            self._get_or_raise_when_process_not_found(session, ent.id)
            .set_all_with(ent)
            .to_ent()
        )

    def get_latest(self, session: Session, image_id: int) -> ProcessEnt | None:
        stmt = (
            select(ProcessRow)
            .where(ProcessRow.image_id == image_id)
            .order_by(ProcessRow.status_started_at.desc())
            .limit(1)
        )
        row = session.scalar(stmt)
        return row.to_ent() if row else None

    def get_latest_or_throw(self, session: Session, image_id: int) -> ProcessEnt:
        process_latest = self.get_latest(session, image_id)
        if process_latest is None:
            raise BadRequestException(
                f"No latest process found for image with ID={image_id}."
            )
        return process_latest

    def delete(self, session: Session, id: int) -> ProcessEnt:
        row = self._get_or_raise_when_process_not_found(session, id)
        ent = row.to_ent()
        session.delete(row)
        return ent

    def _get_or_raise_when_process_not_found(
        self, session: Session, id: int
    ) -> ProcessRow:
        stmt = select(ProcessRow).where(ProcessRow.id == id)
        row = session.scalar(stmt)
        if row:
            return row
        else:
            raise BadRequestException(
                f"The process associated with the provided ID ({id}) does not exist."
            )


sqlalchemy_processes_rep_impl = SqlAlchemyProcessesRep()
