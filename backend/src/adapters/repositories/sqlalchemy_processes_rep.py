from datetime import datetime
from typing import Literal, Optional

from adapters.repositories.configs.base import Base
from entities.process_ent.extension_val import ExtensionVal
from entities.process_ent.image_size_val import ImageSizeVal
from entities.process_ent.process_ent import ProcessEnt
from entities.process_ent.status_val import StatusVal
from helpers.exception_utils import BadRequestException
from sqlalchemy import ForeignKey, func, select
from sqlalchemy.orm import Mapped, Session, mapped_column
from usecases.repositories.processes_rep import ProcessesRep


class ProcessRow(Base):
    __tablename__ = "processes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_image_id: Mapped[int] = mapped_column(ForeignKey("images.id"))
    extension: Mapped[ExtensionVal]
    preserve_ratio: Mapped[bool]
    target_width: Mapped[int]
    target_height: Mapped[int]
    enable_ai: Mapped[bool]
    status_started_at: Mapped[datetime] = mapped_column(default=func.now())
    status_ended_successful_at: Mapped[Optional[datetime]]
    status_ended_failed_at: Mapped[Optional[datetime]]
    status_ended_failed_error: Mapped[Optional[str]]

    @classmethod
    def from_ent(cls, entity: ProcessEnt) -> "ProcessRow":
        return ProcessRow(
            id=entity.id,
            source_image_id=entity.source_image_id,
            extension=entity.extension.value,
            preserve_ratio=entity.preserve_ratio,
            target_width=entity.target.width,
            target_height=entity.target.height,
            enable_ai=entity.enable_ai,
            started_at=entity.status.started_at,
            successful_at=ended_at.at
            if type(ended_at := entity.status.ended) is StatusVal.Successful
            else None,
            failed_at=ended_at.at
            if type(ended_at := entity.status.ended) is StatusVal.Failed
            else None,
            error=ended_at.error
            if type(ended_at := entity.status.ended) is StatusVal.Failed
            else None,
        )

    def to_ent(self) -> ProcessEnt:
        def parse_status_ended_columns() -> (
            StatusVal.Successful | StatusVal.Failed | None
        ):
            if self.status_ended_successful_at:
                return StatusVal.Successful(self.status_ended_successful_at)
            elif self.status_ended_failed_at and self.status_ended_failed_error:
                return StatusVal.Failed(
                    self.status_ended_failed_at, self.status_ended_failed_error
                )
            else:
                return None

        ended = parse_status_ended_columns()
        return ProcessEnt(
            self.id,
            self.source_image_id,
            self.extension,
            self.preserve_ratio,
            ImageSizeVal(self.target_width, self.target_height),
            self.enable_ai,
            StatusVal(self.status_started_at, ended),
        )


class SqlAlchemyProcessesRep(ProcessesRep):
    def create_run(
        self,
        session: Session,
        image_id: int,
        extension: Literal["JPEG", "PNG", "WEBP"],
        preserve_ratio: bool,
        target_width: int,
        target_height: int,
        enable_ai: bool,
    ) -> ProcessEnt:
        row = ProcessRow(
            source_image_id=image_id,
            extension=ExtensionVal(extension),
            preserve_ratio=preserve_ratio,
            target_width=target_width,
            target_height=target_height,
            enable_ai=enable_ai,
        )
        session.add(row)
        session.flush()
        return row.to_ent()

    def get_latest(self, session: Session, image_id: int) -> ProcessEnt:
        ...

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
