from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from sqlalchemy import ForeignKey, func
from sqlalchemy.orm import Mapped, Session, mapped_column

from backend.v2.commands.images.models.image_mod import ImageMod
from backend.v2.commands.processes.controllers.processes_ctrl.process_dto import (
    ProcessDto,
)
from backend.v2.commands.processes.models.process_mod.image_size_val import ImageSizeVal
from backend.v2.commands.processes.models.process_mod.process_mod import ProcessMod
from backend.v2.commands.processes.models.process_mod.status_val import StatusVal
from backend.v2.commands.shared.models.extension_val import ExtensionVal
from backend.v2.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from backend.v2.helpers.exception_utils import (
    BadRequestException,
    ServerInternalErrorException,
)


class ProcessRow(sqlalchemy_conf_impl.Base):
    __tablename__ = "processes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, index=True)
    image_id: Mapped[Optional[int]] = mapped_column(ForeignKey("images.id"), index=True)
    extension: Mapped[ExtensionVal]
    source_width: Mapped[int]
    source_height: Mapped[int]
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

        def get_image_id_or_raise() -> int:
            if self.image_id is None:
                raise ServerInternalErrorException(
                    "Unconvertible dangling process row."
                )
            return self.image_id

        ended = parse_status_ended_columns()
        image_id = get_image_id_or_raise()
        return ProcessMod(
            self.id,
            image_id,
            self.extension,
            ImageSizeVal(self.source_width, self.source_height),
            ImageSizeVal(self.target_width, self.target_height),
            self.enable_ai,
            StatusVal(self.status_started_at, ended),
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
            target_width=dto.target.width,
            target_height=dto.target.height,
            enable_ai=dto.enable_ai,
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

    def bulk_update(self, session: Session, mods: List[ProcessMod]) -> None:
        mods_dict = {mod.id: mod for mod in mods}
        rows = (
            session.query(ProcessRow).where(ProcessRow.id.in_(mods_dict.keys())).all()
        )
        if (nb_rows := len(rows)) != (nb_mods := len(mods)):
            raise ServerInternalErrorException(
                f"Bulk update failed: detected {nb_rows} rows instead of {nb_mods}."
            )
        for row in rows:
            row.update_with(mods_dict[row.id])

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

    def list(self, session: Session, image_ids: List[int]) -> List[ProcessMod]:
        rows = session.query(ProcessRow).where(ProcessRow.image_id.in_(image_ids)).all()
        mods = [row.to_mod() for row in rows]
        return mods

    def delete(self, session: Session, process_id: int) -> None:
        row = session.query(ProcessRow).where(ProcessRow.id == process_id).one_or_none()
        if row:
            session.delete(row)


processes_rep_impl = ProcessesRep()
