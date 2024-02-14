from datetime import datetime

from adapters.controllers.processes_ctrl.dto.process_idto import ProcessIDto
from adapters.controllers.processes_ctrl.dto.process_odto import ProcessODto
from adapters.controllers.processes_ctrl.dto.status_dto import StatusDto
from adapters.controllers.shared.dto.image_size_dto import ImageSizeDto
from entities.process_ent import ProcessEnt
from entities.process_ent.status_val import StatusVal
from fastapi import APIRouter, FastAPI
from usecases.processes_usc import ProcessesUsc, processes_usc_impl


class ProcessesCtrl:
    def __init__(self, processes_usc: ProcessesUsc) -> None:
        self.processes_usc = processes_usc
        self._app = FastAPI()

        @self._app.post(
            summary="Run process",
            path="/images/{id}/process",
            response_model=ProcessODto,
        )
        def _(id: int, dto: ProcessIDto) -> ProcessODto:
            process = self.processes_usc.run(
                id,
                dto.extension,
                dto.target.width,
                dto.target.height,
                dto.enable_ai,
            )
            return self._build_from_ent(process)

        @self._app.get(
            summary="Get latest process",
            path="/images/{id}/process",
            response_model=ProcessODto | None,
        )
        def _(id: int) -> ProcessODto | None:
            ent = self.processes_usc.get_latest_process(id)
            return self._build_from_ent(ent) if ent else None

    def router(self) -> APIRouter:
        return self._app.router

    def _build_from_ent(self, ent: ProcessEnt) -> ProcessODto:
        def build_duration() -> int:
            match ent.status.ended:
                case None:
                    return round(
                        (datetime.now() - ent.status.started_at).total_seconds()
                    )
                case _:
                    return round(
                        (ent.status.ended.at - ent.status.started_at).total_seconds()
                    )

        def parse_status_ended() -> (
            "StatusDto.SuccessfulDto | StatusDto.FailedDto | None"
        ):
            match ent.status.ended:
                case StatusVal.Successful(at=_):
                    return StatusDto.SuccessfulDto(kind="successful")
                case StatusVal.Failed(at=_, error=error):
                    return StatusDto.FailedDto(kind="failed", error=error)
                case _:
                    return ent.status.ended

        duration = build_duration()
        ended = parse_status_ended()
        return ProcessODto(
            id=ent.id,
            target=ImageSizeDto(width=ent.target.width, height=ent.target.height),
            status=StatusDto(duration=duration, ended=ended),
            extension=ent.extension.value,
            enable_ai=ent.enable_ai,
        )


processes_ctrl_impl = ProcessesCtrl(processes_usc_impl)
