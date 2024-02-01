from adapters.controllers.processes_ctrl.dto.image_size_dto import ImageSizeDto
from adapters.controllers.processes_ctrl.dto.process_idto import ProcessIDto
from adapters.controllers.processes_ctrl.dto.process_odto import ProcessODto
from adapters.controllers.processes_ctrl.dto.status_dto import StatusDto
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
                dto.preserve_ratio,
                dto.target.width,
                dto.target.height,
                dto.enable_ai,
            )
            return self._build_from_ent(process)

        @self._app.get(
            summary="Get latest process",
            path="/images/{id}/process",
            response_model=ProcessODto,
        )
        def _(id: int) -> ProcessODto:
            ...

    def router(self) -> APIRouter:
        return self._app.router

    def _build_from_ent(self, ent: ProcessEnt) -> ProcessODto:
        def parse_status_ended() -> (
            "StatusDto.SuccessfulDto | StatusDto.FailedDto | None"
        ):
            match ent.status.ended:
                case StatusVal.Successful(at=at):
                    return StatusDto.SuccessfulDto(at=at)
                case StatusVal.Failed(at=at, error=error):
                    return StatusDto.FailedDto(at=at, error=error)
                case _:
                    return ent.status.ended

        ended = parse_status_ended()
        return ProcessODto(
            id=ent.id,
            target=ImageSizeDto(width=ent.target.width, height=ent.target.height),
            status=StatusDto(started_at=ent.status.started_at, ended=ended),
            extension=ent.extension.value,
            preserve_ratio=ent.preserve_ratio,
            enable_ai=ent.enable_ai,
        )


processes_ctrl_impl = ProcessesCtrl(processes_usc_impl)
