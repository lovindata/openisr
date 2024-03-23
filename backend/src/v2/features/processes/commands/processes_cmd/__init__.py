from dataclasses import dataclass

from fastapi import APIRouter, FastAPI
from v2.features.processes.commands.processes_cmd.process_dto import ProcessDto
from v2.features.processes.services.processes_svc import processes_svc_impl


@dataclass
class ProcessesCmd:
    processes_svc = processes_svc_impl

    def router(self) -> APIRouter:
        app = FastAPI()

        @app.post(
            summary="Run process",
            path="/images/{id}/process",
            response_model=None,
        )
        def _(id: int, dto: ProcessDto) -> None:
            self.processes_svc.run(id, dto)

        @app.post(
            summary="Retry latest process",
            path="/images/{id}/process/retry",
            response_model=None,
        )
        def _(id: int) -> None:
            self.processes_svc.retry(id)

        @app.delete(
            summary="Stop latest process",
            path="/images/{id}/process",
            response_model=None,
        )
        def _(id: int) -> None:
            self.processes_svc.stop(id)

        return app.router


processes_cmd_impl = ProcessesCmd()
