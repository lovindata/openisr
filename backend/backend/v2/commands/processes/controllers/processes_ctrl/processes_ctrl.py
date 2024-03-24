from dataclasses import dataclass

from fastapi import APIRouter, FastAPI
from v2 import commands
from v2.commands.processes.controllers.processes_ctrl.process_dto import ProcessDto
from v2.commands.processes.services.processes_svc import processes_svc_impl


@dataclass
class ProcessesCtrl:
    processes_svc = processes_svc_impl

    def router(self) -> APIRouter:
        app = FastAPI()

        @app.post(
            tags=[commands.__name__],
            summary="Run process",
            path="/images/{id}/process",
            status_code=204,
        )
        def _(id: int, dto: ProcessDto) -> None:
            self.processes_svc.run(id, dto)

        @app.post(
            tags=[commands.__name__],
            summary="Retry latest process",
            path="/images/{id}/process/retry",
            status_code=204,
        )
        def _(id: int) -> None:
            self.processes_svc.retry(id)

        @app.delete(
            tags=[commands.__name__],
            summary="Stop latest process",
            path="/images/{id}/process",
            status_code=204,
        )
        def _(id: int) -> None:
            self.processes_svc.stop(id)

        return app.router


processes_ctrl_impl = ProcessesCtrl()
