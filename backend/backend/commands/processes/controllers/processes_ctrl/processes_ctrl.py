from dataclasses import dataclass

from fastapi import APIRouter, FastAPI

from backend import commands
from backend.commands.processes.controllers.processes_ctrl.process_dto import ProcessDto
from backend.commands.processes.services.processes_svc import processes_svc_impl


@dataclass
class ProcessesCtrl:
    processes_svc = processes_svc_impl

    def router(self) -> APIRouter:
        app = FastAPI()

        @app.post(
            tags=[commands.__name__],
            summary="Run process",
            path="/commands/v1/images/{image_id}/process/run",
            status_code=204,
        )
        def _(image_id: int, dto: ProcessDto) -> None:
            self.processes_svc.run(image_id, dto)

        @app.post(
            tags=[commands.__name__],
            summary="Retry latest process",
            path="/commands/v1/images/{image_id}/process/retry",
            status_code=204,
        )
        def _(image_id: int) -> None:
            self.processes_svc.retry(image_id)

        @app.delete(
            tags=[commands.__name__],
            summary="Stop latest process",
            path="/commands/v1/images/{image_id}/process/stop",
            status_code=204,
        )
        def _(image_id: int) -> None:
            self.processes_svc.stop(image_id)

        return app.router


processes_ctrl_impl = ProcessesCtrl()
