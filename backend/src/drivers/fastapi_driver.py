import uvicorn
from adapters.controllers.images_ctrl import ImagesCtrl, images_ctrl_impl
from adapters.controllers.processes_ctrl import ProcessesCtrl, processes_ctrl_impl
from drivers.os_env_loader_driver import OsEnvLoaderDriver, os_env_laoder_driver_impl
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from helpers.exception_utils import BadRequestException


class FastApiDriver:
    def __init__(
        self,
        images_ctrl: ImagesCtrl,
        processes_ctrl: ProcessesCtrl,
        env_loader_driver: OsEnvLoaderDriver,
    ) -> None:
        self.images_ctrl = images_ctrl
        self.processes_ctrl = processes_ctrl
        self.env_loader_driver = env_loader_driver

        self.app = FastAPI(title="OpenISR")
        self._set_allow_cors_if_dev()
        self._initalize_exception_handler()
        self.app.include_router(self.images_ctrl.router())
        self.app.include_router(self.processes_ctrl.router())

    def run(self) -> None:
        if self.env_loader_driver.prod_mode:
            uvicorn.run(
                app=self.app,
                host="localhost",
                port=self.env_loader_driver.api_port,
                log_level="info",
                access_log=False,
            )
        else:
            uvicorn.run(
                app="drivers.fastapi_driver:_",
                host="localhost",
                port=self.env_loader_driver.api_port,
                log_level="info",
                access_log=False,
                reload=True,
                reload_dirs="./src",
            )

    def _set_allow_cors_if_dev(self) -> None:
        if not self.env_loader_driver.prod_mode:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

    def _initalize_exception_handler(self) -> None:
        @self.app.exception_handler(Exception)
        async def _(_: Request, e: Exception):
            headers = (
                {
                    "access-control-allow-credentials": "true",
                    "access-control-allow-origin": "*",
                }
                if not self.env_loader_driver.prod_mode
                else {}
            )
            if isinstance(e, BadRequestException):
                return JSONResponse(
                    status_code=400,
                    content={"detail": e.args[0]},
                    headers=headers,
                )
            elif not self.env_loader_driver.prod_mode:
                return JSONResponse(
                    status_code=500,
                    content={"detail": e.args[0]},
                    headers=headers,
                )
            else:
                return JSONResponse(
                    status_code=500,
                    content={
                        "detail": "Please try again later. If the issue persists, contact support."
                    },
                    headers=headers,
                )


fastapi_driver_impl = FastApiDriver(
    images_ctrl_impl, processes_ctrl_impl, os_env_laoder_driver_impl
)
_ = None if os_env_laoder_driver_impl.prod_mode else fastapi_driver_impl.app
