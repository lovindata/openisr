from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from v2.commands.images.controllers.images_cmd import images_cmd_impl
from v2.commands.processes.controllers.processes_cmd import processes_cmd_impl
from v2.confs.envs_conf import envs_conf_impl
from v2.helpers.exception_utils import BadRequestException
from v2.queries.app.controllers.app_qry import app_qry_impl


@dataclass
class FastAPIConf:
    envs_conf = envs_conf_impl
    images_cmd = images_cmd_impl
    processes_cmd = processes_cmd_impl
    app_qry = app_qry_impl

    _app = FastAPI(title="OpenISR")

    def run(self) -> None:
        self._set_allow_cors_if_dev()
        self._set_exception_handler()
        self._app.include_router(self.images_cmd.router())
        self._app.include_router(self.processes_cmd.router())
        self._app.include_router(self.app_qry.router())
        self._run_server()

    def _set_allow_cors_if_dev(self) -> None:
        if not self.envs_conf.prod_mode:
            self._app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

    def _set_exception_handler(self) -> None:
        @self._app.exception_handler(Exception)
        async def _(_: Request, e: Exception):
            headers = (
                {
                    "access-control-allow-credentials": "true",
                    "access-control-allow-origin": "*",
                }
                if not self.envs_conf.prod_mode
                else {}
            )
            if isinstance(e, BadRequestException):
                return JSONResponse(
                    status_code=400,
                    content={"detail": e.args[0]},
                    headers=headers,
                )
            elif not self.envs_conf.prod_mode:
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

    def _run_server(self) -> None:
        if self.envs_conf.prod_mode:
            uvicorn.run(
                app=self._app,
                host="0.0.0.0",
                port=self.envs_conf.api_port,
                log_level="info",
                access_log=False,
            )
        else:
            uvicorn.run(
                app="drivers.fastapi_driver:_",
                host="0.0.0.0",
                port=self.envs_conf.api_port,
                log_level="info",
                access_log=False,
                reload=True,
                reload_dirs="./src",
            )


fastapi_conf_impl = FastAPIConf()
_ = None if envs_conf_impl.prod_mode else fastapi_conf_impl._app
