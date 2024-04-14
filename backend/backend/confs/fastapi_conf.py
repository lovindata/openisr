from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from backend.commands.images.controllers.images_ctrl import images_ctrl_impl
from backend.commands.processes.controllers.processes_ctrl.processes_ctrl import (
    processes_ctrl_impl,
)
from backend.commands.processes.services.timeout_resolver_svc import (
    timeout_resolver_svc_impl,
)
from backend.confs.envs_conf import envs_conf_impl
from backend.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from backend.helpers.exception_utils import BadRequestException
from backend.queries.app.controllers.app_ctrl import app_ctrl_impl


@dataclass
class FastAPIConf:
    envs_conf = envs_conf_impl
    images_cmd = images_ctrl_impl
    processes_cmd = processes_ctrl_impl
    app_qry = app_ctrl_impl
    sqlalchemy_conf = sqlalchemy_conf_impl
    timeout_resolver_svc = timeout_resolver_svc_impl

    def __post_init__(self) -> None:
        self._app = FastAPI(title="OpenISR", lifespan=self._lifespan)
        self._set_allow_cors_if_dev()
        self._set_exception_handler()
        self._app.include_router(self.app_qry.router())
        self._app.include_router(self.images_cmd.router())
        self._app.include_router(self.processes_cmd.router())
        self._set_frontend_distribuable_if_prod()

    def run_server(self) -> None:
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
                app=f"{self.__module__}:_",  # Must pass the application as an import string (https://www.uvicorn.org/deployment/#running-programmatically)
                host="0.0.0.0",
                port=self.envs_conf.api_port,
                log_level="info",
                access_log=False,
                reload=True,
                reload_dirs="./backend",
            )

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
                error = (
                    str(e.args[0])
                    if e.args
                    else "Apologies, an unknown error occurred. Please retry later."
                )
                return JSONResponse(
                    status_code=500,
                    content={"detail": error},
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

    def _set_frontend_distribuable_if_prod(self) -> None:
        if self.envs_conf.prod_mode:
            self._app.mount(
                # Order matters! Must be executed after the actual routes!
                # The order of route definitions determines the sequence of request handling.
                "/",
                StaticFiles(directory="../frontend/dist", html=True),
            )

            @self._app.exception_handler(404)
            async def _(*_):
                return FileResponse("../frontend/dist/index.html")

    @asynccontextmanager
    async def _lifespan(self, _: FastAPI) -> AsyncGenerator[None, None]:
        logger.info("Starting database migration.")
        self.sqlalchemy_conf.migrate()  # Auto-migrate on reloading
        logger.info("Starting process timeout cron resolver.")
        self.timeout_resolver_svc.run_cron()  # Threaded execution, link to FastAPI lifespan necessary
        yield


fastapi_conf_impl = FastAPIConf()
_ = (
    None if envs_conf_impl.prod_mode else fastapi_conf_impl._app
)  # For reloads, FastAPI should be accessible in the main context.
