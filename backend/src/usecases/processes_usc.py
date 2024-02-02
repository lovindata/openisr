import asyncio
from typing import Literal, Tuple

from adapters.repositories.sqlalchemy_images_rep import sqlalchemy_images_rep_impl
from adapters.repositories.sqlalchemy_processes_rep import sqlalchemy_processes_rep_impl
from drivers.opcv_pillow_image_processing_driver import (
    opcv_pillow_image_processing_driver_impl,
)
from drivers.os_env_loader_driver import os_env_laoder_driver_impl
from drivers.sqlalchemy_db_driver import sqlalchemy_db_driver_impl
from entities.image_ent import ImageEnt
from entities.process_ent import ProcessEnt
from helpers.exception_utils import BadRequestException
from usecases.drivers.db_driver import DbDriver
from usecases.drivers.env_loader_driver import EnvLoaderDriver
from usecases.drivers.image_processing_driver import ImageProcessingDriver
from usecases.repositories.images_rep import ImagesRep
from usecases.repositories.processes_rep import ProcessesRep


class ProcessesUsc:
    def __init__(
        self,
        env_loader_driver: EnvLoaderDriver,
        db_driver: DbDriver,
        image_processing_driver: ImageProcessingDriver,
        images_rep: ImagesRep,
        processes_rep: ProcessesRep,
    ) -> None:
        self.env_loader_driver = env_loader_driver
        self.db_driver = db_driver
        self.image_processing_driver = image_processing_driver
        self.images_rep = images_rep
        self.processes_rep = processes_rep

    def run(
        self,
        image_id: int,
        extension: Literal["JPEG", "PNG", "WEBP"],
        preserve_ratio: bool,
        target_width: int,
        target_height: int,
        enable_ai: bool,
    ) -> ProcessEnt:
        def raise_when_target_invalid() -> None:
            if (
                target_width > 9999
                or target_width < 0
                or target_height > 9999
                or target_height < 0
            ):
                raise BadRequestException(
                    f"Invalid target: ({target_width}, {target_height})."
                )

        def create_process() -> Tuple[ImageEnt, ProcessEnt]:
            with self.db_driver.get_session() as session:
                image = self.images_rep.get(session, image_id)
                process = self.processes_rep.create_run(
                    session,
                    image.id,
                    extension,
                    preserve_ratio,
                    target_width,
                    target_height,
                    enable_ai,
                )
            return image, process

        async def run_process() -> None:
            out_image_data = self.image_processing_driver.process_image(
                image.data, process
            )
            with self.db_driver.get_session() as session:
                updated_image = image.update_data(out_image_data, process.extension)
                self.images_rep.update(session, updated_image)
                updated_process = process.terminate_sucessfully()
                self.processes_rep.update(session, updated_process)

        raise_when_target_invalid()
        image, process = create_process()
        asyncio.run(run_process())  # TODO BAD IT HAS TO RUN A DEDICATED PROCESS
        return process


processes_usc_impl = ProcessesUsc(
    os_env_laoder_driver_impl,
    sqlalchemy_db_driver_impl,
    opcv_pillow_image_processing_driver_impl,
    sqlalchemy_images_rep_impl,
    sqlalchemy_processes_rep_impl,
)
