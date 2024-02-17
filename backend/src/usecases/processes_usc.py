import time
import traceback
from multiprocessing import Process
from queue import Queue
from threading import Thread
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
from entities.shared.extension_val import ExtensionVal
from helpers.exception_utils import BadRequestException
from PIL.Image import Image
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

        def get_image_and_create_process() -> Tuple[ImageEnt, ProcessEnt]:
            with self.db_driver.get_session() as session:
                image = self.images_rep.get(session, image_id)
                process = self.processes_rep.create_run(
                    session,
                    image.id,
                    ExtensionVal(extension),
                    target_width,
                    target_height,
                    enable_ai,
                )
            return image, process

        raise_when_target_invalid()
        image, process = get_image_and_create_process()
        Process(target=self._pickable_process_task, args=(image, process)).start()
        return process

    def get_latest_process(self, image_id: int) -> ProcessEnt | None:
        with self.db_driver.get_session() as session:
            process_latest = self.processes_rep.get_latest(session, image_id)
            if process_latest:
                process_latest = process_latest.resolve_timeout(
                    self.env_loader_driver.process_timeout
                )
                process_latest = self.processes_rep.update(session, process_latest)
        return process_latest

    def retry(self, image_id: int) -> ProcessEnt:
        def get_image_and_recreate_latest_process(
            image_id: int,
        ) -> Tuple[ImageEnt, ProcessEnt]:
            with self.db_driver.get_session() as session:
                image = self.images_rep.get(session, image_id)
                process_latest = self.processes_rep.get_latest_or_throw(
                    session, image_id
                )
                process_latest = self.processes_rep.create_run(
                    session,
                    image_id,
                    process_latest.extension,
                    process_latest.target.width,
                    process_latest.target.height,
                    process_latest.enable_ai,
                )
                return (image, process_latest)

        image, process_latest = get_image_and_recreate_latest_process(image_id)
        Process(
            target=self._pickable_process_task, args=(image, process_latest)
        ).start()
        return process_latest

    def stop(self, id: int) -> ProcessEnt:
        with self.db_driver.get_session() as session:
            is_ended = (
                self.processes_rep.get_latest_or_throw(session, id).status.ended
                is not None
            )
            if is_ended:
                raise BadRequestException("Cannot stop an ended process.")
            process_stopped = self.processes_rep.delete(session, id)
        return process_stopped

    def _pickable_process_task(self, image: ImageEnt, process: ProcessEnt) -> None:
        def run_while_process_resumable(queue: Queue[Image | None]) -> None:
            resumable = True
            while resumable:
                time.sleep(1)
                with self.db_driver.get_session() as session:
                    uptodate_process = self.processes_rep.get_latest(session, image.id)
                    resumable = (
                        uptodate_process is not None
                        and uptodate_process.status.ended is None
                    )
            queue.put(None)

        def run_process_image(queue: Queue[Image | None]) -> None:
            try:
                out_image_data = self.image_processing_driver.process_image(
                    image.data, process
                )
                queue.put(out_image_data)
            except Exception as e:
                error = (
                    str(e.args[0])
                    if e.args
                    else "Apologies, an unknown error occurred. Please retry later."
                )
                stacktrace = traceback.format_exc()
                with self.db_driver.get_session() as session:
                    updated_process = process.terminate_failed(error, stacktrace)
                    self.processes_rep.update(session, updated_process)
                queue.put(None)

        def handle_result(out_image_data: Image | None) -> None:
            if out_image_data:
                with self.db_driver.get_session() as session:
                    updated_image = image.update_data(out_image_data)
                    self.images_rep.update(session, updated_image)
                    updated_process = process.terminate_success()
                    self.processes_rep.update(session, updated_process)

        queue: Queue[Image | None] = Queue()
        Thread(target=run_while_process_resumable, args=(queue,)).start()
        Thread(target=run_process_image, args=(queue,)).start()
        out_image_data = queue.get(
            timeout=self.env_loader_driver.process_timeout
        )  # Get first result only
        handle_result(out_image_data)


processes_usc_impl = ProcessesUsc(
    os_env_laoder_driver_impl,
    sqlalchemy_db_driver_impl,
    opcv_pillow_image_processing_driver_impl,
    sqlalchemy_images_rep_impl,
    sqlalchemy_processes_rep_impl,
)
