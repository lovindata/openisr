import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process
from queue import Queue
from threading import Thread
from time import time
from typing import Tuple

from PIL.Image import Image
from v2.commands.images.models.image_mod import ImageMod
from v2.commands.images.repositories.images_rep import images_rep_impl
from v2.commands.processes.controllers.processes_cmd.process_dto import ProcessDto
from v2.commands.processes.models.process_mod import ProcessMod
from v2.commands.processes.repositories.processes_rep import processes_rep_impl
from v2.commands.processes.services.image_processing_svc import (
    image_processing_svc_impl,
)
from v2.confs.envs_conf import envs_conf_impl
from v2.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from v2.helpers.exception_utils import BadRequestException


@dataclass
class ProcessesSvc:
    envs_conf = envs_conf_impl
    sqlalchemy_conf = sqlalchemy_conf_impl
    images_rep = images_rep_impl
    processes_rep = processes_rep_impl
    image_processing_svc = image_processing_svc_impl

    def run(self, image_id: int, dto: ProcessDto) -> None:
        def raise_when_target_invalid() -> None:
            if (
                dto.target.width > 1920
                or dto.target.width < 0
                or dto.target.height > 1920
                or dto.target.height < 0
            ):
                raise BadRequestException(
                    f"Invalid target: ({dto.target.width}, {dto.target.height})."
                )

        def start_process() -> None:
            with self.sqlalchemy_conf.get_session() as session:
                image = self.images_rep.get(session, image_id)
                process = self.processes_rep.create_run_with_dto(session, image.id, dto)
                Process(
                    target=self._pickable_process,
                    args=(image, process),
                    daemon=True,
                ).start()

        raise_when_target_invalid()
        start_process()

    def retry(self, image_id: int) -> None:
        def get_image_and_recreate_latest_process(
            image_id: int,
        ) -> Tuple[ImageMod, ProcessMod]:
            with self.sqlalchemy_conf.get_session() as session:
                image = self.images_rep.get(session, image_id)
                process_latest = self.processes_rep.get_latest_or_throw(
                    session, image_id
                )
                process_latest = self.processes_rep.create_run_with_mod(
                    session, image_id, process_latest
                )
                return (image, process_latest)

        image, process_latest = get_image_and_recreate_latest_process(image_id)
        Process(
            target=self._pickable_process,
            args=(image, process_latest),
            daemon=True,
        ).start()

    def stop(self, id: int) -> None:
        with self.sqlalchemy_conf.get_session() as session:
            is_ended = (
                self.processes_rep.get_latest_or_throw(session, id).status.ended
                is not None
            )
            if is_ended:
                raise BadRequestException("Cannot stop an ended process.")
            self.processes_rep.delete(session, id)

    def _pickable_process(self, image: ImageMod, process: ProcessMod) -> None:
        def run_while_process_resumable(queue: Queue[Image | None]) -> None:
            resumable = True
            while resumable:
                time.sleep(1)
                with self.sqlalchemy_conf.get_session() as session:
                    uptodate_process = self.processes_rep.get_latest(session, image.id)
                    resumable = (
                        uptodate_process is not None
                        and uptodate_process.status.ended is None
                    )
            queue.put(None)

        def run_process_image(queue: Queue[Image | None]) -> None:
            try:
                out_image_data = self.image_processing_svc.process_image(
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
                with self.sqlalchemy_conf.get_session() as session:
                    updated_process = process.terminate_failed(error, stacktrace)
                    self.processes_rep.update(session, updated_process)
                queue.put(None)

        def handle_result(out_image_data: Image | None) -> None:
            if out_image_data:
                with self.sqlalchemy_conf.get_session() as session:
                    updated_image = image.update_data(out_image_data)
                    self.images_rep.update(session, updated_image)
                    updated_process = process.terminate_success()
                    self.processes_rep.update(session, updated_process)

        queue: Queue[Image | None] = Queue()
        Thread(target=run_while_process_resumable, args=(queue,), daemon=True).start()
        Thread(target=run_process_image, args=(queue,), daemon=True).start()
        out_image_data = queue.get(
            timeout=self.envs_conf.process_timeout
        )  # Get first result only
        handle_result(out_image_data)


processes_svc_impl = ProcessesSvc()
