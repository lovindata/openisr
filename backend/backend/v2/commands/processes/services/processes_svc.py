import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process
from queue import Queue
from threading import Thread
from typing import List, Tuple

from PIL.Image import Image
from sqlalchemy.orm import Session

from backend.v2.commands.images.models.image_mod import ImageMod
from backend.v2.commands.images.repositories.images_rep import images_rep_impl
from backend.v2.commands.processes.controllers.processes_ctrl.process_dto import (
    ProcessDto,
)
from backend.v2.commands.processes.models.process_mod.process_mod import ProcessMod
from backend.v2.commands.processes.repositories.processes_rep import processes_rep_impl
from backend.v2.commands.processes.services.image_processing_svc import (
    image_processing_svc_impl,
)
from backend.v2.confs.envs_conf import envs_conf_impl
from backend.v2.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from backend.v2.helpers.exception_utils import BadRequestException
from backend.v2.queries.app.repositories.card_download_rep import (
    card_downloads_rep_impl,
)
from backend.v2.queries.app.repositories.card_thumbnails_rep import (
    card_thumbnails_rep_impl,
)
from backend.v2.queries.app.repositories.cards_rep import cards_rep_impl


@dataclass
class ProcessesSvc:
    envs_conf = envs_conf_impl
    sqlalchemy_conf = sqlalchemy_conf_impl
    images_rep = images_rep_impl
    processes_rep = processes_rep_impl
    image_processing_svc = image_processing_svc_impl
    cards_rep = cards_rep_impl
    card_thumbnails_rep = card_thumbnails_rep_impl
    card_download_rep = card_downloads_rep_impl

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

        def get_image_and_create_process() -> Tuple[ImageMod, ProcessMod]:
            with self.sqlalchemy_conf.get_session() as session:
                image = self.images_rep.get(session, image_id)
                process = self.processes_rep.create_run_with_dto(session, image.id, dto)
                self.cards_rep.sync(session, image, process)
                return image, process

        raise_when_target_invalid()
        image, process = get_image_and_create_process()
        Process(
            target=self._pickable_process,
            args=(image, process),
            daemon=True,
        ).start()

    def retry(self, image_id: int) -> None:
        with self.sqlalchemy_conf.get_session() as session:
            image = self.images_rep.get(session, image_id)
            process_latest = self.processes_rep.get_latest_or_throw(session, image_id)
            process_latest = self.processes_rep.create_run_with_mod(
                session, image_id, process_latest
            )
            Process(
                target=self._pickable_process,
                args=(image, process_latest),
                daemon=True,
            ).start()
            self.cards_rep.sync(session, image, process_latest)

    def stop(self, image_id: int) -> None:
        with self.sqlalchemy_conf.get_session() as session:
            is_ended = (
                self.processes_rep.get_latest_or_throw(session, image_id).status.ended
                is not None
            )
            if is_ended:
                raise BadRequestException("Cannot stop an ended process.")
            self.processes_rep.delete(session, image_id)
            image = self.images_rep.get(session, image_id)
            latest_process = self.processes_rep.get_latest(session, image_id)
            self.cards_rep.sync(session, image, latest_process)

    def resolve_timeouts_if_exist(self, session: Session, image_ids: List[int]) -> None:
        process_latests = self.processes_rep.list_latest(session, image_ids)
        process_latests = [
            process_latest.resolve_timeout(self.envs_conf.process_timeout)
            for process_latest in process_latests
        ]
        images = {image.id: image for image in self.images_rep.list(session, image_ids)}
        for process_latest in process_latests:
            if process_latest.image_id:
                self.processes_rep.update(session, process_latest)
                self.cards_rep.sync(
                    session, images[process_latest.image_id], process_latest
                )

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
                    self.cards_rep.sync(session, image, updated_process)
                queue.put(None)

        def handle_result(out_image_data: Image | None) -> None:
            if out_image_data:
                with self.sqlalchemy_conf.get_session() as session:
                    updated_image = image.update_data(out_image_data)
                    self.images_rep.update(session, updated_image)
                    updated_process = process.terminate_success()
                    self.processes_rep.update(session, updated_process)
                    self.cards_rep.sync(session, updated_image, updated_process)
                    self.card_thumbnails_rep.sync(session, image)
                    self.card_download_rep.sync(session, image)

        queue: Queue[Image | None] = Queue()
        Thread(target=run_while_process_resumable, args=(queue,), daemon=True).start()
        Thread(target=run_process_image, args=(queue,), daemon=True).start()
        out_image_data = queue.get(
            timeout=self.envs_conf.process_timeout
        )  # Get first result only
        handle_result(out_image_data)


processes_svc_impl = ProcessesSvc()
