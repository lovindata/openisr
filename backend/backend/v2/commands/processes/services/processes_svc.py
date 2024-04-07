import time
import traceback
from dataclasses import dataclass
from multiprocessing import Process
from queue import Empty, Queue
from threading import Thread

from PIL.Image import Image

from backend.v2.commands.images.models.image_mod import ImageMod
from backend.v2.commands.images.repositories.images_rep import images_rep_impl
from backend.v2.commands.processes.controllers.processes_ctrl.process_dto import (
    ProcessDto,
)
from backend.v2.commands.processes.models.process_mod.process_mod import ProcessMod
from backend.v2.commands.processes.repositories.processes_rep.processes_rep import (
    processes_rep_impl,
)
from backend.v2.commands.processes.services.image_processing_svc import (
    image_processing_svc_impl,
)
from backend.v2.confs.envs_conf import envs_conf_impl
from backend.v2.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from backend.v2.helpers.exception_utils import BadRequestException
from backend.v2.queries.app.repositories.card_downloads_rep import (
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
        def raise_when_target_invalid(image: ImageMod) -> None:
            match dto.scaling:
                case ProcessDto.Bicubic(
                    width=width, height=height
                ) if 0 > width or width > 1920 and 0 > height or height > 1920:
                    raise BadRequestException(
                        f"Target ({width}, {height}) too large (max 1920x1920)."
                    )
                case ProcessDto.AI():
                    source_width, source_height = image.data.size
                    if source_width > 480 or source_height > 480:
                        raise BadRequestException(
                            f"Image ({source_width}, {source_height}) too large for AI upscaling (max 480x480)."
                        )

        with self.sqlalchemy_conf.get_session() as session:
            image = self.images_rep.get_or_raise(session, image_id)
            raise_when_target_invalid(image)
            process = self.processes_rep.create_run_with_dto(session, image, dto)
            self.cards_rep.sync(session, image, process)
        Process(
            target=self._pickable_process,
            args=(image, process),
            daemon=True,
        ).start()

    def retry(self, image_id: int) -> None:
        with self.sqlalchemy_conf.get_session() as session:
            image = self.images_rep.get_or_raise(session, image_id)
            process_latest = self.processes_rep.get_latest_or_raise(session, image_id)
            process_latest = self.processes_rep.create_run_with_mod(
                session, image, process_latest
            )
            Process(
                target=self._pickable_process,
                args=(image, process_latest),
                daemon=True,
            ).start()
            self.cards_rep.sync(session, image, process_latest)

    def stop(self, image_id: int) -> None:
        with self.sqlalchemy_conf.get_session() as session:
            latest_process = self.processes_rep.get_latest_or_raise(session, image_id)
            if latest_process.status.ended:
                raise BadRequestException("Cannot stop an ended process.")
            self.processes_rep.delete(session, latest_process.id)
            image = self.images_rep.get_or_raise(session, image_id)
            latest_process = self.processes_rep.get_latest(session, image_id)
            self.cards_rep.sync(session, image, latest_process)

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

        def handle_result(queue: Queue[Image | None]) -> None:
            try:
                out_image_data = queue.get(
                    timeout=self.envs_conf.process_timeout
                )  # Get first result only
                if out_image_data:
                    with self.sqlalchemy_conf.get_session() as session:
                        updated_image = image.update_data(out_image_data)
                        self.images_rep.update(session, updated_image)
                        updated_process = process.terminate_success()
                        self.processes_rep.update(session, updated_process)
                        self.cards_rep.sync(session, updated_image, updated_process)
                        self.card_thumbnails_rep.sync(session, updated_image)
                        self.card_download_rep.sync(session, updated_image)
            except Empty:
                with self.sqlalchemy_conf.get_session() as session:
                    updated_process = process.terminate_failed_timed_out(
                        self.envs_conf.process_timeout
                    )
                    self.processes_rep.update(session, updated_process)
                    self.cards_rep.sync(session, image, updated_process)

        queue: Queue[Image | None] = Queue()
        Thread(target=run_while_process_resumable, args=(queue,), daemon=True).start()
        Thread(target=run_process_image, args=(queue,), daemon=True).start()
        handle_result(queue)


processes_svc_impl = ProcessesSvc()
