import time
from dataclasses import dataclass
from threading import Thread
from typing import List

from loguru import logger
from sqlalchemy.orm import Session

from backend.v2.commands.images.repositories.images_rep import images_rep_impl
from backend.v2.commands.processes.models.process_mod.process_mod import ProcessMod
from backend.v2.commands.processes.repositories.processes_rep import processes_rep_impl
from backend.v2.confs.envs_conf import envs_conf_impl
from backend.v2.confs.sqlalchemy_conf import sqlalchemy_conf_impl
from backend.v2.queries.app.repositories.cards_rep import cards_rep_impl


@dataclass
class TimeoutResolverSvc:
    envs_conf = envs_conf_impl
    sqlalchemy_conf = sqlalchemy_conf_impl
    images_rep = images_rep_impl
    processes_rep = processes_rep_impl
    cards_rep = cards_rep_impl

    def run_cron(self) -> None:
        Thread(target=self._run_indefinetely, daemon=True).start()

    def _run_indefinetely(self) -> None:
        def resolve_and_get_timed_out_processes(session: Session) -> List[ProcessMod]:
            processes_running = self.processes_rep.list_running(session)
            processes_resolved = [
                process.resolve_timeout(self.envs_conf.process_timeout)
                for process in processes_running
            ]
            processes_timed_out = [
                process
                for process in processes_resolved
                if process not in processes_running
            ]
            for process in processes_timed_out:
                self.processes_rep.update(session, process)
                logger.info(
                    f"Process ID={process.id} will be resolved as timed out after transaction."
                )
            return processes_timed_out

        def sync_cards(session: Session, processes_updated: List[ProcessMod]) -> None:
            image_ids = set(
                [process.image_id for process in processes_updated if process.image_id]
            )
            for image_id in image_ids:
                image = self.images_rep.get_or_raise(session, image_id)
                latest_process = self.processes_rep.get_latest_or_raise(
                    session, image_id
                )
                self.cards_rep.sync(session, image, latest_process)

        with sqlalchemy_conf_impl.get_session() as session:
            processes_timed_out = resolve_and_get_timed_out_processes(session)
            sync_cards(session, processes_timed_out)
        time.sleep(self.envs_conf.process_timeout)
        self._run_indefinetely()


timeout_resolver_svc_impl = TimeoutResolverSvc()
