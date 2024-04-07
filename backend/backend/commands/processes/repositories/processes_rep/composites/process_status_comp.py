from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from backend.commands.processes.models.process_mod.process_status_val import (
    ProcessStatusVal,
)
from backend.helpers.exception_utils import ServerInternalErrorException


@dataclass
class ProcessStatusComp:
    value: ProcessStatusVal

    @classmethod
    def _generate(
        cls,
        status_started_at: datetime,
        status_ended_successful_at: Optional[datetime],
        status_ended_failed_at: Optional[datetime],
        status_ended_failed_error: Optional[str],
        status_ended_failed_stacktrace: Optional[str],
    ) -> "ProcessStatusComp":
        if status_ended_successful_at:
            ended = ProcessStatusVal.Successful(status_ended_successful_at)
        elif status_ended_failed_at and status_ended_failed_error:
            ended = ProcessStatusVal.Failed(
                status_ended_failed_at,
                status_ended_failed_error,
                status_ended_failed_stacktrace,
            )
        elif (
            status_ended_successful_at is None
            and status_ended_failed_at is None
            and status_ended_failed_error is None
        ):
            ended = None
        else:
            raise ServerInternalErrorException(
                f"Composite generation error: '{cls.__name__}'."
            )
        return ProcessStatusComp(ProcessStatusVal(status_started_at, ended))

    def __composite_values__(
        self,
    ) -> Tuple[
        datetime, Optional[datetime], Optional[datetime], Optional[str], Optional[str]
    ]:
        match self.value.ended:
            case None:
                return self.value.started_at, None, None, None, None
            case ProcessStatusVal.Successful(at=at):
                return self.value.started_at, at, None, None, None
            case ProcessStatusVal.Failed(at=at, error=error, stacktrace=stacktrace):
                return self.value.started_at, None, at, error, stacktrace
