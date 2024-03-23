import os
from dataclasses import dataclass


@dataclass
class EnvsConf:
    prod_mode = os.getenv("OPENISR_PROD_MODE", "False") == "True"
    api_port = int(os.getenv("OPENISR_API_PORT", "8000"))
    process_timeout = int(os.getenv("OPENISR_PROCESS_TIMEOUT_IN_SECONDS", "60"))


envs_conf_impl = EnvsConf()
