import os

from usecases.drivers.env_loader_driver import EnvLoaderDriver


class OsEnvLoaderDriver(EnvLoaderDriver):
    @property
    def prod_mode(self) -> bool:
        return os.getenv("OPENISR_PROD_MODE", False) == "True"

    @property
    def api_port(self) -> int:
        return int(os.getenv("OPENISR_API_PORT", 8000))

    @property
    def process_timeout(self) -> int:
        return int(os.getenv("OPENISR_PROCESS_TIMEOUT", 8000))


os_env_laoder_driver_impl = OsEnvLoaderDriver()
