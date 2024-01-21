import os


class EnvLoaderDriver:
    def __init__(self) -> None:
        self.prod_mode = os.getenv("OPENISR_PROD_MODE", False) == "True"
        self.api_port = int(os.getenv("OPENISR_API_PORT", 8000))


env_laoder_driver_impl = EnvLoaderDriver()
