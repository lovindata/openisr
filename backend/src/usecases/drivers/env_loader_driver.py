from abc import ABC, abstractmethod


class EnvLoaderDriver(ABC):
    @property
    @abstractmethod
    def prod_mode(self) -> bool:
        pass

    @property
    @abstractmethod
    def api_port(self) -> int:
        pass
