import configparser
from contextlib import contextmanager
from typing import Generator

from drivers.os_env_loader_driver import OsEnvLoaderDriver, os_env_laoder_driver_impl
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from usecases.drivers.db_driver import DbDriver


class SqlAlchemyDbDriver(DbDriver):
    def __init__(self, env_loader_driver: OsEnvLoaderDriver) -> None:
        self.env_loader_driver = env_loader_driver

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        config = configparser.ConfigParser()
        config.read("alembic.ini")
        with Session(
            create_engine(config["alembic"]["sqlalchemy.url"])
        ) as session, session.begin():
            yield session

    def migrate(self) -> None:
        ...


sqlalchemy_db_driver_impl = SqlAlchemyDbDriver(os_env_laoder_driver_impl)
