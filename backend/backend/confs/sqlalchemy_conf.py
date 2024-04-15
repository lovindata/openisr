from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

from alembic.command import upgrade
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.confs.envs_conf import envs_conf_impl


@dataclass
class SqlAlchemyConf:
    envs_conf = envs_conf_impl

    def __post_init__(self) -> None:
        self._config = Config("alembic.ini")
        self._engine = create_engine(
            self._config.file_config.get("alembic", "sqlalchemy.url")
        )
        self._session_factory = sessionmaker(
            self._engine, autoflush=True
        )  # https://docs.sqlalchemy.org/en/20/orm/session_basics.html#flushing

    class Base(DeclarativeBase):
        pass

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        with self._session_factory.begin() as session:
            yield session

    def migrate(self) -> None:
        with self._engine.begin() as connection:
            self._config.attributes["connection"] = connection
            upgrade(self._config, "head")


sqlalchemy_conf_impl = SqlAlchemyConf()
