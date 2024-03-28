import configparser
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from v2.confs.envs_conf import envs_conf_impl


@dataclass
class SqlAlchemyConf:
    envs_conf = envs_conf_impl

    def __post_init__(self) -> None:
        config = configparser.ConfigParser()
        config.read("alembic.ini")
        engine = create_engine(config["alembic"]["sqlalchemy.url"])
        self._session_factory = sessionmaker(
            engine, autoflush=True
        )  # https://docs.sqlalchemy.org/en/20/orm/session_basics.html#flushing

    class Base(DeclarativeBase):
        pass

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        with self._session_factory.begin() as session:
            yield session

    def migrate(self) -> None: ...


sqlalchemy_conf_impl = SqlAlchemyConf()
