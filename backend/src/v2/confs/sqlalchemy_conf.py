import configparser
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session
from v2.confs.envs_conf import envs_conf_impl


@dataclass
class SqlAlchemyConf:
    envs_conf = envs_conf_impl

    class Base(DeclarativeBase):
        pass

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        config = configparser.ConfigParser()
        config.read("alembic.ini")
        with Session(
            create_engine(config["alembic"]["sqlalchemy.url"])
        ) as session, session.begin():
            yield session

    def migrate(self) -> None: ...


sqlalchemy_conf_impl = SqlAlchemyConf()
