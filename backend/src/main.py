from loguru import logger
from v2.confs.fastapi_conf import fastapi_conf_impl


@logger.catch
def main() -> None:
    logger.info("Starting application.")
    fastapi_conf_impl.run()


if __name__ == "__main__":
    main()
