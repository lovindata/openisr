from drivers.fastapi_driver import fastapi_driver_impl
from loguru import logger


@logger.catch
def main() -> None:
    logger.info("Starting application.")
    fastapi_driver_impl.run()


if __name__ == "__main__":
    main()
