class BadRequestException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ServerInternalErrorException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
