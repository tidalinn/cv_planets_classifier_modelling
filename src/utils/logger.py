import logging
from logging.handlers import RotatingFileHandler

from src.constants import PATH_LOGS


class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level):
        super().__init__()
        self.max_level = max_level

    def filter(self, record):
        return record.levelno <= self.max_level


def init_rotating_handler(
    file_name: str,
    level: int,
    formatter: logging.Formatter,
    logger: logging.Logger,
    max_bytes: int = 5 * 1024 * 1024,  # noqa: WPS404
    backup_count: int = 5,
    level_filter: logging.Filter = None
) -> RotatingFileHandler:

    logs_handler = RotatingFileHandler(
        filename=f'{PATH_LOGS}/{file_name}.log',
        maxBytes=max_bytes,
        backupCount=backup_count
    )

    logs_handler.setLevel(level)
    logs_handler.setFormatter(formatter)

    if level_filter:
        logs_handler.addFilter(level_filter)

    logger.addHandler(logs_handler)


def init_logger() -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    init_rotating_handler('info', logging.INFO, formatter, logger, level_filter=MaxLevelFilter(logging.WARNING))
    init_rotating_handler('error', logging.ERROR, formatter, logger)
