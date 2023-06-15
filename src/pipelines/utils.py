import logging

from src.config.classes import LoggerConfig


def init_logging(config: LoggerConfig = None):
    """
    Configure logger for this project

    :param config: config for logging
    """
    if config is None:
        config = LoggerConfig()
    logging.basicConfig(
        filename=config.log_file,
        encoding=config.encoding,
        level=config.level,
        format=config.format,
        datefmt=config.date_format
    )
