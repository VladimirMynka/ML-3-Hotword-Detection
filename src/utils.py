from src.config.classes import LoggerConfig
import logging


def init_logging(config: LoggerConfig = None):
    if config is None:
        config = LoggerConfig()
    logging.basicConfig(
        filename=config.log_file,
        encoding=config.encoding,
        level=config.level,
        format=config.format,
        datefmt=config.date_format
    )
