import os
from dataclasses import dataclass


@dataclass
class LoggerConfig:
    log_file: os.PathLike | str = 'data/log_file.log'
    encoding: str = 'utf-8'
    level: str = 'INFO'
    format: str = "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
    date_format: str = "%d/%b/%Y %H:%M:%S"
