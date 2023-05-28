from dataclasses import dataclass
from src.config.classes import DatasetCreatorConfig, LoggerConfig


@dataclass
class Config:
    logger_config: LoggerConfig
    dataset_creator_config: DatasetCreatorConfig
