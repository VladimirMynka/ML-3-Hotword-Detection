from dataclasses import dataclass
from src.config.classes import DatasetCreatorConfig, LoggerConfig, TrainConfig, ListenConfig


@dataclass
class Config:
    logger_config: LoggerConfig
    dataset_creator_config: DatasetCreatorConfig
    train_config: TrainConfig
    listen_config: ListenConfig
