from dataclasses import dataclass

from src.config.classes import DatasetCreatorConfig, LoggerConfig, TrainConfig, ListenConfig, StaticAudioConfig


@dataclass
class Config:
    logger_config: LoggerConfig
    dataset_creator_config: DatasetCreatorConfig
    train_config: TrainConfig
    static_audio_config: StaticAudioConfig
    listen_config: ListenConfig
