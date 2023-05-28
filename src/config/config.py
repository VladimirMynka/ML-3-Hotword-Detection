from src.config.classes import (
    Config,
    DatasetCreatorConfig,
    LoggerConfig
)

config = Config(
    dataset_creator_config=DatasetCreatorConfig(
        raw_audio="data/raw_data/thanos_message.wav",
        hotkey_folder="data/raw_data/stones",
        dataset_size=3000,
        output_folder="data/dataset",
        train_val_split_k=0.7
    ),

    logger_config=LoggerConfig(
        log_file='data/log_file.log',
        encoding='utf-8',
        level='INFO',
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        date_format="%d/%b/%Y %H:%M:%S"
    )
)
