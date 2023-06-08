from src.config.classes import (
    Config,
    DatasetCreatorConfig,
    LoggerConfig,
    TrainConfig, ListenConfig
)

config = Config(
    dataset_creator_config=DatasetCreatorConfig(
        raw_audio="data/raw_data/thanos_message.wav",
        hotkey_folder="data/raw_data/stones",
        dataset_size=3000,
        output_folder="data/dataset",
        train_val_split_k=0.7
    ),

    train_config=TrainConfig(
        batch_size=1,
        device='cpu',
        train_path='data/dataset/train.csv',
        val_path='data/dataset/val.csv',
        learning_rate=1e-05,
        n_fft=1023
    ),

    listen_config=ListenConfig(
        url="http://radio.maslovka-home.ru/thanosshow",
        start_time="3:59",
        retries=100,
        sleep_time=10,
        output="data/night_stream.wav"
    ),

    logger_config=LoggerConfig(
        log_file='data/log_file.log',
        encoding='utf-8',
        level='INFO',
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        date_format="%d/%b/%Y %H:%M:%S"
    )
)
