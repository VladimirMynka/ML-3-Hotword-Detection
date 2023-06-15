from src.config.classes import (
    Config,
    DatasetCreatorConfig,
    LoggerConfig,
    TrainConfig, ListenConfig, StaticAudioConfig, RecognizerConfig
)

recognizer_config = RecognizerConfig(
    secs_step=0.2,
    secs_window=1.0,
    save_audio_length=2.2,
    threshold=0.05,
    save_to="data/output/static_audio/",

    path_to_model="data/model/model.pt",
    n_ftt=1023,
    size=(224, 224)
)

config = Config(
    dataset_creator_config=DatasetCreatorConfig(
        raw_audio="data/raw_data/thanos_message.wav",
        hotkey_folder="data/raw_data/stones",
        dataset_size=30000,
        output_folder="data/dataset",
        train_val_split_k=0.7,
        shift_coefficient=0.25
    ),

    train_config=TrainConfig(
        batch_size=1,
        device='cpu',
        n_epochs=3,

        train_path='data/dataset/train.csv',
        val_path='data/dataset/val.csv',

        model_save_folder='data/model',
        model_save_name='model.pt',

        learning_rate=1e-05,
        gamma=0.98,

        n_fft=1023,
        size=(224, 224)
    ),

    listen_config=ListenConfig(
        url="http://radio.maslovka-home.ru/thanosshow",
        start_time="3:59",
        retries=100,
        sleep_time=10,
        recognizer=recognizer_config
    ),

    static_audio_config=StaticAudioConfig(
        path_to_audio="data/raw_data/thanos_message.wav",
        # path_to_audio="data/night_stream.wav",
        recognizer=recognizer_config
    ),

    logger_config=LoggerConfig(
        log_file='data/log_file.log',
        encoding='utf-8',
        level='INFO',
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        date_format="%d/%b/%Y %H:%M:%S"
    )
)
