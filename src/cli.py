import logging

from fire import Fire
from src.config.config import config
from src.pipelines.realtime_listening.listener import Listener
from src.pipelines.train.train_pipeline import TrainPipeline
from src.pipelines.utils import init_logging
from src.pipelines.static_audio_processing import HotWriteRecognizer

from src.pipelines.dataset_preparation.prepare_dataset_pipeline import PrepareDatasetPipeline


class Cli:
    """
    Service input. All pipelines at one place
    """
    def __init__(self):
        """
        Init logging for all pipelines
        """
        init_logging(config.logger_config)
        self.logger = logging.getLogger()

    def prepare_dataset(self):
        """
        Extracts one-second pieces of audio from given .wav-file and generate from them dataset.
        Put it into data/dataset
        """
        pipeline = PrepareDatasetPipeline(config.dataset_creator_config, self.logger)
        pipeline.run()

    def get_stream(self):
        """
        Connects to given online-stream and detect hot keys from it in real-time
        """
        pipeline = Listener(config.listen_config, logger=self.logger)
        pipeline.run()

    def train(self):
        """
        Train hot-words recognize model
        """
        pipeline = TrainPipeline(config.train_config, logger=self.logger)
        pipeline.run()

    def static_audio_process(self):
        """
        Detect hot keys from stativ audio file
        """
        pipeline = HotWriteRecognizer(
            config.static_audio_config.recognizer,
            logger=self.logger,
            path_to_audio=config.static_audio_config.path_to_audio
        )
        pipeline.run()


if __name__ == "__main__":
    Fire(Cli)
