import logging

from fire import Fire
from src.config.config import config
from src.pipelines.realtime_listening.listener import Listener
from src.pipelines.train.train_pipeline import TrainPipeline
from src.pipelines.utils import init_logging

from src.pipelines.dataset_preparation.prepare_dataset_pipeline import PrepareDatasetPipeline


class Pipelines:
    def __init__(self):
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
        Connects to given online-stream and write it into data/night_stream.wav file
        """
        pipeline = Listener(config.listen_config, logger=self.logger)
        pipeline.run()

    def train(self):
        """
        Train hot-words recognize model
        :return:
        """
        pipeline = TrainPipeline(config.train_config, logger=self.logger)
        pipeline.run()


if __name__ == "__main__":
    Fire(Pipelines)
