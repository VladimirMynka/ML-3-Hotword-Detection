import logging

from fire import Fire
from src.config.config import config
from src.realtime_listening.listener import Listener
from src.utils import init_logging

from src.dataset_preparation.prepare_dataset_pipeline import PrepareDatasetPipeline


class Pipelines:
    def __init__(self):
        init_logging(config.logger_config)
        self.logger = logging.getLogger()

    def prepare_dataset(self):
        pipeline = PrepareDatasetPipeline(config.dataset_creator_config, self.logger)
        pipeline.run()

    def get_stream(self):
        pipeline = Listener(config.listen_config, logger=self.logger)
        pipeline.run()


if __name__ == "__main__":
    Fire(Pipelines)
