import logging
import os
import time
from datetime import datetime
from logging import Logger

import requests
import torchaudio.backend.no_backend
from tqdm import tqdm

from src.config.classes import ListenConfig
from src.pipelines.abstract_pipeline import AbstractPipeline
from src.pipelines.static_audio_processing import HotWriteRecognizer


class Listener(AbstractPipeline):
    """
    Pipeline for connecting to stream and real-time hot-keys detection
    """
    def __init__(self, config: ListenConfig, logger: Logger):
        """
        :param config: special config for this pipeline. See src.config.classes.listen_config
        :param logger: logger object
        """
        self.response = None
        self.config = config
        self.logger = logger
        self.time_to_start = config.start_time

        self.is_waiting = True
        self.connected = False

        self.recognizer = HotWriteRecognizer(config.recognizer, logger)

    def run(self) -> None:
        """
        Start listen pipeline. Try to connect while it cant then listen audio with using HotWriteRecognizer
        """
        while self.is_waiting:
            self.check_time()
        if not self.connected:
            self.response = self.try_connect()
            if not self.connected:
                self.logger.error("Cannot connect")
                raise ConnectionError("Cannot connect")

        current_second = 0
        for chunk in tqdm(self.response.iter_content(chunk_size=8192)):
            with open("temp.wav") as f:
                f.write(chunk)
            audio, sample_rate = torchaudio.load("temp.wav")
            audio = audio[0]

            current_second += self.recognizer.process_audio(audio, sample_rate, current_second)

        os.remove("temp.wav")

    def check_time(self) -> None:
        """
        Check is current time is time to start listen or not. If is then change flag
        """
        current_time = datetime.now()
        current_time = current_time.strftime("%H:%M")
        logging.info(f"Current time: {current_time}")
        if current_time == self.time_to_start:
            self.is_waiting = False

    def try_connect(self) -> requests.Response:
        """
        Try to connect defined number of times

        :return: connection
        """
        response = requests.get(self.config.url, stream=True)
        retries_count = 0
        while response.status_code == 404:
            retries_count += 1
            logging.warning(f"Retry {retries_count}...")
            time.sleep(self.config.sleep_time)
            response = requests.get(self.config.url, stream=True)
            if retries_count >= self.config.retries:
                self.connected = False
                return response

        self.connected = True
        return response
