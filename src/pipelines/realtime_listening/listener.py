import logging
import time
from datetime import datetime
from logging import Logger

import requests
from tqdm import tqdm

from src.config.classes import ListenConfig


class Listener:
    def __init__(self, config: ListenConfig, logger: Logger):
        self.response = None
        self.config = config
        self.logger = logger
        self.time_to_start = config.start_time

        self.is_waiting = True
        self.connected = False

    def run(self):
        while not self.is_waiting:
            self.check_time()
        if not self.connected:
            self.response = self.try_connect()
            if not self.connected:
                self.logger.error("Cannot connect")
                raise ConnectionError("Cannot connect")

        output_file = self.config.output
        with open(output_file, 'w') as f:
            f.write("")

        with open(output_file, 'ab') as f:
            for chunk in tqdm(self.response.iter_content(chunk_size=8192)):
                f.write(chunk)

    def check_time(self):
        current_time = datetime.now()
        current_time = current_time.strftime("%H:%M")
        logging.info(f"Current time: {current_time}")
        if current_time == self.time_to_start:
            self.is_waiting = False

    def try_connect(self):
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
