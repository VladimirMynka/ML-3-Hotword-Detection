from dataclasses import dataclass

from src.config.classes import RecognizerConfig


@dataclass
class ListenConfig:
    url: str
    start_time: str
    retries: int
    sleep_time: float

    recognizer: RecognizerConfig
