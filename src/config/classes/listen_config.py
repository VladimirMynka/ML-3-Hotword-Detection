import os
from dataclasses import dataclass


@dataclass
class ListenConfig:
    url: str
    start_time: str
    retries: int
    sleep_time: float
    output: str | os.PathLike
