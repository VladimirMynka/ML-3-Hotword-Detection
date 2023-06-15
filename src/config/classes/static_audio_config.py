from dataclasses import dataclass
from src.config.classes import RecognizerConfig


@dataclass
class StaticAudioConfig:
    path_to_audio: str
    recognizer: RecognizerConfig
