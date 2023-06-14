from dataclasses import dataclass


@dataclass
class StaticAudioConfig:
    path_to_audio: str
    secs_window: float
    secs_step: float
    save_audio_length: float
    threshold: float
    save_to: str

    path_to_model: str
    n_ftt: int
    size: tuple[int, int]
