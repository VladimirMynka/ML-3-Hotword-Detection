import os
from dataclasses import dataclass


@dataclass
class DatasetCreatorConfig:
    dataset_size: int
    raw_audio: str | os.PathLike
    hotkey_folder: str | os.PathLike

    output_folder: str | os.PathLike
    train_val_split_k: float
    shift_coefficient: float
