from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int
    device: str
    n_epochs: int

    train_path: str
    val_path: str

    model_save_folder: str
    model_save_name: str

    learning_rate: float
    gamma: float

    n_fft: int
    size: tuple[int, int]
