from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int
    device: str

    train_path: str
    val_path: str

    learning_rate: float

    n_fft: int
