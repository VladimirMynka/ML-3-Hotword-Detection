from logging import Logger

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram
from torchvision.transforms import Resize


class WavDataset(Dataset):
    def __init__(self, df: pd.DataFrame, n_fft: int, logger: Logger):
        self.df = df
        self.spectrogram = Spectrogram(n_fft)
        self.resize = Resize(size=(224, 224))

        test_example, _ = self[0]
        logger.info(f"Dataset creation succeed. Output shape: {test_example.shape}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[index]
        path = row['path']
        label = row['label']
        wave, sample_rate = torchaudio.load(path)
        wave = wave[0]
        wave = self.spectrogram(wave)
        wave = 10 * torch.log10(wave)
        wave = wave[None]
        # wave = wave.repeat(3, 1, 1)
        wave = self.resize(wave)
        return wave, label
