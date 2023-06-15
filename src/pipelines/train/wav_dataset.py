from logging import Logger

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram
from torchvision.transforms import Resize


class WavDataset(Dataset):
    """
    Dataset for comfortable getting audio files as tensors and their labels
    """
    def __init__(self, df: pd.DataFrame, n_fft: int, size: tuple[int, int], logger: Logger):
        """
        :param df: dataframe with paths to audio files and labels
        :param n_fft: using for transformation audio into spectrogram
        :param size: size of image for spectrogram
        :param logger: logger object
        """
        self.df = df
        self.spectrogram = Spectrogram(n_fft)
        self.resize = Resize(size)

        test_example, _ = self[0]
        logger.info(f"Dataset creation succeed. Output shape: {test_example.shape}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Transform an audio and return its spectrogram

        :param index: index of sample in dataset

        :return: spectrogram, label
        """
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
