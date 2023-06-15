import os
import random
from logging import Logger

import pandas as pd
import torch
import torchaudio
from tqdm import trange

from src.pipelines.abstract_pipeline import AbstractPipeline
from src.config.classes import DatasetCreatorConfig


class PrepareDatasetPipeline(AbstractPipeline):
    """
    Pipeline of dataset preparation. Can be started with run function.
    """
    def __init__(self, config: DatasetCreatorConfig, logger: Logger):
        """
        :param config: special config for this pipeline. See src.config.classes.dataset_creator_config
        :param logger: logger object
        """
        self.config = config
        self.logger = logger

        self.hotkey_folder = config.hotkey_folder
        self.raw_audio = config.raw_audio
        self.dataset_size = config.dataset_size
        self.output_folder = config.output_folder
        self.k = config.train_val_split_k

    def run(self) -> None:
        """
        Run dataset preparation pipeline. It's configuring by config.
        You will get data/dataset/wavs and data/dataset/labels.csv
        """
        try:
            wave, sample_rate = torchaudio.load(self.raw_audio)
            wave = wave[0]
        except Exception as e:
            self.logger.error(e)
            raise e

        wavs_folder = os.path.join(self.output_folder, "wavs")
        os.makedirs(wavs_folder)

        labels = []
        paths = []
        self.logger.info(f"Form {self.dataset_size} data files")
        for i in trange(self.dataset_size):
            wav, label = self.get_sample(wave, sample_rate)
            path = os.path.join(wavs_folder, f"{i}.wav")

            paths.append(path)
            labels.append(label)

            torchaudio.save(path, wav.reshape((1, -1)), sample_rate)

        pd.DataFrame({
            "path": paths[:int(self.k * self.dataset_size)],
            "label": labels[:int(self.k * self.dataset_size)]
        }).to_csv(os.path.join(self.output_folder, "train.csv"), index=False)

        pd.DataFrame({
            "path": paths[int(self.k * self.dataset_size):],
            "label": labels[int(self.k * self.dataset_size):]
        }).to_csv(os.path.join(self.output_folder, "val.csv"), index=False)

        self.logger.info(f"Data saved: {self.output_folder}")

    def get_sample(self, raw_wave: torch.Tensor, sample_rate: int = 16000) -> tuple[torch.Tensor, int]:
        """
        Cuts randomly a piece of the audio and add (or doesn't add) randomly a hot-word piece

        :param raw_wave: raw audio file from which will be got a piece
        :param sample_rate: number of samples per second

        :return: processed piece and label: 0 or 1
        """
        one_second = self.get_random_piece(raw_wave, sample_rate, seconds=1)
        do_apply_hotkeys = (random.random() > 0.5)
        if do_apply_hotkeys:
            hotkey = self.get_random_hotkey_sound()
            max_shift = int(self.config.shift_coefficient * len(hotkey))
            shift = random.randint(-max_shift, max_shift)
            one_second = self.apply_hotkey(one_second, hotkey, shift)

        return one_second, int(do_apply_hotkeys)

    def apply_hotkey(self, one_second_wave: torch.Tensor, hotkey_wave: torch.Tensor, shift: int) -> torch.Tensor:
        """
        Apply hot word piece to raw one-second piece with defined shift

        :param one_second_wave: piece of the raw audio
        :param hotkey_wave: audio with hot key
        :param shift: shift as samples count

        :return: processed audio
        """
        hotkey = torch.zeros(len(one_second_wave))
        hotkey_real = self.shift_tensor(hotkey_wave, shift)
        hotkey[:len(hotkey_real)] = hotkey_real
        return one_second_wave + hotkey

    @staticmethod
    def get_random_piece(raw_audio: torch.Tensor, sample_rate: int, seconds: int = 1) -> torch.Tensor:
        """
        Cuts randomly audio

        :param raw_audio: raw audio tensor
        :param sample_rate: samples number per second
        :param seconds: size of new piece

        :return: piece of the audio
        """
        start_i = random.randint(0, len(raw_audio) - seconds * sample_rate)
        end_i = start_i + sample_rate * seconds
        return raw_audio[start_i:end_i]

    def get_random_hotkey_sound(self) -> torch.Tensor:
        """
        Choice randomly one file with hot-key

        :return: chosen audio as tensor
        """
        files = [os.path.join(self.hotkey_folder, file) for file in os.listdir(self.hotkey_folder)]
        i = random.randint(0, len(files) - 1)
        wave, sample_rate = torchaudio.load(files[i])
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        wave = resampler(wave)
        wave = wave[0]
        return wave[:16000]  # remove hard-code

    @staticmethod
    def shift_tensor(tensor: torch.Tensor, shift: int) -> torch.Tensor:
        """
        Applies shift to tensor

        :param tensor: input tensor
        :param shift: positive or negative

        :return: shifted tensor
        """
        if shift >= 0:
            return torch.concat([torch.zeros(shift), torch.roll(tensor, shift)[shift:]])
        return torch.concat([torch.roll(tensor, shift)[:shift], torch.zeros(-shift)])
