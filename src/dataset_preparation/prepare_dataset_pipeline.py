import os
import random
from logging import Logger

import pandas as pd
import torch
import torchaudio
from tqdm import trange

from src.abstract_pipeline import AbstractPipeline
from src.config.classes import DatasetCreatorConfig


class PrepareDatasetPipeline(AbstractPipeline):
    def __init__(self, config: DatasetCreatorConfig, logger: Logger):
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

    def get_sample(self, raw_wave, sample_rate=16000):
        one_second = self.get_random_piece(raw_wave, sample_rate, seconds=1)
        do_apply_hotkeys = (random.random() > 0.5)
        if do_apply_hotkeys:
            hotkey = self.get_random_hotkey_sound()
            shift = random.randint(-len(hotkey) // 2, len(hotkey) // 2)
            one_second = self.apply_hotkey(one_second, hotkey, shift)

        return one_second, int(do_apply_hotkeys)

    def apply_hotkey(self, one_second_wave, hotkey_wave, shift):
        hotkey = torch.zeros(len(one_second_wave))
        hotkey_real = self.shift_tensor(hotkey_wave, shift)
        hotkey[:len(hotkey_real)] = hotkey_real
        return one_second_wave + hotkey

    @staticmethod
    def get_random_piece(raw_audio, sample_rate, seconds=1):
        start_i = random.randint(0, len(raw_audio) - seconds * sample_rate)
        end_i = start_i + sample_rate * seconds
        return raw_audio[start_i:end_i]

    def get_random_hotkey_sound(self):
        files = [os.path.join(self.hotkey_folder, file) for file in os.listdir(self.hotkey_folder)]
        i = random.randint(0, len(files) - 1)
        wave, sample_rate = torchaudio.load(files[i])
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        wave = resampler(wave)
        wave = wave[0]
        return wave[:16000]

    @staticmethod
    def shift_tensor(tensor, shift):
        if shift >= 0:
            return torch.concat([torch.zeros(shift), torch.roll(tensor, shift)[shift:]])
        return torch.concat([torch.roll(tensor, shift)[:shift], torch.zeros(-shift)])
