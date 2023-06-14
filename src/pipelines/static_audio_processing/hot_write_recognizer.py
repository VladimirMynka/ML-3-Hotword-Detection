import logging
import os
from tqdm import tqdm

import torch
import torchaudio
import torchvision

from src.pipelines.abstract_pipeline import AbstractPipeline
from src.config.classes import StaticAudioConfig
from src.pipelines.train.models.resnet18_realization.model import Model


class HotWriteRecognizer(AbstractPipeline):
    def __init__(self, config: StaticAudioConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def run(self) -> None:
        logging.info("Audio loading...")
        audio, sample_rate = torchaudio.load(self.config.path_to_audio)
        model = Model().eval()
        model.load_state_dict(torch.load(self.config.path_to_model))

        spectrogram = torchaudio.transforms.Spectrogram(self.config.n_ftt)
        resize = torchvision.transforms.Resize(self.config.size)
        sigmoid = torch.nn.Sigmoid()

        os.makedirs(self.config.save_to, exist_ok=True)

        audio = audio[0]
        logging.info("Audio loaded!")

        progressbar = tqdm(
            total=len(audio) // sample_rate // 60,
            bar_format="{l_bar}{bar} | {n_fmt:.3}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )

        second = 0
        with torch.no_grad():
            while second < len(audio) // sample_rate:
                second = 33 * 60 + 26
                start = int(second * sample_rate)
                end = int((second + self.config.secs_window) * sample_rate)
                fragment = audio[start:end]
                fragment = self.prepare_fragment(fragment, spectrogram, resize)

                output = model(fragment)
                output = sigmoid(output)

                if output[0, 1] > self.config.threshold:
                    in_minutes = f"{int(second // 60)}-{int(second % 60)}"
                    self.logger.info(
                        f"Found fragment! At {in_minutes}. Confidence: {output[0, 1]}"
                        f"\nAnother confidence: {output[0, 0]}"
                    )
                    start = int(second * sample_rate)
                    end = int((second + self.config.save_audio_length) * sample_rate)
                    wav = audio[start:end]
                    try:
                        torchaudio.save(
                            os.path.join(self.config.save_to, f"{in_minutes}.wav"),
                            wav.reshape((1, -1)),
                            sample_rate
                        )
                    except Exception as e:
                        logging.warning(e)

                    break

                second += self.config.secs_step
                progressbar.update(self.config.secs_step / 60)

    @staticmethod
    def prepare_fragment(
        fragment: torch.Tensor,
        spectrogram: torchaudio.transforms.Spectrogram,
        resize: torchvision.transforms.Resize
    ) -> torch.Tensor:
        fragment = spectrogram(fragment)
        fragment = 10 * torch.log10(fragment)
        fragment = fragment[None, None]
        fragment = resize(fragment)
        return fragment
