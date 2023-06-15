import logging
import os

import torch
import torchaudio
import torchvision
from tqdm import tqdm

from src.config.classes import RecognizerConfig
from src.pipelines.abstract_pipeline import AbstractPipeline
from src.pipelines.train.models.resnet18_realization.model import Model


class HotWriteRecognizer(AbstractPipeline):
    """
    Pipeline for offline-recognizing of hot keys and keywords.
    """
    def __init__(self, config: RecognizerConfig, logger: logging.Logger, path_to_audio: str = None):
        """
        :param config: recognizer config. See src/config/classes/recognizer_config
        :param logger: logger object
        :param path_to_audio: path to static audio for running pipeline. This class can be used without it
        """
        self.config = config
        self.logger = logger

        self.path_to_audio = path_to_audio

        self.spectrogram = torchaudio.transforms.Spectrogram(self.config.n_ftt)
        self.resize = torchvision.transforms.Resize(self.config.size)
        self.sigmoid = torch.nn.Sigmoid()

        self.model = Model().eval()
        self.model.load_state_dict(torch.load(self.config.path_to_model))

    def run(self) -> None:
        """
        Start pipeline for processing offline audio
        """
        logging.info("Audio loading...")
        audio, sample_rate = torchaudio.load(self.path_to_audio)
        audio = audio[0]
        logging.info("Audio loaded!")

        self.process_audio(audio, sample_rate)

    def process_audio(self, audio: torch.Tensor, sample_rate: int, seconds_shift: float = 0.0) -> float:
        """
        Do full process for one audio.

        :param audio: audio as torch tensor
        :param sample_rate: number of samples per second
        :param seconds_shift: using for logging. Time will be logged with plus this value

        :return: current audio length in seconds
        """
        os.makedirs(self.config.save_to, exist_ok=True)

        progressbar = tqdm(
            total=len(audio) // sample_rate // 60,
            bar_format="{l_bar}{bar} | {n_fmt:.3}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )

        second = 0
        # second = 33 * 60 + 24

        with torch.no_grad():
            while second < len(audio) // sample_rate:
                start = int(second * sample_rate)
                end = int((second + self.config.secs_window) * sample_rate)
                fragment = audio[start:end]

                confidence, anti_confidence = self.predict_for_one_fragment(fragment)

                if (confidence > self.config.threshold) or (confidence > anti_confidence):
                    self.save_fragment(audio, second, sample_rate, confidence, anti_confidence, seconds_shift)

                second += self.config.secs_step
                progressbar.update(self.config.secs_step / 60)

                # if second > 33 * 60 + 30:
                #     break

        return len(audio) / sample_rate

    def predict_for_one_fragment(self, fragment: torch.Tensor) -> tuple[float, float]:
        """
        Got model confidence for current fragment

        :param fragment: current fragment

        :return: confidence for positive class, confidence for negative class
        """
        fragment = self.prepare_fragment(fragment)

        output = self.model(fragment)
        output = self.sigmoid(output)

        return output[0, 1], output[0, 0]

    def save_fragment(
        self,
        audio: torch.Tensor,
        second: float,
        sample_rate: int,
        confidence: float,
        anti_confidence: float,
        seconds_shift: float
    ) -> None:
        """
        Save fragment for given second. Saved fragment can be different with processed by model fragment. See configuration

        :param audio: full raw audio
        :param second: first second for which was found positive class
        :param sample_rate: number of samples per second
        :param confidence: model confidence for positive class
        :param anti_confidence: model confidence for negative class
        :param seconds_shift: using for logging. Will be added to second
        """
        start = int(second * sample_rate)
        end = int((second + self.config.save_audio_length) * sample_rate)
        wav = audio[start:end]

        second += seconds_shift
        in_minutes = f"{int(second // 60)}-{int(second % 60)}"
        self.logger.info(
            f"Found fragment! At {in_minutes}. Confidence: {confidence}. Anti confidence: {anti_confidence}"
        )
        try:
            torchaudio.save(
                os.path.join(self.config.save_to, f"{in_minutes}.wav"),
                wav.reshape((1, -1)),
                sample_rate
            )
        except Exception as e:
            logging.warning(e)

    def prepare_fragment(
        self,
        fragment: torch.Tensor
    ) -> torch.Tensor:
        """
        Translate audio to spectrogram and do some additional actions

        :param fragment: audio fragment as tensor

        :return: processed fragment
        """
        fragment = self.spectrogram(fragment)
        fragment = 10 * torch.log10(fragment)
        fragment = fragment[None, None]
        fragment = self.resize(fragment)
        return fragment
