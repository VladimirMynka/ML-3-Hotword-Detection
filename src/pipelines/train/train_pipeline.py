import os
from logging import Logger

import pandas as pd
import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.pipelines.abstract_pipeline import AbstractPipeline
from src.pipelines.train.models.resnet18_realization.model import Model
from src.pipelines.train.wav_dataset import WavDataset
from src.config.classes import TrainConfig


class TrainPipeline(AbstractPipeline):
    """
    Train pipeline
    """
    def __init__(self, config: TrainConfig, logger: Logger):
        """
        :param config: train config. See src/config/classes/train_pipeline
        :param logger: logger object
        """
        self.config = config
        self.logger = logger

    def run(self) -> None:
        """
        Run this pipeline. Load data, initialize model, train its, save its
        """
        train_loader, val_loader = self.load_data()
        model = Model().to(self.config.device)
        self.train(model, train_loader, val_loader)

    def load_data(self) -> tuple[DataLoader, DataLoader]:
        """
        Load and prepare data

        :return: train data loader, validation data loader
        """
        train_path, val_path = self.config.train_path, self.config.val_path

        self.logger.info("Loading data...")

        train_df, val_df = [pd.read_csv(path) for path in [train_path, val_path]]
        train_dataset, val_dataset = [
            WavDataset(df, self.config.n_fft, self.config.size, self.logger)
            for df in [train_df.iloc[:], val_df.iloc[:]]
        ]

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        self.logger.info("Data loaded!")

        return train_loader, val_loader

    @staticmethod
    def train_epoch(
        train_loader: DataLoader,
        model: nn.Module,
        criterion: nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ExponentialLR
    ) -> tuple[float, float]:
        """
        Train process for one epoch

        :param train_loader: data loader with train data
        :param model: training model. Returns two classes
        :param criterion: loss function
        :param optimizer: using optimizer
        :param scheduler: Exponential scheduler for changing learning_rate while training

        :return: train loss, train accuracy
        """
        model.train()
        epoch_acc = 0
        epoch_loss = 0

        i = 0

        for batch in tqdm(train_loader):
            input_, label = batch
            output = model(input_)

            loss = criterion(output, label)
            loss.backward()

            i += 1

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += float(torch.sum(torch.argmax(output, dim=1) == label))

        scheduler.step()

        torch.cuda.empty_cache()

        dataset_length = len(train_loader.dataset)

        epoch_loss /= dataset_length
        epoch_acc /= dataset_length

        return epoch_loss, epoch_acc

    @staticmethod
    def val_epoch(
        val_loader: DataLoader,
        model: nn.Module,
        criterion: nn.CrossEntropyLoss
    ) -> tuple[float, float]:
        """
        Evaluate epoch while training

        :param val_loader: DataLoader with validation data
        :param model: training (and evaluating) model
        :param criterion: loss function

        :return: validation loss, validation accuracy
        """
        model.eval()
        epoch_acc = 0
        epoch_loss = 0

        for batch in tqdm(val_loader):
            input_, label = batch
            output = model(input_)

            loss = criterion(output, label)

            epoch_loss += loss.item()
            epoch_acc += float(torch.sum(torch.argmax(output, dim=1) == label))

        dataset_length = len(val_loader.dataset)

        epoch_loss /= dataset_length
        epoch_acc /= dataset_length

        return epoch_loss, epoch_acc

    def train(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """
        Full train process

        :param model: training model
        :param train_loader: data loader with train data
        :param val_loader: data loader with validation data
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config.gamma)

        best_acc = 0

        for epoch in range(self.config.n_epochs):
            self.logger.info(f"Epoch {epoch}...")

            loss, acc = self.train_epoch(train_loader, model, criterion, optimizer, scheduler)
            self.logger.info(f"TRAIN. Accuracy: {acc}, Loss: {loss}")

            with torch.no_grad():
                loss, acc = self.val_epoch(val_loader, model, criterion)
                self.logger.info(f"VALIDATION. Accuracy: {acc}, Loss: {loss}")

                if acc > best_acc:
                    best_acc = acc
                    self.save_model(model)

    def save_model(self, model: nn.Module):
        """
        Save model and log it

        :param model: saving model
        """
        os.makedirs(self.config.model_save_folder, exist_ok=True)

        self.logger.info("Saving model...")

        torch.save(model.state_dict(), os.path.join(self.config.model_save_folder, self.config.model_save_name))

        self.logger.info("Model saved!")
