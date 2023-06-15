import torch
from torch import nn

from src.train.models.own_created_model import model_layers as layers


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.image_decoder = nn.Sequential(                     # (201, 81)
            layers.ConvPoolAndConvNorm(1, 64),                  # (100, 40)
            layers.ConvPoolAndConvNorm(64, 128),                # (50, 20)
            layers.ConvPoolAndConvNorm(128, 128),               # (25, 10)
            layers.ConvPoolAndConvNorm(128, 256),               # (12, 5)
            layers.ConvPoolAndConvNorm(256, 512),               # (6, 2)
            layers.ConvPoolAndConvNorm(256, 512),               # (4, 2)
        )
        self.classifier = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor):
        return self.classifier(self.image_decoder(x))
