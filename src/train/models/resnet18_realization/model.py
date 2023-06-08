import torch
from torch import nn

import src.train.models.resnet18_realization.model_layers as layers


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.image_decoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            layers.BaseLayer(64, 64),
            layers.BaseLayer(64, 64),

            layers.BaseLayerWithDownSample(64, 128),
            layers.BaseLayer(128, 128),

            layers.BaseLayerWithDownSample(128, 256),
            layers.BaseLayer(256, 256),

            layers.BaseLayerWithDownSample(256, 512),
            layers.BaseLayer(512, 512),

            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        self.classifier = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor):
        x = self.image_decoder(x)
        x = torch.squeeze(x, (2, 3))
        return self.classifier(x)
