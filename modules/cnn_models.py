import torch
from torch import nn

class TinyVGG(nn.Module):
  def __init__(self,
               in_feature: int,
               hidden_units: int,
               output_features: int,
               remaining_dimensions: int):
    super().__init__()
    self.conv_block1 = nn.Sequential(
        nn.Conv2d(in_channels=in_feature,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2))
    self.conv_block2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2))
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*remaining_dimensions*remaining_dimensions,
                  out_features=output_features)
    )

  def forward(self, x):
    return self.classifier(
        self.conv_block2(self.conv_block1(x)))


class VGG11architecture(nn.Module):
  def __init__(self):
    super().__init__()
    self.convblock1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64,
                  kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        # nn.Conv2d(in_channels=64, out_channels=64,
        #           kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=2))

    self.convblock2 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128,
                  kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        # nn.Conv2d(in_channels=128, out_channels=128,
        #           kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=2))

    self.convblock3 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=256,
                  kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256,
                  kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        # nn.Conv2d(in_channels=256, out_channels=256,
        #           kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=2))

    self.convblock4 = nn.Sequential(
        nn.Conv2d(in_channels=256, out_channels=512,
                  kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=512, out_channels=512,
                  kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        # nn.Conv2d(in_channels=512, out_channels=512,
        #           kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=2))

    self.convblock5 = nn.Sequential(
        nn.Conv2d(in_channels=512, out_channels=512,
                  kernel_size=3, stride=1, padding=1),
        nn.Conv2d(in_channels=512, out_channels=512,
                  kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        # nn.Conv2d(in_channels=512, out_channels=512,
        #           kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=2))

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=256*28*28, out_features=4096),
        nn.ReLU(),
        # nn.Linear(in_features=4096, out_features=4096),
        # nn.ReLU(),
        nn.Linear(in_features=4096, out_features=5),
        nn.ReLU()
    )

  def forward(self, x):
    return self.classifier(
        self.convblock5(
            self.convblock4(
                self.convblock3(
                    self.convblock2(
                        self.convblock1(x))))))