import torch.nn as nn
import torch

class BotNeck(nn.Module):
  def __init__(self,input_channels, output_channels, stride):
    super().__init__()

    botneck = output_channels//4
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels= input_channels, out_channels= botneck, kernel_size=1, padding=0, stride=1, bias= False),
        nn.BatchNorm2d(botneck),
        nn.ReLU(),
        nn.Conv2d(in_channels= botneck, out_channels= botneck, kernel_size=3, padding=1, stride=stride, bias= False),
        nn.BatchNorm2d(botneck),
        nn.ReLU(),
        nn.Conv2d(in_channels= botneck, out_channels= output_channels, kernel_size=1, padding=0, stride=1, bias= False),
        nn.BatchNorm2d(output_channels),
    )

    self.shortcut = nn.Identity()

    if stride != 1 or input_channels != output_channels:
      self.shortcut = nn.Sequential(
          nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride, bias= False),
          nn.BatchNorm2d(output_channels)
      )
    self.relu = nn.ReLU()

  def forward(self, x):
    x_shortcut = self.shortcut(x)
    x = self.conv(x)
    x = x + x_shortcut
    x = self.relu(x)
    return x


if __name__ == "__main__":
    test = BotNeck(3, 64, 1)
    dummy_input = torch.randn(1, 3, 32, 32)
    output = test(dummy_input)
    print(output.shape)