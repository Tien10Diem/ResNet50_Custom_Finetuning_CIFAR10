import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

import torch.nn as nn
import torch
from Model.Layer import Layer

class ResNet50(nn.Module):
  def __init__(self, numbers =[3,4,6,3], nums_class = 10):
    super().__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias= False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
    )

    in_channels = [64, 256, 512, 1024]
    out_channels = [256, 512, 1024, 2048]
    strides = [1, 2, 2, 2]
    all_layers = []
    for i in range(len(numbers)):
      all_layers.append(Layer(in_channels[i], out_channels[i],strides[i], numbers[i]))
    self.all_layers = nn.Sequential(*all_layers)

    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.flatten = nn.Flatten()
    self.fc = nn.Linear(2048, nums_class)
  def forward(self, x):
    x = self.conv1(x)
    x = self.all_layers(x)
    x = self.avgpool(x)
    x = self.flatten(x)
    x = self.fc(x)
    return x

if __name__== "__main__":
    test = ResNet50()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = test(dummy_input)
    print(output.shape)