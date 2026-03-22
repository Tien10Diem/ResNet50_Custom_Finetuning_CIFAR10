import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from Model.Botneck import BotNeck
import torch.nn as nn
import torch

class Layer(nn.Module):
  def __init__(self, in_channels, out_channels, stride, number):
    super().__init__()
    layer=[]
    layer.append(BotNeck(in_channels, out_channels, stride))

    for i in range(number-1):
      layer.append(BotNeck(out_channels, out_channels, 1))

    self.layer = nn.Sequential(*layer)
  def forward(self,x):
    x = self.layer(x)
    return x

if __name__ == "__main__":
    test = Layer(3, 64, 1, 3)
    dummy_input = torch.randn(1, 3, 32, 32)
    output = test(dummy_input)
    print(output.shape)