
import torch
import torch.nn as nn
from torch.onnx import TrainingMode

class GaussianNoise(nn.Module):
    def __init__(self):
        super(GaussianNoise, self).__init__()
        self.stddev = 0.1

    def forward(self, x):
        if self.training:
            x_scale = self.stddev * x
            x = torch.normal(mean=x, std=x_scale)
        return x
    
    # YOU ARE HERE!!!! TRYING TO FIGURE WHY NORMAL GENERATOR IS NOT WORKING ON GOOGLE COLAB
    # commit test

class ResidualDenseBlock(nn.Module):

    def __init__(self) -> None:
        super(ResidualDenseBlock, self).__init__()

        self.noise = GaussianNoise()