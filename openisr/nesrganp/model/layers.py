import torch
import torch.nn as nn

class GaussianNoise(nn.Module):
    def __init__(self):
        super(GaussianNoise, self).__init__()
        self.gamma = 0.1 # The gamma defined is the paper

    def forward(self, x):
        if self.training:
          x_scale = self.gamma * x
          x = torch.normal(mean=x, std=x_scale)
        return x

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.noise = GaussianNoise()
        self.conv1x1 = nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(160, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        self.beta = 0.2 # The beta defined in the paper

    def forward(self, x):
        x1 = self.conv1(x) # 64 -> 32
        x2 = self.conv2(torch.cat((x, x1), 1)) # 64 + 32 -> 32, concatenation along the channels
        x2 = x2 + self.conv1x1(x) # 32 -> 32, the conv1x1 is used to get same size
        x3 = self.conv3(torch.cat((x, x1, x2), 1)) # 64 + 32 + 32 -> 32
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1)) # 64 + 32 + 32 + 32 -> 32
        x4 = x4 + x2
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) # 64 + 32 + 32 + 32 + 32 -> 64
        return self.noise(x5.mul(self.beta) + x) # 64 -> 64, the skip connection defined is the paper

class ResidualDenseBlock(nn.Module):
    def __init__(self):
        super(ResidualDenseBlock, self).__init__()
        self.RDB1 = ResidualInResidualDenseBlock()
        self.RDB2 = ResidualInResidualDenseBlock()
        self.RDB3 = ResidualInResidualDenseBlock()
        self.noise = GaussianNoise()
        self.beta = 0.2

    def forward(self, x):
        out = self.RDB1(x) # 64 -> 64
        out = self.RDB2(out) # 64 -> 64
        out = self.RDB3(out) # 64 -> 64
        return self.noise(out.mul(self.beta) + x) # 64 -> 64
    
class ResidualDenseBlocks(nn.Module):
    def __init__(self):
        super(ResidualDenseBlocks, self).__init__()
        self.sub = nn.Sequential(
            *[ResidualDenseBlock() for _ in range(23)],
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    
    def forward(self, x):
        return x + self.sub(x) # 64 -> 64