import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    """Class containing the attributes and methods to work with the EDSR model.

    Attributes:
        gamma (float): The gamma related to the noise and defined in the nESRGAN+ paper.
    """
    
    def __init__(self):
        super(GaussianNoise, self).__init__()
        self.gamma = 0.1 # The gamma defined in the paper

    def forward(self, x):
        """The method to apply the GaussianNoise on `x`.

        Args:
            x (Tensor): The input Tensor.

        Returns:
            Tensor: Output tensor with the GaussianNoise applied.
        """
        
        if self.training:
          x_scale = self.gamma * x
          x = torch.normal(mean=x, std=x_scale)
        return x

class ResidualInResidualDenseBlock(nn.Module):
    """The RRDB basic building block used in nESRGAN+.

    Note:
        Please see the original paper ESRGAN+ Fig.2 (b) for more architecture details.

    Attributes:
        noise (GaussianNoise): Gaussian noise not useful only here because of the actual architecture when training.
        conv1x1 (Conv2d): Conv2D 64/32c no bias for the first skip connection.
        conv1 (Sequential): Conv2D 64/32c with bias and LeakyReLU.
        conv2 (Sequential): Conv2D 96/32c with bias and LeakyReLU.
        conv3 (Sequential): Conv2D 128/32c with bias and LeakyReLU.
        conv4 (Sequential): Conv2D 160/32c with bias and LeakyReLU.
        conv5 (Sequential): Conv2D 192/64c with bias and LeakyReLU.
        beta (float): Scaling factor for conv5 output.
    """
    
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
        self.beta = 0.2

    def forward(self, x):
        """The method to apply the RRDB on `x` (64/64c).

        Args:
            x (Tensor): The input tensor with 64c.

        Returns:
            Tensor: Output tensor 64c with the RRDB applied.
        """
        
        x1 = self.conv1(x)                                  # 64 -> 32
        x2 = self.conv2(torch.cat((x, x1), 1))              # 64 + 32 -> 32 (concatenation along the channels)
        x2 = x2 + self.conv1x1(x)                           # 32 -> 32 (first skip connection and the conv1x1 is used to get same size)
        x3 = self.conv3(torch.cat((x, x1, x2), 1))          # 64 + 32 + 32 -> 32
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))      # 64 + 32 + 32 + 32 -> 32
        x4 = x4 + x2                                        # 32 -> 32 (second skip connection)
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))  # 64 + 32 + 32 + 32 + 32 -> 64
        return self.noise(x5.mul(self.beta) + x)            # 64 -> 64 (third skip connection with a scaling factor)

class ResidualDenseBlock(nn.Module):
    """The RDB advanced building block used in nESRGAN+.

    Attributes:
        RDB1 (ResidualInResidualDenseBlock): RRDB 64/64c.
        RDB2 (ResidualInResidualDenseBlock): RRDB 64/64c.
        RDB3 (ResidualInResidualDenseBlock): RRDB 64/64c.
        noise (GaussianNoise): Gaussian noise not useful only here because of the actual architecture when training.
        beta (float): Scaling factor for RDB3 output.
    """
    
    def __init__(self):
        super(ResidualDenseBlock, self).__init__()
        self.RDB1 = ResidualInResidualDenseBlock()
        self.RDB2 = ResidualInResidualDenseBlock()
        self.RDB3 = ResidualInResidualDenseBlock()
        self.noise = GaussianNoise()
        self.beta = 0.2

    def forward(self, x):
        """The method to apply the RDB on `x` (64/64c).

        Args:
            x (Tensor): The input tensor with 64c.

        Returns:
            Tensor: Output tensor 64c with the RDB applied.
        """
        
        out = self.RDB1(x)                          # 64 -> 64
        out = self.RDB2(out)                        # 64 -> 64
        out = self.RDB3(out)                        # 64 -> 64
        return self.noise(out.mul(self.beta) + x)   # 64 -> 64 (Skip connection with a scaling factor)
    
class ResidualDenseBlocks(nn.Module):
    """The RRDB final building block used in nESRGAN+.

    Attributes:
        noise (GaussianNoise): The gaussian noise layer.
        conv1x1 (Conv2d): The gaussian noise layer.
    """
    
    def __init__(self):
        super(ResidualDenseBlocks, self).__init__()
        self.sub = nn.Sequential(
            *[ResidualDenseBlock() for _ in range(23)],
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    
    def forward(self, x):
        return x + self.sub(x) # 64 -> 64