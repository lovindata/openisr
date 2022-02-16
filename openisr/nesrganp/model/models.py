import torch
import torch.nn as nn
from nesrganp.model.layers import ResidualDenseBlocks

class Generator(nn.Module):
    """The global architecture of nESRGAN+ generator.

    Note:
        Please see the original SRGAN Fig.4, ESRGAN and ESRGAN+ papers for more architecture details.

    Attributes:
        model (Sequential): Conv2D 3/64c, RDBs 64/64c, 2xUpsample 64/64c, some Conv2D with activations, 2xUpsample 64/64c, some Conv2D with activations and Conv2D 64/3c.
    """
    
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            ResidualDenseBlocks(),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Upsample(scale_factor=2.0, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The method to apply the nESRGAN+ generator on `x` (64/64c).

        Args:
            x (Tensor): The input tensor with 3c.

        Returns:
            Tensor: Output tensor 3c with the nESRGAN+ generator applied.
        """
        
        return self.model(x) # 3 -> 3