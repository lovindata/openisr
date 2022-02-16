import cv2
import numpy as np
import torch
from nesrganp.model.models import Generator


class NErganp:
    """Class containing the attributes and methods to work with the nESRGAN+ generator model.
    
    Args:
        model_path (str): The model weights path.

    Attributes:
        model (Generator): The nESRGAN+ generator model loaded.

    Examples:
        >>> import os
        >>> from nesrganp.nesrganp import NEsrganp
        >>> nesrganp = NErganp(os.path.join('nesrganp', 'resources', 'nESRGANplus.pth'))
    """

    def __init__(self, model_path: str):

        self.model = Generator()
        self.model.load_state_dict(torch.load(model_path), strict=True)
        self.model.eval()
        for _, v in self.model.named_parameters():
            v.requires_grad = False

    def predict(self, in_img: np.ndarray) -> np.ndarray:
        """Predict with the nESRGAN+ model.

        Args:
            in_img (np.ndarray): The input image in BGR with coefficients in (``np.uint8``) and between 0 and 255.

        Returns:
            np.ndarray: An RGB ``np.ndarray`` with coefficients in (``np.float32``) and between 0 and 255.

        Examples:
            >>> edsr = Edsr(os.path.join('edsr', 'resources', 'EDSR_x4.pb'))
            >>> in_img = cv2.imread(in_path, cv2.IMREAD_COLOR)
            >>> out_nerganp = nesrganp.predict(in_img)
        """

        in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB) / 255
        in_img = torch.from_numpy(in_img.transpose((2, 0, 1))).float().unsqueeze(0)     # To float32 input tensor
        out_img = self.model(in_img)
        out_img = out_img.squeeze().float().clamp(0, 1).numpy() 
        out_img = out_img.transpose((1, 2, 0))                                          # Channels, Height, Length -> Height, Length, Channels
        return (out_img * 255.0).round()                                                # Return as np.float32 np.ndarray