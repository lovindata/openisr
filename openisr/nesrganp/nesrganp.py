import cv2
import numpy as np
import torch
from nesrganp.model.models import Generator

class NErganp:
    """
    This is a class containing the attributes & methods to work with the nESRGAN+ model.

    :param model: The nESRGAN+x4 model
    :type model: Generator, optional
    """

    def __init__(self, model_path: str):
        """
        The constructor to load the nESRGAN+ model.

        :param model_path: The model weights path
        """

        self.model = Generator()
        self.model.load_state_dict(torch.load(model_path), strict=True)
        self.model.eval()
        for _, v in self.model.named_parameters():
            v.requires_grad = False

    def predict(self, in_img: np.ndarray) -> np.ndarray:
        """
        Predict with the nESRGAN+ model.

        :param in_img: The input image in BVR with 0 <= coefficient_{i,j,k} (np.uint8) <= 255.0 
        :return: An RGB np.float32 np.ndarray with 0 <= coefficient_{i,j,k} (np.uint8) <= 255.0
        """

        in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB) / 255
        in_img = torch.from_numpy(in_img.transpose((2, 0, 1))).float().unsqueeze(0) # To float32 input tensor
        out_img = self.model(in_img)
        out_img = out_img.squeeze().float().clamp(0, 1).numpy() 
        out_img = out_img.transpose((1, 2, 0)) # Channels, Height, Length -> Height, Length, Channels
        return (out_img * 255.0).round() # Return as np.float32 np.ndarray