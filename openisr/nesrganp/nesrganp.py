import cv2
import numpy as np
import torch
from model.models import Generator

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

    def predict(self, in_image_path: str) -> np.ndarray:
        """
        Predict with the nESRGAN+ model.

        :param in_image_path: The input image path
        :return: An RGB np.float32 np.ndarray with 0 <= coefficient_{i,j} <= 255.0
        """

        img = cv2.imread(in_image_path, cv2.IMREAD_COLOR) # Read as BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255
        img = torch.from_numpy(img.transpose((2, 0, 1))).float().unsqueeze(0) # To float32 input tensor
        output = self.model(img)
        output = output.squeeze().float().clamp(0, 1).numpy() 
        output = output.transpose((1, 2, 0)) # Channels, Height, Length -> Height, Length, Channels
        return (output * 255.0).round() # Return as np.float32 np.ndarray