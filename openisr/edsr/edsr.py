import cv2
import numpy as np

class Edsr:
    """
    This is a class containing the attributes & methods to work with the EDSR model

    :param model: The EDSRx4 model
    :type model: cv2.dnn_superres_DnnSuperResImpl, optional
    """

    def __init__(self, model_path: str):
        """
        The constructor to load the edsr model.

        :param model_path: The model weights path
        """

        self.model = cv2.dnn_superres.DnnSuperResImpl_create()
        self.model.readModel(model_path)
        self.model.setModel("edsr", 4)

    def predict(self, in_image_path: str) -> np.ndarray:
        """
        Predict with the edsr model.

        :param in_image_path: The input image path
        :return: An RVB array result of the model
        """

        img = cv2.imread(in_image_path, cv2.IMREAD_COLOR) # Read as BVR
        output = self.model.upsample(img) # Predicted as BVR
        return output[:, :, [2, 1, 0]] # Return the result in RVB