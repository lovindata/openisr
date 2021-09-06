import cv2
import numpy as np

class Edsr:
    """
    This is a class containing the attributes & methods to work with the EDSR model.

    :param model: The EDSRx4 model
    :type model: cv2.dnn_superres_DnnSuperResImpl, optional
    """

    def __init__(self, model_path: str):
        """
        The constructor to load the EDSR model.

        :param model_path: The model weights path
        """

        self.model = cv2.dnn_superres.DnnSuperResImpl_create()
        self.model.readModel(model_path)
        self.model.setModel("edsr", 4)

    def predict(self, in_img: np.ndarray) -> np.ndarray:
        """
        Predict with the EDSR model.

        :param in_img: The input image in BVR with 0 <= coefficient_{i,j,k} (np.uint8) <= 255.0 
        :return: An RGB np.float32 np.ndarray with 0 <= coefficient_{i,j,k} (np.uint8) <= 255.0
        """
        
        out_img = self.model.upsample(in_img) # Predicted as BGR
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB) # BGR -> RGB
        return np.asarray(out_img, dtype=np.float32) # np.uint8 -> np.float32