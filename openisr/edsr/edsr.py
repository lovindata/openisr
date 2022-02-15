import cv2
import numpy as np


class Edsr:
    """Class containing the attributes and methods to work with the EDSR model.

    Args:
        model_path (str): The model weights path.

    Attributes:
        model (cv2.dnn_superres_DnnSuperResImpl): The EDSRx4 model loaded.

    Examples:
        >>> import os
        >>> from edsr.edsr import Edsr
        >>> edsr = Edsr(os.path.join('edsr', 'resources', 'EDSR_x4.pb'))
    """

    def __init__(self, model_path: str):

        self.model = cv2.dnn_superres.DnnSuperResImpl_create()
        self.model.readModel(model_path)
        self.model.setModel("edsr", 4)

    def predict(self, in_img: np.ndarray) -> np.ndarray:
        """Predict with the EDSR model.

        Args:
            in_img (np.ndarray): The input image in BGR with coefficients in (``np.uint8``) and between 0 and 255.

        Returns:
            An RGB ``np.ndarray`` with coefficients in (``np.float32``) and between 0 and 255.

        Examples:
            >>> edsr = Edsr(os.path.join('edsr', 'resources', 'EDSR_x4.pb'))
            >>> in_img = cv2.imread(in_path, cv2.IMREAD_COLOR)
            >>> out_edsr = edsr.predict(in_img)
        """
        
        out_img = self.model.upsample(in_img) # Predicted as BGR
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB) # BGR -> RGB
        return np.asarray(out_img, dtype=np.float32) # np.uint8 -> np.float32