import cv2
import numpy as np

class Edsr:
    r"""A class containing the attributes & methods to work with the EDSR model.

    Example::

        import os
        from edsr.edsr import Edsr
        edsr = Edsr(os.path.join('edsr', 'resources', 'EDSR_x4.pb'))

    :ivar model: The EDSRx4 model loaded.
    :vartype model: cv2.dnn_superres_DnnSuperResImpl
    """

    def __init__(self, model_path: str):
        r"""The constructor to load the EDSR model.

        Args:
            model_path (str): The model weights path.
        """

        self.model = cv2.dnn_superres.DnnSuperResImpl_create()
        self.model.readModel(model_path)
        self.model.setModel("edsr", 4)

    def predict(self, in_img: np.ndarray) -> np.ndarray:
        r"""Predict with the EDSR model.

        Example::

            edsr = Edsr(os.path.join('edsr', 'resources', 'EDSR_x4.pb'))
            in_img = cv2.imread(in_path, cv2.IMREAD_COLOR)
            out_edsr = edsr.predict(in_img)

        Args:
            in_img (np.ndarray): The input image in BGR with 0 <= coefficient_{i,j,k} (``np.uint8``) <= 255.0

        Returns:
            An RGB ``np.float32`` ``np.ndarray`` with 0 <= coefficient_{i,j,k} (``np.uint8``) <= 255.0
        """
        
        out_img = self.model.upsample(in_img) # Predicted as BGR
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB) # BGR -> RGB
        return np.asarray(out_img, dtype=np.float32) # np.uint8 -> np.float32