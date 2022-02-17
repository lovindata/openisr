import time
import os

from celery import Celery

import cv2
from edsr.edsr import Edsr
from nesrganp.nesrganp import NErganp


# Global paths
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379")

# Initalize applications
app = Celery("tasks", broker=CELERY_BROKER_URL, result_backend=CELERY_RESULT_BACKEND)

# Load the EDSR and nESRGAN+ models
edsr = Edsr(os.path.join('edsr', 'resources', 'EDSR_x4.pb'))
nesrganp = NErganp(os.path.join('nesrganp', 'resources', 'nESRGANplus.pth'))

""" OLD CODES
# Celery routes
@app.task(name="create_task")
def create_task(task_type):
    time.sleep(int(task_type) * 10)
    return True
"""

# Celery task routes
@app.task(name="process_save")
def process_save(in_path: str, out_path: str):
    """EDSR and nESRGAN+ merged predicts on the image at `in_path` and write the result image at `out_path`.

    Note:
        Celery background task.

    Args:
        in_path (str): The input image path.
        out_path (str): The output image path.
    """
    
    # Read image as BGR
    in_img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    
    # Predict on both EDSR and nESRGAN+ then merge the results
    out_edsr, out_nerganp = edsr.predict(in_img), nesrganp.predict(in_img)
    output = (out_edsr + out_nerganp) / 2
    
    # Write as BVR result
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_path), output)
    
    # Return `True` when its finished
    return True