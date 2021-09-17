import time
import os

from celery import Celery


CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379")


app = Celery("tasks", broker=CELERY_BROKER_URL, result_backend=CELERY_RESULT_BACKEND)

@app.task(name="create_task")
def create_task(task_type):
    time.sleep(int(task_type) * 10)
    return True
