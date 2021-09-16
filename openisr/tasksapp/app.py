import time

from celery import Celery

CELERY_BROKER_URL = "redis://redis:6379/0"
CELERY_RESULT_BACKEND = "redis://redis:6379/0"

app = Celery("tasks", broker=CELERY_BROKER_URL, result_backend=CELERY_RESULT_BACKEND)

@app.task(name="create_task")
def create_task(task_type):
    time.sleep(int(task_type) * 10)
    return True
