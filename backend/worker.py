from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

redis_url = os.environ.get("REDIS_URL", "redis://localhost:6381/0")

celery_app = Celery(
    "worker",
    broker=redis_url,
    backend=redis_url
)

@celery_app.task
def process_video_task(submission_id: str, filepath: str):
    # TODO: implement video adapter processing
    pass

@celery_app.task
def process_audio_task(submission_id: str, filepath: str):
    # TODO: implement audio adapter processing
    pass
