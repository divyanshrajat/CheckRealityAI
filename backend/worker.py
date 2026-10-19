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
    """
    Asynchronous Celery worker task for video deepfake detection.
    Evaluates the video through the fine-tuned EfficientNet-B0 + BiLSTM adapter.
    """
    from ml.video.adapter import VideoAdapter
    adapter = VideoAdapter()
    return adapter.analyze(filepath)

@celery_app.task
def process_audio_task(submission_id: str, filepath: str):
    """
    Asynchronous Celery worker task for audio spoof detection.
    Evaluates audio track through the fine-tuned ResNet-18 Log-Mel Spectrogram adapter.
    """
    from ml.audio.adapter import AudioAdapter
    adapter = AudioAdapter()
    return adapter.analyze(filepath)
