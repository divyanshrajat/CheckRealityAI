import random
import time

class VideoAdapter:
    def __init__(self):
        self.modality = "video"
        self.model_version = "mock-1.0"
        
    def analyze(self, filepath: str):
        # Simulate longer processing time for video
        time.sleep(5)
        score = random.uniform(0.1, 0.9)
        label = "real" if score < 0.5 else "fake"
        
        return {
            "modality": self.modality,
            "model_version": self.model_version,
            "raw_score": score,
            "label": label,
            "evidence": {
                "type": "frames",
                "ref": "mocked-frames-evidence"
            },
            "inference_ms": 5000
        }
