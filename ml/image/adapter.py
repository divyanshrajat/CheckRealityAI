import random
import time

class ImageAdapter:
    def __init__(self):
        self.modality = "image"
        self.model_version = "mock-1.0"
        
    def analyze(self, filepath: str):
        # Simulate processing time
        time.sleep(2)
        score = random.uniform(0.1, 0.9)
        label = "real" if score < 0.5 else "fake"
        
        return {
            "modality": self.modality,
            "model_version": self.model_version,
            "raw_score": score,
            "label": label,
            "evidence": {
                "type": "gradcam",
                "ref": "mocked-gradcam-evidence"
            },
            "inference_ms": 2000
        }
