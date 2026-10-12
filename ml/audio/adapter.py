import random
import time

class AudioAdapter:
    def __init__(self):
        self.modality = "audio"
        self.model_version = "mock-1.0"
        
    def analyze(self, filepath: str):
        # Simulate processing time for audio
        time.sleep(3)
        score = random.uniform(0.1, 0.9)
        label = "real" if score < 0.5 else "fake"
        
        return {
            "modality": self.modality,
            "model_version": self.model_version,
            "raw_score": score,
            "label": label,
            "evidence": {
                "type": "saliency",
                "ref": "mocked-saliency-evidence"
            },
            "inference_ms": 3000
        }
