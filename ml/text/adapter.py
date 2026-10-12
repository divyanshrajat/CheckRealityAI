import random
import time

class TextAdapter:
    def __init__(self):
        self.modality = "text"
        self.model_version = "mock-1.0"
        
    def analyze(self, text_content: str):
        # Simulate processing time
        time.sleep(1)
        score = random.uniform(0.1, 0.9)
        label = "real" if score < 0.5 else "fake"
        
        return {
            "modality": self.modality,
            "model_version": self.model_version,
            "raw_score": score,
            "label": label,
            "evidence": {
                "type": "shap",
                "ref": "mocked-shap-evidence"
            },
            "inference_ms": 1000
        }
