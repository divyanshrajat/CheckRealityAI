"""
CheckReality AI - Video Deepfake Detection Adapter (Production Integration)
Wraps the fine-tuned EfficientNet-B0 + Attentive BiLSTM model pipeline.
"""

import os
import time
import json
from typing import Dict, Any

class VideoAdapter:
    def __init__(self):
        self.modality = "video"
        self.model_version = "efficientnet_b0_lstm-v2.0"
        
    def analyze(self, filepath: str) -> Dict[str, Any]:
        t0 = time.time()
        
        if not filepath or not os.path.exists(filepath):
            return {
                "error": f"Video file not found: {filepath}",
                "modality": self.modality,
                "model_version": self.model_version,
                "label": "unknown",
                "raw_score": 0.0,
                "inference_ms": 0.0
            }
            
        try:
            from backend.app.inference import predict_video
            res = predict_video(filepath)
            
            prob = res.get("fake_probability", 0.0)
            label = res.get("label", "Real" if prob < 0.5 else "Fake")
            elapsed_ms = round((time.time() - t0) * 1000.0, 2)
            
            return {
                "modality": self.modality,
                "model_version": self.model_version,
                "raw_score": float(prob),
                "fake_probability": float(prob),
                "label": label.lower(),
                "evidence": {
                    "type": "frames",
                    "ref": "dual-stream-crops",
                    "sampled_frames": 16,
                    "backbone": "efficientnet_b0",
                    "classifier": "attentive_bilstm"
                },
                "inference_ms": elapsed_ms
            }
        except Exception as e:
            elapsed_ms = round((time.time() - t0) * 1000.0, 2)
            return {
                "error": str(e),
                "modality": self.modality,
                "model_version": self.model_version,
                "label": "error",
                "raw_score": 0.0,
                "inference_ms": elapsed_ms
            }
