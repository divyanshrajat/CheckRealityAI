from pydantic import BaseModel
from typing import Any, Dict, Optional

class AdapterResult(BaseModel):
    label: str
    raw_score: float
    normalized_score: float
    confidence: float
    evidence: Dict[str, Any]
    model_version: str
    inference_ms: float
    status: str

class BaseModelAdapter:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.is_loaded = False
        self.model = None

    def load_model(self):
        """Loads the model into memory. Must be overridden by subclasses."""
        raise NotImplementedError

    def infer(self, data: Any) -> AdapterResult:
        """Runs inference. Must be overridden by subclasses."""
        raise NotImplementedError

    def get_not_available_result(self) -> AdapterResult:
        """Returns a standard response when the model checkpoint is missing."""
        return AdapterResult(
            label="UNKNOWN",
            raw_score=0.0,
            normalized_score=0.5,
            confidence=0.0,
            evidence={"error": "Model checkpoint not found. Waiting for CrowdCore drop."},
            model_version="N/A",
            inference_ms=0.0,
            status="MODEL_NOT_AVAILABLE"
        )
