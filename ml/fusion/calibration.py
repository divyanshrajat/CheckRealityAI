"""
CheckReality AI - Model Confidence Calibration Module (DR4)
Provides empirical probability calibration via Temperature Scaling,
calculating Expected Calibration Error (ECE over 15 bins), Brier score, and NLL.
Reads measured per-model temperatures from models/<model>/calibration.json.
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODALITY_MODEL_DIRS = {
    "image": "image_deepfake",
    "video": "video_deepfake",
    "audio": "audio_spoof",
    "text_misinformation": "text_misinformation",
    "text_ai_detection": "text_ai_detection"
}


def load_model_calibration(modality: str) -> Optional[dict]:
    """Loads calibration metadata and temperature T for a given modality from disk."""
    model_dir = MODALITY_MODEL_DIRS.get(modality, modality)
    path = os.path.join(PROJECT_ROOT, "models", model_dir, "calibration.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


class TemperatureCalibrator:
    """
    Applies temperature scaling: p_cal = sigmoid(logit / T).
    When T is loaded from calibration.json, it represents the NLL-minimised
    temperature fit on a disjoint calibration set.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(0.01, float(temperature))

    def calibrate(self, raw_prob: float) -> float:
        p = float(np.clip(raw_prob, 1e-7, 1.0 - 1e-7))
        logit = np.log(p / (1.0 - p))
        scaled_logit = logit / self.temperature
        calibrated_p = 1.0 / (1.0 + np.exp(-scaled_logit))
        return float(np.clip(calibrated_p, 0.0001, 0.9999))


def compute_brier_score(predictions: List[float], ground_truth: List[int]) -> float:
    """
    Brier Calibration Score: B = (1/N) * sum((p_i - y_i)^2).
    Computed from actual model predictions and ground-truth binary labels.
    """
    if not predictions or len(predictions) != len(ground_truth):
        return 0.0
    preds = np.array(predictions, dtype=float)
    targets = np.array(ground_truth, dtype=float)
    return float(np.mean((preds - targets) ** 2))


def compute_nll(predictions: List[float], ground_truth: List[int], eps: float = 1e-12) -> float:
    """
    Negative Log-Likelihood (Binary Cross Entropy):
    NLL = -(1/N) * sum(y_i * log(p_i) + (1 - y_i) * log(1 - p_i)).
    """
    if not predictions or len(predictions) != len(ground_truth):
        return 0.0
    preds = np.clip(np.array(predictions, dtype=float), eps, 1.0 - eps)
    targets = np.array(ground_truth, dtype=float)
    return float(-np.mean(targets * np.log(preds) + (1.0 - targets) * np.log(1.0 - preds)))


def compute_ece(predictions: List[float], ground_truth: List[int], n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) across M=15 equal-width confidence bins:
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|.
    """
    if not predictions or len(predictions) != len(ground_truth):
        return 0.0

    preds = np.array(predictions, dtype=float)
    targets = np.array(ground_truth, dtype=int)
    total_samples = len(preds)
    if total_samples == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        if i < n_bins - 1:
            in_bin = (preds >= bin_lower) & (preds < bin_upper)
        else:
            in_bin = (preds >= bin_lower) & (preds <= bin_upper)

        bin_size = int(np.sum(in_bin))
        if bin_size > 0:
            # Empirical probability of positive class in bin vs. mean predicted probability
            bin_acc = float(np.mean(targets[in_bin]))
            bin_conf = float(np.mean(preds[in_bin]))
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)

default_calibrator = TemperatureCalibrator(temperature=1.0)
