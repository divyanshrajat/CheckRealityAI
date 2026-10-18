"""
CheckReality AI - Model Confidence Calibration Module
Implements Temperature Scaling, Platt Scaling (Logistic Calibration),
Expected Calibration Error (ECE), and Brier Score evaluation.
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional

class ProbabilityCalibrator:
    """
    Calibrates raw neural network posteriors using temperature scaling
    and Platt logistic scaling, measuring Expected Calibration Error (ECE).
    """
    
    def __init__(self, temperature: float = 1.15, platt_a: float = 1.05, platt_b: float = -0.02):
        self.temperature = max(0.1, float(temperature))
        self.platt_a = float(platt_a)
        self.platt_b = float(platt_b)
        
    def calibrate_probability(self, raw_prob: float) -> float:
        """
        Applies Platt logistic calibration with temperature smoothing
        to transform overconfident raw softmax scores into true empirical likelihoods.
        """
        p = float(np.clip(raw_prob, 1e-7, 1.0 - 1e-7))
        # Logit computation
        logit = np.log(p / (1.0 - p))
        # Temperature-scaled logit
        scaled_logit = logit / self.temperature
        # Platt affine transformation
        calibrated_logit = self.platt_a * scaled_logit + self.platt_b
        # Sigmoid reconstruction
        calibrated_p = 1.0 / (1.0 + np.exp(-calibrated_logit))
        return float(np.clip(calibrated_p, 0.0001, 0.9999))
        
    def calibrate_vector(self, probabilities: Dict[str, float]) -> Dict[str, float]:
        """Calibrates a dictionary of modality probabilities."""
        return {mod: round(self.calibrate_probability(p), 4) for mod, p in probabilities.items()}

    @staticmethod
    def compute_brier_score(predictions: List[float], ground_truth: List[int]) -> float:
        """
        Computes Brier Calibration Score: B = (1/N) * sum((p_i - y_i)^2).
        Scores < 0.05 represent near-perfect calibration.
        """
        if not predictions or len(predictions) != len(ground_truth):
            return 0.0
        preds = np.array(predictions, dtype=float)
        targets = np.array(ground_truth, dtype=float)
        brier = float(np.mean((preds - targets) ** 2))
        return round(brier, 4)

    @staticmethod
    def compute_ece(predictions: List[float], ground_truth: List[int], n_bins: int = 10) -> float:
        """
        Computes Expected Calibration Error (ECE) across M equal-width confidence bins:
        ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
        """
        if not predictions or len(predictions) != len(ground_truth):
            return 0.0
            
        preds = np.array(predictions, dtype=float)
        targets = np.array(ground_truth, dtype=int)
        
        # Binary prediction accuracy based on 0.5 threshold
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        total_samples = len(preds)
        
        for i in range(n_bins):
            bin_lower = bin_edges[i]
            bin_upper = bin_edges[i + 1]
            
            in_bin = (preds >= bin_lower) & (preds < bin_upper) if i < n_bins - 1 else (preds >= bin_lower) & (preds <= bin_upper)
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                bin_acc = np.mean(targets[in_bin] == (preds[in_bin] >= 0.5))
                bin_conf = np.mean(preds[in_bin])
                ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)
                
        return round(float(ece), 4)

# Global default calibrator instance
default_calibrator = ProbabilityCalibrator(temperature=1.12, platt_a=0.98, platt_b=0.01)
