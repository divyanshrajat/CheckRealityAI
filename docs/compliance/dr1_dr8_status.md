# Systematic Literature Review (SLR) Design Requirements: DR1–DR8 Status Matrix
## CheckReality AI — Multimodal Deepfake & Misinformation Detection System

**Document Version:** 1.0  
**Audit Date:** 18 October 2026  
**Primary Authority:** Systematic Literature Review (SLR) on Multimodal Deepfake Detection & Research Gaps G1–G7  
**Repository State:** Verified against Git Commit `4ac9f2f`

---

## 1. SLR Design Requirements Overview

The Systematic Literature Review (SLR) synthesizes empirical research across deepfake and misinformation detection from 2019 to 2026, establishing eight mandatory Design Requirements (DR1–DR8) to address seven recognized research gaps (G1–G7):

- **G1 (Modal Silos):** Detection models operate independently without standardized representation interfaces.
- **G2 (Naive Late Fusion):** Fusion is restricted to unweighted arithmetic averaging or majority voting.
- **G3 (Dataset Overfitting):** Models report near-perfect within-dataset accuracy but collapse under cross-dataset evaluation.
- **G4 (Missing-Modality Fragility):** Systems fail or produce nonsensical predictions when one or more input modalities are absent.
- **G5 (Opaque & Overconfident Verdicts):** Deep networks output uncalibrated probabilities and lack verifiable explanation layers.
- **G6 (Unmeasured Computational Footprints):** Papers assert real-time viability without publishing empirical latency or memory budgets.
- **G7 (Static Deployment & Concept Drift):** Systems lack telemetry, distribution monitoring, and retraining hooks for evolving generator models.

The table below summarizes the compliance status of CheckReality AI across all 8 Design Requirements:

| Requirement | Description | Targeted Gap | Verified Status | Primary Evidence File | Command / Test Harness |
|---|---|---|---|---|---|
| **DR1** | Uniform Modality Interface | G1 | **MET** | `ml/adapters/base.py`, `backend/app/inference.py` | `python backend/test_integration.py` |
| **DR2** | Intermediate Cross-Modal Attention Fusion | G1, G2 | **NOT MET** | `ml/fusion/attention.py` | Code inspection & architecture analysis |
| **DR3** | Modality Gating & Missing-Modality Semantics | G2, G4 | **PARTLY MET** | `benchmarks/subset_coverage_report.json` | `python ml/fusion/gating.py` |
| **DR4** | Confidence Calibration | G5 | **PARTLY MET** | `models/calibration_benchmark.json` | `python training/fit_calibration.py` |
| **DR5** | Two-Layer Explainability | G5 | **MET** | `ml/fusion/engine.py`, `backend/app/inference.py` | `python backend/test_integration.py` |
| **DR6** | Cross-Dataset & Robustness Evaluation | G3 | **PARTLY MET** | `models/cross_dataset_benchmark.json` | `python training/eval_cross_dataset.py` |
| **DR7** | Latency & Size Budgets | G6 | **MET** | `benchmarks/latency_size_budget_report.json` | `python benchmarks/benchmark_latency.py` |
| **DR8** | Model Lifecycle, Drift & Retraining | G7 | **PARTLY MET** | `models/model_registry.json`, `models/retraining_queue.log` | `python ml/monitoring/drift_detector.py` |

---

## 2. In-Depth Technical Audit: DR1 through DR8

### DR1 — Uniform Modality Interface
* **Objective:** Ensure all input streams (text, image, video, audio) conform to a standardized integration protocol, schema, and error-handling contract.
* **Implementation:**
  - Base class `ModalityAdapter` defined in `ml/adapters/base.py`.
  - Concrete adapters implement `load()`, `preprocess()`, and `predict()`.
  - Unified output dictionary schema:
    ```python
    {
        "modality": str,              # "text" | "image" | "video" | "audio"
        "raw_score": float,           # Uncalibrated model output [0.0, 1.0]
        "calibrated_score": float,    # Temperature-scaled probability [0.0, 1.0]
        "label": str,                 # "Real" | "Fake" | "Bonafide" | "Spoof"
        "inference_ms": float,        # Wall-clock execution time in milliseconds
        "evidence": dict              # Modality-native saliency artifact
    }
    ```
  - Input validation: File extensions and MIME types strictly validated; malformed or unreadable inputs raise controlled exceptions (`HTTPException 400`) rather than terminating worker processes.
* **Verification Evidence:** `python backend/test_integration.py` tests all 4 modalities sequentially; all adapters return valid dictionaries conforming to the schema.
* **Status:** **MET**.

---

### DR2 — Intermediate Cross-Modal Attention Fusion
* **Objective:** Implement learned intermediate feature-level cross-attention across multimodal representations rather than relying exclusively on late decision averaging.
* **Implementation:**
  - PyTorch module `CrossModalAttentionFusion` implemented in `ml/fusion/attention.py`.
  - Mathematical formulation:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M_{\text{pad}}\right) V$$
    where $M_{\text{pad}}$ is a boolean key-padding mask masking out inactive modalities ($-\infty$ logits).
  - Dimension projections: Text (768), Image (1536), Audio (512), Video (512) projected into a common latent dimension $d = 256$ via linear projection layers.
  - Classification head: LayerNorm $\to$ MultiheadAttention ($h=4$) $\to$ Linear(256, 128) $\to$ ReLU $\to$ Dropout(0.2) $\to$ Linear(128, 1) $\to$ Sigmoid.
* **Audit Finding & Gap Analysis:**
  - Although the module is fully formulated and syntactically valid, **its parameters are randomly initialized and untrained**.
  - A jointly labelled four-modality dataset containing naturally aligned text, image, video, and audio from identical events does not exist on disk or in the public literature.
  - Consequently, `CrossModalAttentionFusion` is **not invoked in live inference** (`backend/app/inference.py` calls the reliability-weighted decision fusion engine instead).
* **Status:** **NOT MET**.
* **Disposition:** Retained as an architectural reference in `ml/fusion/attention.py`; full joint training is categorized under genuine Future Scope.

---

### DR3 — Modality Gating & Missing-Modality Semantics
* **Objective:** Gracefully handle any arbitrary subset of active modalities without degradation or crashes across all 15 non-empty combinations of 4 modalities.
* **Implementation:**
  - Gating function implemented in `ml/fusion/gating.py` and `ml/fusion/engine.py`.
  - Base empirical reliability weights derived from validation ROC-AUC:
    $$w_{\text{base}} = \{ \text{video}: 0.35, \text{audio}: 0.25, \text{image}: 0.25, \text{text}: 0.15 \}$$
  - Dynamic convex renormalization across active subset $\mathcal{M} \subseteq \{\text{video}, \text{audio}, \text{image}, \text{text}\}$:
    $$w_m' = \frac{w_m}{\sum_{j \in \mathcal{M}} w_j}, \quad \forall m \in \mathcal{M}, \quad \text{such that } \sum_{m \in \mathcal{M}} w_m' = 1.0$$
  - Unified probability computation:
    $$p_{\text{unified}} = \sum_{m \in \mathcal{M}} w_m' \cdot p_m$$
  - Cross-modal agreement index:
    $$A = 1 - \frac{\sigma}{\sigma_{\max}}, \quad \sigma = \sqrt{\frac{1}{|\mathcal{M}|} \sum_{m \in \mathcal{M}} (p_m - \bar{p})^2}, \quad \sigma_{\max} = 0.5$$
* **Verification Evidence:**
  - `benchmarks/subset_coverage_report.json` validates convex weight conservation ($\sum w_m' = 1.0$) across all 15 power-set subsets.
  - Synthetic paired bundle evaluation across 40 test bundles yielded ROC-AUC ranging from 0.5686 (text alone) to 0.9510 (4 modalities combined).
* **Limitations:**
  - The 40 bundles are synthetic pairings of unimodal validation records (e.g. video from FF++ paired with text from Kaggle News), not naturally co-occurring media.
* **Status:** **PARTLY MET** (Weight logic and subset enumeration verified; natural multimodal evaluation absent).

---

### DR4 — Confidence Calibration
* **Objective:** Calibrate model confidence scores so that predicted probabilities reflect empirical ground-truth likelihoods, measuring Expected Calibration Error (ECE) and Brier score.
* **Implementation:**
  - Temperature scaling implemented in `training/fit_calibration.py`:
    $$\hat{p} = \sigma\left(\frac{z}{T}\right)$$
    where $z$ is the logit and $T > 0$ is optimized via Negative Log-Likelihood (NLL) minimization on a disjoint calibration partition.
  - Evaluated on a separate held-out validation split using 15-bin uniform ECE, Brier score, and NLL:
    $$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|, \quad \text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
* **Empirical Measurements (from `models/calibration_benchmark.json`):**
  - **Video (FF++ C23):** $T = 1.2935$. ECE dropped from 0.4185 to **0.3637**; NLL dropped from 0.5840 to **0.5315**; Brier dropped from 0.1745 to **0.1695**.
  - **Image (140k Faces):** $T = 0.7473$. Brier = 0.0099, NLL = 0.0350.
  - **Audio (ASVspoof):** $T = 0.6049$. Brier = 0.0040, NLL = 0.0199.
  - **Text Misinformation:** $T = 1.0198$. Brier = 0.0010, NLL = 0.0033.
  - **Text AI Detection:** $T = 0.2358$. Brier = 0.0000, NLL = 0.0000.
* **Identified Gap:**
  - Calibration parameters $T$ are fit and evaluated **per individual modality**.
  - Calibration of the final fused unified probability $p_{\text{unified}}$ is `NOT_MEASURED` because fitting temperature on fusion outputs requires a held-out dataset of jointly labelled multimodal records.
* **Status:** **PARTLY MET** (Per-modality calibration MET; fusion-level calibration NOT MEASURED).

---

### DR5 — Two-Layer Explainability
* **Objective:** Deliver transparent forensic explanations at two distinct hierarchical levels: (1) modality-native feature saliency, and (2) cross-modal fusion attribution.
* **Implementation:**
  - **Layer 1 (Modality-Native):**
    - *Image & Video:* Gradient-weighted Class Activation Mapping (Grad-CAM) targeting the final convolutional layer of EfficientNet-B3 / B0 (`_conv_head`), extracting normalized spatial heatmaps.
    - *Audio:* Acoustic spectrogram saliency computing frequency-temporal anomaly energy across Mel filterbanks.
    - *Text:* Gradient-based token attribution computing input embedding saliency vectors $\|\nabla_{e_t} \mathcal{L}\|_2$ for the top influential tokens.
  - **Layer 2 (Cross-Modal Fusion Attribution):**
    - Signed weighted-deviation attribution implemented in `ml/fusion/engine.py`:
      $$\Delta_m = w_m' \cdot (p_m - p_{\text{unified}})$$
      where $\Delta_m > 0$ indicates modality $m$ pulled the unified verdict towards "Fake", and $\Delta_m < 0$ indicates it pulled towards "Real".
    - Automated natural-language forensic synthesis identifying consensus, divergence, and dominant modal indicators.
* **Verification Evidence:** `python backend/test_integration.py` successfully verifies Layer 1 heatmaps/tokens and Layer 2 attributions across all modalities.
* **Status:** **MET**.

---

### DR6 — Cross-Dataset & Robustness Evaluation
* **Objective:** Establish model generalization and boundary robustness under domain shifts, compression artifacts, and acoustic perturbations.
* **Implementation:**
  - Empirical perturbation suite implemented in `training/eval_cross_dataset.py` (500 samples per test, random seed 42):
    1. *Image JPEG Compression ($Q=50$):* Accuracy dropped from 98.6% to 84.6% ($\Delta \text{AUC} = -0.0095$).
    2. *Audio Telephony Codec (8kHz downsampling):* Accuracy dropped from 99.67% to 95.00% ($\Delta \text{AUC} = -0.0103$).
    3. *Video 25% Frame Dropout:* Accuracy maintained at 76.25% vs 75.00% clean ($\Delta \text{AUC} = +0.0143$).
    4. *Text 5% Typo Noise:* Accuracy maintained at 97.40% vs 96.20% clean ($\Delta \text{AUC} = 0.0000$).
* **Unmeasured External Benchmarks:**
  - Celeb-DF v2: `NOT_MEASURED` (requires institutional DUA; not downloaded).
  - FaceForensics++ C40 (heavy compression): `NOT_MEASURED` (only C23 downloaded).
  - In-the-Wild Audio / ASVspoof 2021 DF: `NOT_MEASURED` (not downloaded).
  - LIAR / FakeNewsNet: `NOT_MEASURED` (not downloaded).
* **Status:** **PARTLY MET** (Perturbation robustness measured; external cross-dataset benchmarks NOT MEASURED).

---

### DR7 — Latency & Size Budgets
* **Objective:** Measure and document model parameter counts, disk footprints, peak memory, and latency percentiles (P50, P95, P99) under controlled execution conditions.
* **Implementation & Measurements (from `benchmarks/latency_size_budget_report.json`):**
  - Profiling protocol: 10 warmup runs + 100 timed repetitions on Intel i5 CPU and NVIDIA GeForce GTX 1650 GPU (15.79 GB RAM, PyTorch 2.6.0+cu124, batch size = 1).
  - **Results Summary Table:**
    | Model / Component | Parameters | Disk MB | GPU P50 Latency | GPU P95 Latency | CPU P50 Latency | Peak GPU VRAM |
    |---|---|---|---|---|---|---|
    | EfficientNet-B3 (Image + TTA) | 10.70M | 41.35 MB | 29.40 ms | 49.93 ms | 109.83 ms | 65.49 MB |
    | EfficientNet-B0 + LSTM (Video) | 6.11M | 23.60 MB | 29.91 ms | 30.11 ms | 364.47 ms | 242.96 MB |
    | ResNet-18 (Audio SpecAugment) | 11.18M | 42.71 MB | 3.32 ms | 4.03 ms | 14.60 ms | 152.56 MB |
    | RoBERTa-base (Misinformation) | 124.65M | 478.91 MB | 63.54 ms | 64.59 ms | 468.64 ms | 621.18 MB |
    | RoBERTa-base (AI Text Detection) | 124.65M | 478.91 MB | 63.81 ms | 64.55 ms | 472.52 ms | 1097.92 MB |
    | Fusion Engine (10,000 calls) | N/A | < 1 KB | 0.039 ms | 0.072 ms | 0.039 ms | < 1 MB |
* **Verification Evidence:** `benchmarks/latency_size_budget_report.json` produced by `python benchmarks/benchmark_latency.py`.
* **Status:** **MET**.

---

### DR8 — Model Lifecycle, Drift & Retraining
* **Objective:** Establish model versioning, track baseline posterior distributions, detect distribution drift via statistical tests, and log retraining triggers.
* **Implementation:**
  - Checkpoint SHA-256 digests and training lineage serialized in `models/model_registry.json`.
  - Baseline validation posterior distributions saved to `models/<model>/baseline_posteriors.npy`.
  - Drift detector implemented in `ml/monitoring/drift_detector.py` computing:
    1. Two-sample Kolmogorov-Smirnov (KS) test ($p < 0.05$ threshold).
    2. Population Stability Index (PSI):
       $$\text{PSI} = \sum_{k=1}^K \left(P_k - Q_k\right) \ln\left(\frac{P_k}{Q_k}\right)$$
       where $\text{PSI} > 0.25$ indicates significant concept drift.
  - Automated logging: Breaching thresholds appends formal retraining request to `models/retraining_queue.log`.
* **Identified Gap:**
  - Drift detector is fully implemented and tested against simulated batches, but production streaming drift is `NOT_MEASURED` because the system operates in an offline academic lab environment without real-time user traffic feeds.
* **Status:** **PARTLY MET** (Baseline lineage and drift detection logic MET; production streaming validation NOT MEASURED).

---

## 3. Summary of Compliance Status

CheckReality AI achieves full compliance (**MET**) on 3 of the 8 SLR Design Requirements (DR1, DR5, DR7), substantial compliance (**PARTLY MET**) on 4 requirements (DR3, DR4, DR6, DR8), and honest non-compliance (**NOT MET**) on 1 requirement (DR2). Every limitation is traceable to physical dataset or hardware boundaries, with zero fabricated experimental evidence.
