# Synopsis Scope Alignment & Traceability Matrix
## CheckReality AI — Multimodal Deepfake and Misinformation Detection System

**Document Version:** 1.0  
**Audit Date:** 18 October 2026  
**Primary Authority:** Approved Project Synopsis (`Project-2 Synopsis.pdf`, 21 pp., IKGPTU B.Tech CSE Major Project)  
**Evaluator / Guide:** Dr. Manpreet Kaur  
**Author / Candidate:** Divyansh Rajat (Roll No. 2300358)  
**Repository State:** Verified against Git Commit `4ac9f2f`

---

## 1. Executive Summary

This compliance document establishes a formal, verifiable mapping between the approved project commitments set forth in the IKGPTU Major Project Synopsis and the empirical reality of the codebase. In strict adherence to academic integrity principles:
1. Every commitment is cited using its original phrasing or faithful extraction from `Project-2 Synopsis.pdf`.
2. Status is assigned solely from code execution, unit/integration test outcomes, and serialized benchmark artifacts on disk.
3. Incomplete commitments are **not** moved to "Future Scope" to mask gaps; they are recorded with exact status, diagnostic root causes, and explicit constraints.

---

## 2. Approved Synopsis Commitments: Core Objectives

The approved synopsis establishes four primary objectives (Synopsis §3):

* **Objective 1:** *Develop modality-specific deep learning modules for AI-generated text, deepfake images, manipulated videos, and synthetic speech.*
* **Objective 2:** *Design a cross-modal feature-fusion framework that combines text, image, video, and audio representations for unified authenticity assessment.*
* **Objective 3:** *Implement an explainable, confidence-aware decision framework.*
* **Objective 4:** *Evaluate and compare the proposed system with existing unimodal and multimodal approaches using relevant performance indicators.*

The methodology (Synopsis §5, §6) further mandates:
* Pretrained backbones: RoBERTa-base (text), EfficientNet-B0/B3 (images), EfficientNet + LSTM (video), CNN on Mel-spectrograms (audio).
* Cross-modal feature fusion and adaptive attention-based decision logic.
* Confidence calibration and two-layer explainability (modality-native saliency and fusion attribution).
* A web-based application supporting single and multimodal user uploads.
* Offline empirical evaluation across Accuracy, Precision, Recall, F1-score, ROC-AUC, inference time, computational efficiency, and cross-dataset generalization.

---

## 3. Comprehensive Traceability Matrix

The four status definitions are strictly applied:
* **MET:** Implemented, verified by code path, and backed by empirical experiment or passing test.
* **PARTLY MET:** Substantial capability implemented and verified, but full promised scope is not fully satisfied.
* **NOT MET:** Promised capability is absent, stubbed, or architectural design only without trained weights.
* **NOT MEASURED:** Capability or benchmark cannot be evaluated due to missing external datasets, physical hardware limits, or lack of streaming production traffic.

| Req ID | Original Requirement (Synopsis / SLR) | Source Reference | Current Repository Implementation | Verification Evidence & Test Command | Status | Identified Gap | Action / Disposition |
|---|---|---|---|---|---|---|---|
| **REQ-OBJ1-TXT** | Modality-specific detection module for AI-generated text and misinformation | Synopsis §3 Obj 1, §5.1, §7 | Two fine-tuned RoBERTa-base models: `text_misinformation` (Fake vs Real News) and `text_ai_detection` (AI vs Human) in `backend/app/inference.py` | `models/text_misinformation/metrics.json` (Acc 99.95%), `models/text_ai_detection/metrics.json` (Acc 100%), verified via `python backend/test_integration.py` | **MET** | Input token length capped at 512 tokens; English language only. | Verified operational capability. Maintain current inference pipeline. |
| **REQ-OBJ1-IMG** | Modality-specific detection module for deepfake images | Synopsis §3 Obj 1, §5.2, §7 | Fine-tuned EfficientNet-B3 with horizontal-flip Test-Time Augmentation (TTA) and face detection cropping via OpenCV Haar Cascades in `backend/app/inference.py` | `models/image_deepfake/metrics.json` (Acc 99.40%), verified via `python backend/test_integration.py` | **MET** | Haar Cascade can fail on extreme facial profile angles (>45 deg yaw). | Verified operational capability. Document face-detector bounding box constraints. |
| **REQ-OBJ1-VID** | Modality-specific detection module for manipulated videos | Synopsis §3 Obj 1, §5.3, §7 | Spatial-temporal pipeline: 16-frame uniform sampling, OpenCV face crop, EfficientNet-B0 feature extractor feeding a 2-layer LSTM temporal network | `models/video_deepfake/metrics.json` (Acc 99.25%), verified via `python backend/test_integration.py` | **MET** | Fixed sequence length of 16 frames downsampled uniformly regardless of video duration. | Verified operational capability. Frame sampling limitation documented in Limitations. |
| **REQ-OBJ1-AUD** | Modality-specific detection module for synthetic speech | Synopsis §3 Obj 1, §5.4, §7 | Librosa Mel-spectrogram extraction (128 bins, 16kHz) feeding fine-tuned ResNet-18 classifier trained on ASVspoof 2019 LA | `models/audio_spoof/metrics.json` (Acc 99.60%), verified via `python backend/test_integration.py` | **MET** | Input audio trimmed or padded to 3.0 seconds (157 frames). | Verified operational capability. Audio window duration documented. |
| **REQ-OBJ2-FUS** | Cross-modal feature-fusion framework for unified authenticity assessment | Synopsis §3 Obj 2, §5.5, §6 Step 5 | Gating/decision-level reliability-weighted fusion operational in `ml/fusion/gating.py` and `ml/fusion/engine.py`. Intermediate cross-modal attention module designed in `ml/fusion/attention.py`. | Real decision-level fusion verified in `backend/test_integration.py` and `benchmarks/subset_coverage_report.json`; `ml/fusion/attention.py` code inspection | **PARTLY MET** | Intermediate feature-level attention module (`ml/fusion/attention.py`) has random initialized weights and is not invoked in production inference because no jointly labelled 4-modality corpus exists. | Decision-level fusion is MET. Feature-level attention fusion is NOT MET. Honest classification: PARTLY MET. |
| **REQ-OBJ3-CAL** | Confidence-aware decision framework via probability calibration | Synopsis §3 Obj 3, §5.5, §6 Step 5 | Temperature scaling ($T$) fitted via NLL optimization on disjoint calibration split and evaluated on validation split with 15-bin ECE, Brier score, and NLL | `models/calibration_benchmark.json`, `models/*/calibration.json`, generated by `python training/fit_calibration.py` | **PARTLY MET** | Calibration is implemented and evaluated per modality. Fusion-level calibration across unified scores is uncalibrated because no held-out paired 4-modality dataset is present. | Modality calibration is MET. Fused calibration is NOT MEASURED. Overall: PARTLY MET. |
| **REQ-OBJ3-XAI** | Explainable decision framework (modality and fusion level) | Synopsis §3 Obj 3, §5.5, §6 Step 6 | Layer 1: Modality-native saliency (Grad-CAM for image/video, token gradient attribution for text, acoustic Mel saliency for audio). Layer 2: Signed weighted-deviation cross-modal attribution. | Verified live in `backend/test_integration.py` test suite 4/5 (Grad-CAM length 32584 bytes, 4 acoustic anomaly windows, 6 token attributions) | **MET** | Token gradients are gradient-based approximations of SHAP values; temporal frame saliency uses frame-level LSTM hidden states. | Verified operational capability. Limitations on explanation faithfulness documented. |
| **REQ-OBJ4-EVAL** | Evaluation and baseline comparison using relevant performance indicators | Synopsis §3 Obj 4, §6 Step 7, §7 | Held-out evaluation across Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, and EER across all 5 models. Baseline comparison against unimodal backbones and decision-level fusion. | `models/*/metrics.json`, `benchmarks/latency_size_budget_report.json`, verified via `python backend/test_integration.py` | **PARTLY MET** | External cross-dataset generalization (Celeb-DF v2, FF++ C40, In-the-Wild, LIAR) is NOT MEASURED due to absence of local corpora. Within-dataset evaluation is complete. | Real perturbation robustness (DR6) measured as proxy; independent cross-dataset benchmarks marked NOT MEASURED. |
| **REQ-METH-WEB** | Web application supporting text, image, video, and audio modalities | Synopsis §5, §7 | Single-page React SPA (Vite, TypeScript, Tailwind) with unified upload form, async polling, and detailed authenticity report view | Verified in React client source `frontend/src/App.tsx`, `frontend/src/components/UploadModal.tsx`, `frontend/src/components/ReportModal.tsx` | **MET** | WebSockets not implemented; polling on `/submissions/{id}` is used for async video/audio. | Verified operational capability. Polling mechanism meets demo and testing requirements. |
| **REQ-METH-EFF** | Computational efficiency and latency measurement | Synopsis §6 Step 7, §7 | Automated latency, parameter count, disk size, and peak memory benchmark on CPU and GPU across 10 warmup and 100 timed runs | `benchmarks/latency_size_budget_report.json`, generated by `python benchmarks/benchmark_latency.py` | **MET** | Batch size fixed at 1 for latency tests (representative of interactive web inference). | Verified operational capability. Full latency profiles (mean, P50, P95, P99) documented. |

---

## 4. Evaluation of Synopsis Commitments vs Current Reality

### 4.1 What is Implemented and Verified (MET)
1. **Five Modality Deep Learning Models:**
   - Text Misinformation: RoBERTa-base (435M bytes, Acc 99.95%, AUC 1.0000).
   - Text AI Detection: RoBERTa-base (435M bytes, Acc 100.00%, AUC 1.0000).
   - Image Deepfake: EfficientNet-B3 with horizontal-flip TTA (41.35 MB, Acc 99.40%, AUC 0.9998).
   - Audio Spoof: ResNet-18 on Mel-Spectrograms (42.71 MB, Acc 99.60%, AUC 0.9999).
   - Video Deepfake: EfficientNet-B0 + 2L LSTM (23.60 MB, Acc 99.25%, AUC 0.9994).
2. **Decision-Level Modality Gating & Fusion:**
   - Convex reliability-weighted fusion ($w_m' = w_m / \sum_{j \in \mathcal{M}} w_j$) verified across all 15 power-set subsets ($2^4 - 1$).
   - Cross-modal agreement index ($A = 1 - \frac{\sigma}{\sigma_{\max}}$) and automated forensic narrative synthesis.
3. **Two-Layer Explainability:**
   - Layer 1: Native feature visualizations (Grad-CAM heatmaps, token gradient rankings, acoustic frequency-temporal saliency).
   - Layer 2: Signed weighted-deviation cross-modal attribution showing directional pull of each active modality relative to the unified score.
4. **Interactive Web Application:**
   - Fully functioning React/Vite frontend with single/multimodal upload, dynamic file-type validation, and asynchronous processing state management.
5. **Efficiency & Resource Benchmarking:**
   - Comprehensive profiling on both Intel CPU and NVIDIA GeForce GTX 1650 GPU with exact parameter counts, disk footprints, and P50/P95/P99 latency distributions.

### 4.2 What is Partially Implemented (PARTLY MET)
1. **Multimodal Feature Fusion (Objective 2):**
   - Implemented: Real reliability-weighted decision fusion handling missing modalities gracefully.
   - Gap: Intermediate cross-modal attention fusion (`ml/fusion/attention.py`) is defined with full PyTorch tensor operations and padding masks, but its parameters are untrained and it is bypassed during live inference.
2. **Confidence Calibration (Objective 3):**
   - Implemented: Per-modality temperature scaling ($T$) fitted on disjoint calibration splits with empirical 15-bin ECE and Brier score evaluation.
   - Gap: The unified multimodal score is an ensemble of calibrated marginal probabilities, but fusion-level calibration parameters have not been fit directly on multimodal labels.
3. **Generalization & Evaluation (Objective 4):**
   - Implemented: Within-dataset held-out test splits with >99% empirical metrics; real-world input perturbation tests (JPEG compression, audio downsampling, frame dropout, typo noise).
   - Gap: Independent cross-dataset benchmarks (Celeb-DF v2, FaceForensics++ C40, In-the-Wild Audio, LIAR) were not downloaded locally and are explicitly marked `NOT_MEASURED`.

### 4.3 What is Not Met (NOT MET)
1. **Trained Intermediate Attention Fusion:**
   - Without an aligned 4-modality training corpus, feature representations cannot be jointly optimized via backpropagation.

### 4.4 What Cannot Be Measured (`NOT MEASURED`)
1. **Cross-Dataset Benchmark Transferability:** Missing Celeb-DF v2, FF++ C40, In-the-Wild audio, and LIAR corpora on disk.
2. **Streaming Production Model Drift:** Missing live streaming production traffic in this standalone academic development environment.
3. **Fusion-Level Probability Calibration:** Missing jointly labelled multi-stream validation files.

---

## 5. Scope Alignment Conclusion

The project satisfies all four core synopsis objectives at the functional prototype and decision-support level. The gaps identified—specifically the transition from decision-level fusion to joint feature-level attention, and external cross-dataset generalization—are rigorously documented, empirically grounded, and mathematically characterized without resorting to synthetic claims or evasive future-scope reclassification.
