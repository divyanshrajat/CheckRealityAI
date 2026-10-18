# Current Limitations & Known Constraints
## CheckReality AI — Multimodal Deepfake and Misinformation Detection System

**Document Version:** 1.0  
**Audit Date:** 18 October 2026  
**Document Purpose:** Academic transparency regarding technical, data, hardware, and evaluation boundaries  
**Repository State:** Verified against Git Commit `4ac9f2f`

---

## 1. Executive Statement on Limitations

In accordance with academic research standards, this document formally delineates the technical, experimental, and operational boundaries of the CheckReality AI system. CheckReality AI is an academic research prototype designed as a **decision-support tool** for digital content authenticity assessment. It is **not** an infallible forensic truth detector, legal verification instrument, or commercial-scale content moderation platform.

Every limitation documented herein is traceable to physical resource constraints, dataset availability, or methodological scope boundaries.

---

## 2. Dataset Availability & Cross-Dataset Gaps

### 2.1 Missing External Cross-Dataset Benchmarks
While CheckReality AI exhibits exceptional within-dataset performance (>99% accuracy across all five modality models on held-out test splits), external cross-dataset generalization could not be empirically measured for several key corpora:

| Target Modality | External Benchmark | Intended Purpose | Availability Status | Root Cause for Non-Measurement |
|---|---|---|---|---|
| **Video** | Celeb-DF v2 | High-quality deepfake cross-dataset generalization | **NOT MEASURED** | Celeb-DF v2 requires an institutional Data Use Agreement (DUA) and identity verification; not downloaded to the local workspace. |
| **Video** | FaceForensics++ C40 | Extreme lossy compression artifact robustness | **NOT MEASURED** | Only the C23 (high quality / light compression) subset was downloaded; the 300GB+ C40 raw archive exceeded local disk quotas. |
| **Audio** | In-the-Wild Audio / ASVspoof 2021 DF | Unconstrained real-world acoustic deepfakes | **NOT MEASURED** | ASVspoof 2021 DF and In-the-Wild (Mueller et al., 2022) were not downloaded locally; evaluation relies on ASVspoof 2019 LA. |
| **Text** | LIAR / FakeNewsNet | Cross-domain short-statement misinformation transfer | **NOT MEASURED** | External political fact-checking corpora were not downloaded; evaluation is limited to Kaggle Fake & Real News and AI vs Human Text. |

> [!WARNING]
> Within-dataset accuracy cannot be assumed to transfer to unseen generators, novel synthesis algorithms, or in-the-wild social media platforms without explicit cross-dataset validation.

### 2.2 Lack of a Naturally Aligned 4-Modality Corpus
A fundamental limitation of existing deepfake research is the near-total absence of publicly available, naturally aligned four-modality datasets where a single event or story contains co-occurring, semantically entangled text, image, video, and audio with verified ground-truth labels.
- Consequently, multimodal evaluation in `benchmarks/subset_coverage_report.json` relies on **synthetic bundles** (pairing an arbitrary FF++ video with an arbitrary Kaggle news article and ASVspoof audio sample).
- While synthetic bundles validate the mathematical correctness of convex gating and weight normalization, they do not simulate semantic cross-modal incoherence (e.g. an authentic video paired with an out-of-context manipulative text caption).

---

## 3. Hardware, Compute & Resource Constraints

### 3.1 Local Hardware Profile
All experiments, training runs, and inference benchmarks were executed on an entry-level mobile workstation:
- **CPU:** Intel64 Family 6 Model 140 Stepping 2 (4 cores / 8 threads)
- **Host RAM:** 15.79 GB DDR4
- **GPU:** NVIDIA GeForce GTX 1650 (Laptop GPU) with **3.99 GB (4096 MB) VRAM**
- **CUDA Capability:** 7.5 (Turing architecture), CUDA 12.4, PyTorch 2.6.0+cu124

### 3.2 Impact on Architecture and Training
1. **VRAM Bottleneck:** With only ~4 GB of GPU VRAM, batch sizes for RoBERTa-base were strictly capped at $B=4$ or $B=8$ with gradient accumulation. Simultaneous execution of all five models concurrently on GPU causes an Out-Of-Memory (CUDA OOM) fault; models must be evaluated sequentially or run on CPU.
2. **Video Sequence Length:** Video analysis is constrained to 16 uniformly sampled frames ($16 \times 3 \times 224 \times 224$) feeding the LSTM. Modeling longer temporal contexts (e.g. 64 or 128 frames) was precluded by VRAM allocation limits.
3. **Training Duration:** Training full transformer backbones required substantial clock time (several hours per model), limiting extensive hyperparameter sweeps or 10-fold cross-validation.

---

## 4. Model Architecture & Pipeline Limitations

### 4.1 Untrained Intermediate Feature-Level Attention (DR2)
- Although `ml/fusion/attention.py` implements a mathematically complete multi-head cross-attention fusion module with dimension projection and key-padding masks, its parameters remain randomly initialized.
- Because no jointly labelled 4-modality training corpus was available, the feature representations could not be optimized end-to-end.
- Production inference explicitly bypasses this module, using reliability-weighted decision-level fusion (`ml/fusion/engine.py`).

### 4.2 Uncalibrated Multimodal Fusion Scores (DR4)
- Temperature scaling parameters ($T$) were fit and evaluated for each of the five unimodal models on disjoint calibration splits.
- However, the final unified probability ($p_{\text{unified}}$) output by the fusion engine represents an ensemble of calibrated marginal probabilities, not a directly calibrated joint distribution. Fused-level ECE and Brier scores remain `NOT MEASURED`.

### 4.3 Modality Preprocessing & Boundary Constraints
1. **Text Preprocessing:**
   - Maximum sequence length is truncated to 512 subword tokens by RoBERTa's tokenizer. Articles exceeding ~350 words have their trailing paragraphs discarded.
   - The models were trained exclusively on English-language corpora; non-English text is not supported.
2. **Image Preprocessing:**
   - Face detection relies on OpenCV Haar Cascades (`haarcascade_frontalface_default.xml`). The detector exhibits sensitivity to extreme pose variations (yaw > 45°), heavy occlusion, or non-frontal profiles.
   - Forgery detection focuses predominantly on facial artifacts; background manipulations or non-facial splices may be overlooked.
3. **Video Preprocessing:**
   - 16 frames are uniformly sampled across the video duration. Brief, localized manipulation flashes (lasting only 1–3 frames) can be skipped entirely by uniform sampling.
4. **Audio Preprocessing:**
   - Audio is resampled to 16 kHz and trimmed or padded to a fixed 3.0-second window (157 STFT time bins). Longer audio tracks are currently truncated to their initial segment.

---

## 5. Production Lifecycle & Telemetry Constraints (DR8)

1. **Absence of Real-World Traffic Feeds:**
   - The automated drift detection suite (`ml/monitoring/drift_detector.py`) successfully implements Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests against baseline validation distributions (`baseline_posteriors.npy`).
   - However, because the system is deployed locally without an active public web endpoint, real streaming production drift is `NOT MEASURED`.
2. **Manual Model Retraining Execution:**
   - When drift thresholds are exceeded ($\text{PSI} > 0.25$ or $p < 0.05$), the system logs an alert to `models/retraining_queue.log`.
   - Automatic execution of retraining jobs, dataset re-ingestion, and zero-downtime model hot-swapping are deliberately not automated, as human oversight is required before deploying updated model weights.

---

## 6. Prohibited Overclaims & Research Non-Goals

To maintain strict academic integrity, the following claims are explicitly **rejected**:

- **NOT a "Truth Machine":** CheckReality AI does not verify factual reality, objective truth, or historical accuracy. It predicts the statistical likelihood of synthetic manipulation or text misinformation patterns based on training distributions.
- **NOT "Universal SOTA":** High empirical metrics (>99%) reflect held-out validation on known benchmarks (Kaggle, 140k Faces, FF++ C23, ASVspoof 2019) and do not imply universal superiority over all published literature under unseen conditions.
- **NOT Infallible Against Novel Generators:** Zero-day diffusion models (e.g. FLUX, Sora, ElevenLabs v3) generated after the collection of the training sets may present visual and acoustic artifacts that evade detection.
- **NOT a Certified Forensic Tool:** The system is explicitly designed for academic demonstration and decision-support, and its verdicts cannot be used as sole legal evidence in judicial proceedings.
