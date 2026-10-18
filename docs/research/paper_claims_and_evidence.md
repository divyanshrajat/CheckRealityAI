# Research Paper Claims & Empirical Evidence Mapping
## CheckReality AI — Multimodal Deepfake and Misinformation Detection System

**Document Version:** 1.0  
**Audit Date:** 18 October 2026  
**Document Purpose:** Verification of academic claims against reproducible experimental evidence, code, and test logs  
**Repository State:** Verified against Git Commit `4ac9f2f`

---

## 1. Academic Integrity & Claim Verification Protocol

To ensure full compliance with IEEE / ACM academic publishing ethics, every assertion intended for the research paper or project report must be rigorously validated against empirical data in the repository.

Each claim is evaluated across five criteria:
1. **Proposed Claim:** The initial or informal assertion.
2. **Supporting Artifact / Code:** The specific file path, test suite, or benchmark report validating the claim.
3. **Evidence Directness:**
   - **Direct:** The exact claim is directly measured and confirmed by local empirical runs.
   - **Partial:** A substantial portion of the claim is confirmed, but scope boundaries apply.
   - **Indirect:** Inferred from related proxy experiments.
   - **Absent:** No local code or empirical data supports the assertion.
4. **Conservative Academic Wording:** The exact revised text that must be used in the paper/thesis.
5. **Mandatory Limitations / Caveats:** Required qualifying context that must accompany the claim.

---

## 2. Claim-by-Claim Evidence Matrix

### Category A: Model Accuracy & Unimodal Performance

| Claim ID | Proposed Draft Claim | Supporting Artifact & Code | Evidence Directness | Conservative Academic Wording | Mandatory Accompanying Limitations |
|---|---|---|---|---|---|
| **CLM-ACC-01** | *"CheckReality AI achieves state-of-the-art 99.6% accuracy across all deepfake and misinformation modalities."* | `models/*/metrics.json`, verified via `python backend/test_integration.py` | **Partial** | *"On held-out validation partitions of the five benchmark datasets, the fine-tuned modality models achieved empirical accuracies ranging from 99.25% to 100.00%."* | Must explicitly state that these metrics represent within-dataset evaluation on specific benchmark splits (Kaggle, 140k Faces, FF++ C23, ASVspoof 2019) and cannot be extrapolated to universal in-the-wild content. |
| **CLM-ACC-02** | *"The system guarantees near-zero false positive rates for human-written text and authentic speech."* | `models/text_ai_detection/metrics.json` (FPR = 0.0), `models/audio_spoof/metrics.json` (FPR = 0.008) | **Direct** | *"On the curated evaluation split of AI vs Human Text, the RoBERTa model yielded zero false positives across 492 test samples, while the ResNet-18 audio classifier achieved a false positive rate of 0.8% on ASVspoof 2019 LA."* | Must disclose sample counts ($N=492$ and $N=500$) and acknowledge that real-world stylistic diversity may elevate FPR. |

---

### Category B: Multimodal Fusion & Attention Mechanism

| Claim ID | Proposed Draft Claim | Supporting Artifact & Code | Evidence Directness | Conservative Academic Wording | Mandatory Accompanying Limitations |
|---|---|---|---|---|---|
| **CLM-FUS-01** | *"The system employs an end-to-end trained intermediate cross-modal attention network to fuse representations."* | `ml/fusion/attention.py` vs `backend/app/inference.py` | **Absent** | *"While an intermediate cross-attention fusion architecture with key-padding masks was designed (`CrossModalAttentionFusion`), the operational system utilizes an empirical reliability-weighted decision fusion layer due to the absence of a jointly labelled 4-modality training corpus."* | **Do not claim trained cross-attention weights.** Explicitly state that intermediate attention is an architectural proposal evaluated in future work. |
| **CLM-FUS-02** | *"The decision engine dynamically handles missing modalities across all 15 possible input subsets."* | `ml/fusion/gating.py`, `benchmarks/subset_coverage_report.json` | **Direct** | *"The decision gating mechanism dynamically renormalizes modality reliability weights across all 15 non-empty power-set combinations of the four modalities, ensuring that the convex weight sum strictly equals 1.0 for every active subset."* | Must note that multi-stream evaluation was conducted on synthetic bundles of unimodal validation records, as naturally aligned 4-modality files were unavailable. |
| **CLM-FUS-03** | *"Multimodal fusion consistently outperforms every individual unimodal detector."* | `benchmarks/subset_coverage_report.json` | **Partial** | *"In synthetic multi-stream bundle testing, combining four modalities achieved an empirical ROC-AUC of 0.9510, substantially exceeding the individual performance of text (0.5686) and audio (0.7157), while matching or refining video performance (0.7794)."* | Must state that synthetic pairings do not model real-world contextual misalignment and that multimodal gains depend on the specific modality combination. |

---

### Category C: Explainability & Confidence Calibration

| Claim ID | Proposed Draft Claim | Supporting Artifact & Code | Evidence Directness | Conservative Academic Wording | Mandatory Accompanying Limitations |
|---|---|---|---|---|---|
| **CLM-XAI-01** | *"CheckReality AI provides fully faithful, causal explanations of deepfake decisions."* | `ml/fusion/engine.py`, `backend/app/inference.py` (test suite 4/5) | **Partial** | *"The system implements a two-layer explainability framework providing modality-native spatial, temporal, and token saliency (Layer 1) alongside signed weighted-deviation cross-modal attribution (Layer 2)."* | Saliency maps (Grad-CAM, gradient token attributions) are local post-hoc gradient approximations and do not constitute formal causal proofs of model reasoning. |
| **CLM-CAL-01** | *"The multimodal platform outputs perfectly calibrated Bayesian posterior probabilities."* | `training/fit_calibration.py`, `models/calibration_benchmark.json` | **Partial** | *"Post-hoc temperature scaling was fitted on disjoint calibration splits for each individual modality, reducing Expected Calibration Error on video from 0.4185 to 0.3637 and achieving Brier scores below 0.01 on image, audio, and text models."* | Must state that temperature scaling is applied per modality. The final fused score is an ensemble of calibrated marginal probabilities, but fusion-level calibration parameters remain uncalibrated on joint data. |

---

### Category D: Robustness & Cross-Dataset Generalization

| Claim ID | Proposed Draft Claim | Supporting Artifact & Code | Evidence Directness | Conservative Academic Wording | Mandatory Accompanying Limitations |
|---|---|---|---|---|---|
| **CLM-ROB-01** | *"The models are thoroughly evaluated on independent external benchmarks including Celeb-DF v2 and In-the-Wild datasets."* | `training/eval_cross_dataset.py`, `models/cross_dataset_benchmark.json` | **Absent** | *"External cross-dataset evaluation on Celeb-DF v2, FaceForensics++ C40, In-the-Wild Audio, and LIAR was not measured due to local repository and licensing constraints; model robustness was instead evaluated via controlled input perturbation stress tests."* | Independent cross-dataset transfer must be marked `NOT_MEASURED`. Do not cite external literature numbers as if they were measured in this codebase. |
| **CLM-ROB-02** | *"CheckReality AI demonstrates quantifiable resilience to common real-world transmission degradations."* | `models/cross_dataset_benchmark.json` | **Direct** | *"Empirical stress testing demonstrated that under heavy JPEG compression ($Q=50$), image classification accuracy degraded from 98.6% to 84.6% ($\Delta \text{AUC} = -0.0095$), while audio downsampling to an 8 kHz telephony codec reduced accuracy from 99.67% to 95.00% ($\Delta \text{AUC} = -0.0103$)."* | Must report the exact accuracy drops alongside AUC retention; acknowledge that lossy compression degrades fine facial texture cues. |

---

### Category E: Computational Efficiency & Real-Time Deployment

| Claim ID | Proposed Draft Claim | Supporting Artifact & Code | Evidence Directness | Conservative Academic Wording | Mandatory Accompanying Limitations |
|---|---|---|---|---|---|
| **CLM-EFF-01** | *"All models run in real time with sub-millisecond latencies across edge devices."* | `benchmarks/latency_size_budget_report.json` | **Direct** (with exact numbers) | *"Benchmarking on an NVIDIA GeForce GTX 1650 GPU revealed P50 inference latencies of 3.32 ms for audio, 29.40 ms for image (with TTA), 29.91 ms for video (16 frames), and 63.54 ms for text. The decision fusion engine executes in 0.039 ms (21,900 calls/sec). On CPU, text models require ~470 ms (P50)."* | Edge mobile viability is a future work objective; published numbers reflect a desktop/laptop Intel i5 and GTX 1650 environment at batch size 1. |
| **CLM-EFF-02** | *"The model suite fits within standard workstation memory constraints."* | `benchmarks/latency_size_budget_report.json` | **Direct** | *"The total on-disk model footprint is 1.09 GB (146.4 MB for vision/audio models; 957.8 MB for two RoBERTa backbones). Peak GPU VRAM utilization remained below 1.1 GB during single-model execution."* | Concurrent execution of all five models on GPU exceeds the 4GB VRAM ceiling; sequential execution or CPU offloading is required under 4GB hardware. |

---

### Category F: Model Lifecycle & Operational Monitoring

| Claim ID | Proposed Draft Claim | Supporting Artifact & Code | Evidence Directness | Conservative Academic Wording | Mandatory Accompanying Limitations |
|---|---|---|---|---|---|
| **CLM-OPS-01** | *"The platform features fully automated online retraining in production traffic."* | `models/model_registry.json`, `models/retraining_queue.log`, `ml/monitoring/drift_detector.py` | **Partial** | *"The architecture incorporates a drift detection suite computing Population Stability Index (PSI) and Kolmogorov-Smirnov statistics against baseline validation distributions, automatically logging retraining triggers when drift thresholds are breached."* | Must state that actual streaming production drift is `NOT_MEASURED` due to the lack of live production traffic, and retraining execution requires manual review. |

---

## 3. Disallowed Phrasing & Replacement Guidelines

To ensure the highest academic integrity in viva presentations, project reports, and research papers, the following phrasing substitutions must be strictly enforced:

| Disallowed / Sensationalist Phrasing | Mandatory Academically Defensible Alternative | Rationale |
|---|---|---|
| *"Truth detector" / "Automated lie detector"* | *"Statistical authenticity and manipulation classifier"* | The system detects learned statistical artifacts, not objective factual truth. |
| *"State-of-the-art across all modalities"* | *"Achieved high competitive accuracy (>99%) on five standard benchmark splits"* | SOTA requires comprehensive, verified benchmarking against dozens of contemporary published works under identical protocols. |
| *"Universal deepfake detection"* | *"Deepfake detection evaluated on known manipulation techniques"* | Generalization to novel or unseen zero-day architectures remains unverified. |
| *"End-to-end deep multimodal attention"* | *"Empirical reliability-weighted decision fusion with architectural specification of cross-attention"* | Attention weights are untrained; operational fusion is decision-level. |
| *"Production-grade autonomous MLOps"* | *"Monitored prototype with baseline drift detection telemetry and logged triggers"* | True production operations require active live traffic and automated deployment pipelines. |
