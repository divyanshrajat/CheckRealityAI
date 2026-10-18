# Reproducible Experimental & Evaluation Protocol
## CheckReality AI — Multimodal Deepfake and Misinformation Detection System

**Document Version:** 1.0  
**Audit Date:** 18 October 2026  
**Document Purpose:** Complete specification of datasets, preprocessing pipelines, training splits, calibration partitions, and evaluation protocols  
**Repository State:** Verified against Git Commit `4ac9f2f`

---

## 1. Experimental Integrity & Reproducibility Standards

To guarantee strict reproducibility in all academic deliverables (thesis, viva examination, and research paper), this protocol defines the exact conditions under which models were trained, calibrated, and evaluated.

All random seeds are fixed to **42** across NumPy, PyTorch, and Python random modules. Zero test-set data leakage was maintained across training, calibration, and validation phases.

---

## 2. Dataset Specifications & Partitioning Protocol

The system utilizes five established open-access benchmark datasets corresponding to the target modalities:

| Dataset Name | Target Task | Primary Source / Citation | Original Modality | Class Definitions | Local Partition Strategy | Sample Counts (Train / Val / Test) | Licensing / Terms |
|---|---|---|---|---|---|---|---|
| **140k Real and Fake Faces** | Face Forgery Detection | Kaggle (Yonatan Elul / StyleGAN) | Image (JPEG/PNG) | 0: Real Face, 1: Fake (StyleGAN) | Disjoint stratified split | 24,000 Train / 2,000 Val / 1,000 Test | CC BY-SA 4.0 |
| **ASVspoof 2019 LA** | Synthetic Speech Detection | ASVspoof Consortium (Yamagishi et al., 2019) | Audio (FLAC, 16kHz) | 0: Bonafide (Real), 1: Spoof (TTS/VC) | Disjoint speaker split | 4,000 Train / 1,000 Val / 500 Test | ASVspoof Open License |
| **FaceForensics++ (FF++ C23)** | Deepfake Video Detection | Rossler et al. (ICCV 2019) | Video (MP4, H.264, C23) | 0: Real Video, 1: Manipulated Video | Disjoint video sequence split | 320 Train / 80 Val / 40 Test | Non-commercial academic license |
| **Fake & Real News Dataset** | Misinformation Detection | Kaggle / University of Victoria (ISOT) | Text (Articles) | 0: Real News (Reuters), 1: Fake News | Disjoint chronological split | 8,000 Train / 2,000 Val / 1,000 Test | Open Database License (ODbL) |
| **AI vs Human Text Dataset** | AI-Generated Text Detection | Kaggle / DAIGT V2 | Text (Essays/Prompts) | 0: Human-Written, 1: AI-Generated (GPT/Claude) | Disjoint author/prompt split | 4,000 Train / 983 Val / 492 Test | Open Access |

### 2.1 Disjoint Split & Leakage Prevention Rules
1. **Video Temporal Leakage:** Frames extracted from the same video are strictly confined to the same partition. A video never contributes frames to both training and test splits.
2. **Audio Speaker Leakage:** ASVspoof speakers in the evaluation set do not appear in the training split.
3. **Calibration Isolation:** In accordance with DR4 calibration standards, the calibration split ($0:N/2$) and the validation evaluation split ($N/2:N$) are mutually exclusive partitions of the held-out manifest, preventing post-hoc overfitting.

---

## 3. Data Preprocessing & Feature Extraction Pipelines

### 3.1 Text Processing Pipeline
- **Input:** Raw Unicode text string or `.txt` file.
- **Normalization:** Lowercase stripping, URL replacement with token `<URL>`, and whitespace normalization.
- **Tokenization:** RoBERTa Byte-Pair Encoding (BPE) tokenizer (`roberta-base`).
- **Sequence Truncation:** Maximum sequence length $L = 512$ tokens. Padding with `<pad>` token up to sequence length.
- **Tensors:** `input_ids` ($1 \times 512$) and `attention_mask` ($1 \times 512$).

### 3.2 Image Processing Pipeline
- **Input:** Standard image files (`.jpg`, `.jpeg`, `.png`, `.webp`).
- **Face Detection:** OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`). If a face is detected with confidence bounding box $[x, y, w, h]$, crop region with $15\%$ context margin. If no face is detected, fall back to center square crop.
- **Spatial Normalization:** Bilinear resize to $224 \times 224$ pixels.
- **Normalization:** ImageNet mean $[0.485, 0.456, 0.406]$ and standard deviation $[0.229, 0.224, 0.225]$.
- **Test-Time Augmentation (TTA):** Forward pass on original image and horizontal-flip image, averaging logits:
  $$p_{\text{TTA}} = \frac{1}{2}\left[\sigma(f(x)) + \sigma(f(\text{flip}(x)))\right]$$

### 3.3 Video Processing Pipeline
- **Input:** Video container (`.mp4`, `.mov`, `.avi`).
- **Temporal Subsampling:** Uniform extraction of $T = 16$ equidistant frames spanning the full video duration using OpenCV `VideoCapture`.
- **Face Tracking / Crop:** Haar Cascade face detection applied to each sampled frame with temporal bounding-box smoothing to prevent jitter.
- **Tensor Formatting:** Dimension $1 \times 16 \times 3 \times 224 \times 224$.
- **Sequence Modeling:** EfficientNet-B0 extracts spatial features ($16 \times 1280$), projected to 256, feeding a 2-layer bidirectional LSTM ($h = 128$) with mean pooling across timesteps.

### 3.4 Audio Processing Pipeline
- **Input:** Digital audio container (`.wav`, `.flac`, `.mp3`).
- **Resampling:** Monophonic downsampling to $16,000$ Hz via `librosa`.
- **Temporal Trimming:** Fixed window of 3.0 seconds ($48,000$ samples). Signals shorter than 3.0s are zero-padded; signals longer are center-cropped.
- **Acoustic Transform:** Short-Time Fourier Transform (STFT) with FFT window $N_{\text{fft}} = 1024$, hop length $H = 512$, transformed to 128 Mel filterbanks spanning $0 - 8000$ Hz.
- **Output Spectrogram:** Normalized log-Mel spectrogram tensor of dimension $1 \times 3 \times 128 \times 157$ (replicated across 3 RGB channels for ResNet-18 backbone compatibility).

---

## 4. Hardware, Software & Environmental Setup

- **Host Machine:** Lenovo Workstation / Windows 11 (build 10.0.26200)
- **CPU:** Intel64 Family 6 Model 140 Stepping 2, 4 Cores / 8 Threads, 15.79 GB RAM
- **GPU:** NVIDIA GeForce GTX 1650 (Laptop), 3.99 GB VRAM, Driver 572.16
- **CUDA Runtime:** CUDA 12.4, cuDNN 9.1
- **Deep Learning Framework:** PyTorch 2.6.0+cu124, TorchVision 0.21.0+cu124, Torchaudio 2.6.0+cu124
- **NLP Library:** Hugging Face Transformers 5.0.0, Tokenizers 0.21.0
- **Computer Vision & Signal Processing:** OpenCV 4.10.0, Librosa 0.10.2
- **Numerical & ML Utilities:** NumPy 2.4.2, SciPy 1.15.2, scikit-learn 1.6.1

---

## 5. Mathematical Metric Formulations

### 5.1 Classification Metrics
Let $TP, TN, FP, FN$ denote true positives, true negatives, false positives, and false negatives for class 1 (Fake / Spoof):

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 TP}{2 TP + FP + FN}$$

$$\text{False Positive Rate (FPR)} = \frac{FP}{FP + TN}$$

$$\text{ROC-AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt$$

$$\text{Equal Error Rate (EER)} = \text{FPR}(\tau^*) \quad \text{where } \text{FPR}(\tau^*) = 1 - \text{TPR}(\tau^*)$$

### 5.2 Confidence Calibration Metrics
For $N$ samples partitioned into $B=15$ confidence bins:

$$\text{ECE} = \sum_{b=1}^B \frac{|B_b|}{N} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$

$$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^N \left(p_i - y_i\right)^2, \quad y_i \in \{0, 1\}$$

$$\text{Negative Log-Likelihood (NLL)} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$

### 5.3 Distribution Drift Metrics
For baseline distribution $P$ and incoming batch distribution $Q$ over $K=10$ histogram bins:

$$\text{PSI} = \sum_{k=1}^K \left(P_k - Q_k\right) \ln\left(\frac{P_k}{Q_k}\right)$$

$$D_{\text{KS}} = \sup_x \left| F_P(x) - F_Q(x) \right|$$

---

## 6. Controlled Robustness Perturbation Protocols (DR6)

To rigorously evaluate model boundary resilience in the absence of external cross-dataset benchmarks, four controlled real-world degradation protocols were executed:

1. **JPEG Compression Stress Test (Image):**
   - Protocol: Compress clean validation images to JPEG format with quality factor $Q = 50$ (simulating aggressive social media recompression), re-decode, and re-classify.
   - Evaluated Sample Count: $N = 500$ held-out images.
2. **Telephony Codec Downsampling (Audio):**
   - Protocol: Resample 16 kHz studio audio to 8 kHz (G.711 / narrowband telephony codec standard) using bandlimited sinc interpolation, then upsample back to 16 kHz for Mel feature extraction.
   - Evaluated Sample Count: $N = 300$ held-out utterances.
3. **Temporal Frame Dropout (Video):**
   - Protocol: Randomly drop $25\%$ of video frames (4 out of 16 frames zeroed out), simulating network packet loss or dropped frames during video streaming.
   - Evaluated Sample Count: $N = 80$ video sequences.
4. **Typographic Noise Injection (Text):**
   - Protocol: Randomly inject character substitutions, deletions, and transpositions at a $5\%$ character error rate (CER), simulating mobile keyboard typos and social media spelling informalities.
   - Evaluated Sample Count: $N = 500$ news articles.

---

## 7. Latency and Efficiency Measurement Protocol (DR7)

- **Execution Protocol:** 10 unmeasured warmup runs to ensure CUDA kernel compilation and PyTorch JIT cache population, followed by 100 timed iterations.
- **Timing Boundary:** Precise synchronization via `torch.cuda.synchronize()` before and after model inference to measure exact hardware kernel execution times:
  ```python
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  output = model(input_tensor)
  torch.cuda.synchronize()
  latency_ms = (time.perf_counter() - t0) * 1000.0
  ```
- **Reported Statistics:** Mean, Median (P50), 95th Percentile (P95), and 99th Percentile (P99).
- **Batch Size:** Fixed at $B = 1$ to simulate interactive single-request web application usage.
- **Fusion Engine Profiling:** 10,000 consecutive calls to `multimodal_fusion()` across all 4 modalities to measure throughput.
