<div align="center">

# CheckReality AI

**Unified Multimodal Forensic Verification & Deepfake Detection Framework**

B.Tech Major Project (2026) · Developed by **Divyansh Rajat** (Roll No. 2300358) · Baba Banda Singh Bahadur Engineering College (IKGPTU) · Department of Computer Science & Engineering · Supervised by **Dr. Manpreet Kaur**

`PyTorch 2.5` · `FastAPI` · `React 18` · `TypeScript` · `Celery` · `Redis` · `MongoDB` · `Transformers` · `OpenCV` · `Docker Compose`

</div>

---

## Contents

1. [What is CheckReality AI?](#1-what-is-checkreality-ai)
2. [System Architecture: Ingestion to Forensic Report](#2-system-architecture-ingestion-to-forensic-report)
   - [Clean Repository Directory Layout](#clean-repository-directory-layout)
3. [Model Suite & Organic Fine-Tuned Benchmarks](#3-model-suite--organic-fine-tuned-benchmarks)
4. [Multimodal Fusion Engine & Explainable AI (XAI)](#4-multimodal-fusion-engine--explainable-ai-xai)
5. [Running the Application](#5-running-the-application)
6. [API Specification & Ingestion Workflows](#6-api-specification--ingestion-workflows)
7. [Forensic Authenticity Report & Interactive UI](#7-forensic-authenticity-report--interactive-ui)
8. [Security, Evidence Integrity & Privacy Model](#8-security-evidence-integrity--privacy-model)
9. [Verification, Testing & Integration Suite](#9-verification-testing--integration-suite)
10. [Current Status, Limitations & Out of Scope](#10-current-status-limitations--out-of-scope)
11. [Future Scope and Roadmap](#11-future-scope-and-roadmap)
12. [Documentation Directory](#12-documentation-directory)
13. [Academic Attribution & Project Author](#13-academic-attribution--project-author)
14. [License](#14-license)

---

## 1. What is CheckReality AI?

The rapid proliferation of generative artificial intelligence has made creating hyper-realistic deepfake images, fabricated videos, synthetic voice clones, and automated misinformation virtually frictionless. While commodity synthesis tools continue to advance, forensic detection tools remain fragmented: existing solutions are typically isolated to a single modality (e.g. text-only or image-only), treat models as uninterpretable black boxes, fail to expose localization artifacts, and degrade severely on compressed real-world media.

**CheckReality AI** is an end-to-end, multimodal forensic platform designed to detect manipulation across **all four digital content modalities: Text, Image, Video, and Audio**. 

Rather than relying on isolated heuristic checks, CheckReality AI couples deep convolutional and transformer backbones with a **Bayesian Multimodal Fusion Engine** and a **multi-engine Explainable AI (XAI) suite**. It translates raw digital media into an authoritative, human-readable **Forensic Authenticity Report** containing unified credibility scores, cross-modal agreement metrics, localized Grad-CAM heatmaps, acoustic spectral distortion windows, and token-level attribution rankings.

> CheckReality AI is designed as a rigorous decision-support tool for intelligence analysts, fact-checkers, journalists, and forensic examiners.

### Key Highlights

| Pillar | Capability | Technical Implementation |
|---|---|---|
| **Full Modality Coverage** | 4-in-1 unified verification platform | Deepfake Images, Temporal Deepfake Videos, Audio Spoofs/Voice Clones, and AI Misinformation/Essays |
| **Organic 99+% Tier** | Fine-tuned on standard benchmarks | Rigorously trained without data leakage: ASVspoof 2019, FaceForensics++ C23, 140k Faces, and Real/Fake News |
| **Multimodal Bayesian Fusion** | Cross-modal agreement & risk indexing | Calibrated reliability weighting, cross-stream discrepancy detection (e.g. audio dubbed onto real video), and risk tiering |
| **Genuine Explainable AI (XAI)** | Full transparency, zero black boxes | Real gradient backpropagation: Visual Grad-CAM heatmaps, acoustic harmonic saliency, and token-level gradient attribution |
| **Production Ready** | Single-terminal containerized execution | Async FastAPI gateway, Celery task workers, isolated MongoDB/Redis brokers, and responsive 16:9 React dashboard |

---

## 2. System Architecture: Ingestion to Forensic Report

CheckReality AI implements a clean separation between **Ingestion Gateway**, **Asynchronous Task Workers**, **Specialized Neural Ensembles**, and the **Decision & XAI Layer**.

```mermaid
flowchart TD
  subgraph INGEST["1. INGESTION & DISPATCH"]
    U["User / API Client"] --> G["FastAPI Ingestion Gateway (:8001)"]
    G --> V["Media Store / Temporary Vault"]
    G --> Q["Redis Message Broker (:6381)"]
    Q --> W["Celery Asynchronous Workers"]
  end

  subgraph INFERENCE["2. SPECIALIZED NEURAL BACKBONES (CUDA)"]
    W -->|Image| M_IMG["EfficientNet-B3 (TTA)<br/>140k Faces Benchmark"]
    W -->|Video| M_VID["EfficientNet-B0 + 2L LSTM<br/>FaceForensics++ C23"]
    W -->|Audio| M_AUD["ResNet-18 (SpecAugment)<br/>ASVspoof 2019 LA"]
    W -->|Text| M_TXT["Dual RoBERTa-base<br/>Misinfo & AI Essay Detector"]
  end

  subgraph XAI_LAYER["3. EXPLAINABLE AI (XAI) ENGINES"]
    M_IMG --> X_IMG["Visual Grad-CAM<br/>features[-1] Conv Saliency"]
    M_VID --> X_VID["Temporal Grad-CAM<br/>Peak Feature Variance Frame"]
    M_AUD --> X_AUD["Acoustic Saliency<br/>Mel Spectral Gradient Profiler"]
    M_TXT --> X_TXT["Token Attribution<br/>Embedding Gradient Saliency"]
  end

  subgraph FUSION["4. MULTIMODAL FUSION & DECISION"]
    M_IMG & M_VID & M_AUD & M_TXT --> F_ENG["Bayesian Multimodal Fusion Engine<br/>ml/fusion/engine.py"]
    F_ENG --> AGR["Cross-Modal Agreement Index"]
    F_ENG --> RSK["Risk Severity Matrix (Low/Mod/High/Critical)"]
    F_ENG --> SYN["Forensic Synthesis Summary"]
  end

  subgraph REPORT["5. PRESENTATION & STORAGE"]
    X_IMG & X_VID & X_AUD & X_TXT & AGR & RSK & SYN --> REP["Forensic Authenticity Report"]
    REP --> DB[("MongoDB (:27019)")]
    REP --> UI["React 18 Forensic Studio (:5173 / :8001)"]
  end
```

### Technology Stack

| Layer | Technology | Version / Specification |
|---|---|---|
| **Deep Learning Engine** | PyTorch, TorchAudio, TorchVision, HuggingFace | PyTorch 2.5.1+cu124, Transformers 4.49, Librosa 0.10 |
| **Backend API Gateway** | FastAPI, Uvicorn, Celery, Pydantic | Python 3.11/3.13, Celery 5.3, Uvicorn ASGI |
| **Database & Broker** | MongoDB, Redis | MongoDB 6.0 (Port 27019), Redis 7-Alpine (Port 6381) |
| **Frontend UI** | React 18, Vite, TypeScript, Tailwind CSS | React 18.3, Vite 8.3, Lucide / Material Symbols, Tailwind |
| **Deployment** | Docker Compose, PowerShell Orchestrator | Isolated Bridge Network (`checkreality-net`), `start_local.ps1` |

### Clean Repository Directory Layout

```text
CheckRealityAI/
├── backend/                  # Asynchronous FastAPI Gateway & Task Distribution
│   ├── app/                  # REST endpoints (auth, upload, history, inference, database)
│   ├── worker.py             # Celery background worker for audio/video pipeline tasks
│   ├── test_integration.py   # Automated 5/5 end-to-end multi-modal test suite
│   ├── requirements.txt      # Backend Python dependencies (FastAPI, PyTorch, Celery)
│   └── Dockerfile            # Container definition for backend service
├── Datasets/                 # Verified academic training corpuses (zero leakage)
│   ├── audio/                # ASVspoof 2019 Logical Access (LA) protocols & FLAC samples
│   ├── image/                # 140k Real & Fake Faces balanced subsets
│   ├── text/                 # Sanitized Fake/True News & AI vs Human essay corpuses
│   └── video/                # FaceForensics++ C23 compressed manipulation sequences
├── docs/                     # Academic & Engineering Specification Documents
│   ├── PRD.md                # Product Requirements Document
│   ├── SRS.md                # IEEE 830-style Software Requirements Specification
│   ├── TRD.md                # Technical Requirements & Database Schema
│   ├── IMPLEMENTATION.md     # Architectural Roadmap & Milestones
│   └── TASK.md               # Implementation Backlog & Work Tracking
├── frontend/                 # Forensic Studio Web Application
│   ├── src/
│   │   ├── api/              # Axios HTTP client & API route definitions
│   │   ├── components/       # Reusable UI primitives (ProtectedRoute, layout guards)
│   │   ├── context/          # React AuthContext session & JWT state management
│   │   ├── pages/            # Viewports (Landing, Upload, Report, History, Transparency)
│   │   ├── App.tsx           # Client router & navigation root
│   │   └── main.tsx          # React DOM initialization
│   ├── package.json          # Frontend dependencies (React 18, Vite, Tailwind CSS)
│   └── vite.config.ts        # Vite build & development proxy configuration
├── ml/                       # Core Machine Learning & Forensic Intelligence
│   ├── audio/adapter.py      # ResNet-18 Log-Mel Spectrogram inference adapter
│   ├── image/adapter.py      # EfficientNet-B3 TTA face manipulation adapter
│   ├── text/adapter.py       # Dual RoBERTa sequence classification adapters
│   ├── video/adapter.py      # EfficientNet-B0 + 2L LSTM temporal sequence adapter
│   ├── fusion/engine.py      # Bayesian Multimodal Fusion Engine & Risk Severity index
│   └── xai/engine.py         # Explainable AI: Grad-CAM, Spectral Saliency & Token Attributions
├── models/                   # Fine-Tuned Model Weights & Immutable Metric Records
│   ├── audio_spoof/          # audio_cnn.pth (99.60% Acc) & metrics.json
│   ├── image_deepfake/       # efficientnet_b3.pth (99.40% Acc) & metrics.json
│   ├── text_ai_detection/    # fine-tuned RoBERTa-base (100.00% Acc) & metrics.json
│   ├── text_misinformation/  # fine-tuned RoBERTa-base (99.95% Acc) & metrics.json
│   └── video_deepfake/       # efficientnet_lstm.pth (99.25% Acc) & metrics.json
├── training/                 # Deep Learning Pipelines & Experiment Artifacts
│   ├── experiments/          # Immutable JSON execution logs per training run
│   ├── manifests/            # Zero-leakage train/validation CSV manifests & feature tensors
│   └── scripts/              # Standalone production training scripts (train_*.py)
├── .env                      # Environment variables & service connection strings
├── docker-compose.yml        # Multi-container orchestration (Backend, Worker, Mongo, Redis)
├── start_local.ps1           # Windows native single-command development launcher
└── README.md                 # Master system documentation & benchmark specification
```

---

## 3. Model Suite & Organic Fine-Tuned Benchmarks

Every model in CheckReality AI is fine-tuned on verified public academic datasets, validated against strict **zero-leakage** partitions, and recorded in immutable metric records ([`models/*/metrics.json`](models/)).

### Final Verified Benchmark Matrix

| Modality / Domain | Model Architecture | Training Script | Dataset Source | Training Samples | Validation Samples | Total Samples | Accuracy | Precision | Recall | F1-Score | ROC-AUC | FPR | FNR |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Audio Spoofing** | ResNet-18 (SpecAugment) | [`train_audio_spoof.py`](training/scripts/train_audio_spoof.py) | ASVspoof 2019 LA | 4,000 | 1,000 | 5,000 | **99.60%** | 0.9940 | 0.9980 | **0.9960** | **0.9999** | 0.60% | 0.20% |
| **Image Deepfake** | EfficientNet-B3 (TTA) | [`train_image_deepfake.py`](training/scripts/train_image_deepfake.py) | 140k Real & Fake Faces | 24,000 | 2,000 | 26,000 | **99.40%** | 0.9950 | 0.9930 | **0.9940** | **0.9998** | 0.50% | 0.70% |
| **Text Misinformation** | RoBERTa-base | [`train_text_misinformation.py`](training/scripts/train_text_misinformation.py) | Fake & Real News (Sanitized) | 8,000 | 2,000 | 10,000 | **99.95%** | 0.9990 | 1.0000 | **0.9995** | **1.0000** | 0.10% | 0.00% |
| **Text AI Detection** | RoBERTa-base | [`train_text_ai_detection.py`](training/scripts/train_text_ai_detection.py) | AI vs Human (Sanitized) | 4,000 | 983 | 4,983 | **100.00%** | 1.0000 | 1.0000 | **1.0000** | **1.0000** | 0.00% | 0.00% |
| **Video Deepfake** | EfficientNet-B0 + 2L LSTM | [`train_video_deepfake.py`](training/scripts/train_video_deepfake.py) | FaceForensics++ C23 | 3,200 | 400 | 3,600 | **99.25%** | 0.9901 | 0.9950 | **0.9925** | **0.9994** | 0.99% | 0.50% |


---

### Detailed Model Profiles

#### 1. Audio Voice Cloning & Spoofing Detection
- **Architecture**: Residual Neural Network (`ResNet-18`) with 128-band Mel-Spectrogram acoustic front-end.
- **Audio Preprocessing**: Unified 16 kHz resampler, 2048 FFT window, 512 hop length, log-power scaling (`AmplitudeToDB`).
- **Data Augmentation**: Time masking and frequency masking via **SpecAugment**, simulating compression noise and transmission jitter.
- **Dataset**: ASVspoof 2019 Logical Access (LA) evaluation benchmark (4,000 train / 1,000 validation).
- **Artifact**: [`models/audio_spoof/audio_cnn.pth`](models/audio_spoof/audio_cnn.pth) (42.7 MB).

#### 2. Image Deepfake & Synthetic Face Detection
- **Architecture**: `EfficientNet-B3` convolutional network initialized with ImageNet weights and fine-tuned end-to-end on synthetic facial boundaries.
- **Inference Strategy**: Test-Time Augmentation (TTA) with multi-scale horizontal flipping to suppress spatial pose variance.
- **Dataset**: 140k Real and Fake Faces balanced dataset (24,000 train / 2,000 validation).
- **Artifact**: [`models/image_deepfake/efficientnet_b3.pth`](models/image_deepfake/efficientnet_b3.pth) (43.3 MB).

#### 3. Text Misinformation & Fake News Classification
- **Architecture**: `RoBERTa-base` sequence classification model (125M parameters, 12 attention heads).
- **Leakage Prevention**: Complete lexical scrubbing of publisher stamps (e.g. `Reuters`, `AP`), publication metadata, and author lines to prevent spurious correlation shortcuts.
- **Dataset**: Sanitized News Corpus (8,000 train / 2,000 validation).
- **Artifact**: [`models/text_misinformation/`](models/text_misinformation/) (HuggingFace format, 498 MB).

#### 4. AI-Generated Text & Essay Detection
- **Architecture**: `RoBERTa-base` fine-tuned for stylistic perplexity and syntactical repetition signatures.
- **Leakage Prevention**: Stripped uniform formatting artifacts, prompt template echoes, and artificial preamble tags.
- **Dataset**: Expanded authentic human essays vs. advanced LLM outputs (4,000 train / 983 validation).
- **Artifact**: [`models/text_ai_detection/`](models/text_ai_detection/) (HuggingFace format, 498 MB).

#### 5. Video Deepfake & Manipulation Detection
- **Architecture**: Spatial `EfficientNet-B0` feature extractor combined with a **2-layer Temporal LSTM Head** (`hidden_dim=256`, `dropout=0.3`).
- **Temporal Modeling**: Samples 16 uniformly spaced keyframes across video duration, encoding spatial feature vectors (`[16, 1280]`) and analyzing inter-frame blending incoherence.
- **Dataset**: FaceForensics++ C23 compressed benchmark across all 6 manipulation families (*Deepfakes, Face2Face, FaceSwap, FaceShifter, NeuralTextures, DeepFakeDetection*).
- **Artifact**: [`models/video_deepfake/efficientnet_lstm.pth`](models/video_deepfake/efficientnet_lstm.pth) (23.9 MB).

---

## 4. Multimodal Fusion Engine & Explainable AI (XAI)

### Multimodal Fusion Engine (`ml/fusion/engine.py`)

When evaluating multi-stream digital media (e.g., a video containing speech audio, or an article with accompanying images), CheckReality AI fuses predictions using an empirical **Bayesian Multimodal Decision Fusion Pipeline**:

#### 1. Input Modality Posteriors
For each active modality $m \in \mathcal{M} \subseteq \{\text{video}, \text{audio}, \text{image}, \text{text}\}$, the dedicated fine-tuned neural backbone outputs a manipulation probability $p_m \in [0, 1]$:

$$
p_m = P(Y = \text{Fake} \mid \mathbf{x}_m)
$$

#### 2. Empirical Reliability Priors
Modalities are assigned base reliability weights $w_m^{(0)}$ proportional to their benchmark discriminative capacity:

$$
w_{\text{video}}^{(0)} = 0.35, \quad w_{\text{audio}}^{(0)} = 0.25, \quad w_{\text{image}}^{(0)} = 0.25, \quad w_{\text{text}}^{(0)} = 0.15
$$

#### 3. Dynamic Confidence Scaling
If any single stream detects manipulation or authenticity with extreme certainty ($p_m > 0.92$ or $p_m < 0.08$), its effective weight $w_m$ is dynamically scaled by **$+25\%$** to let unambiguous forensic cues dominate the decision:

$$
w_m = \begin{cases} 
1.25 \times w_m^{(0)} & \text{if } p_m > 0.92 \;\lor\; p_m < 0.08 \\
w_m^{(0)} & \text{otherwise}
\end{cases}
$$

#### 4. Unified Posterior Expectation & Verdict
The unified manipulation score $\hat{P}_{\text{unified}}$ is computed via normalized convex Bayesian expectation:

$$
\hat{P}_{\text{unified}} = \frac{\sum_{m \in \mathcal{M}} w_m \cdot p_m}{\sum_{m \in \mathcal{M}} w_m}
$$

The final classification verdict $\hat{Y} \in \{\text{Real}, \text{Fake}\}$ is determined by the Bayes optimal threshold ($\tau = 0.50$):

$$
\hat{Y} = \begin{cases} 
\text{Fake} & \text{if } \hat{P}_{\text{unified}} > 0.50 \\
\text{Real} & \text{if } \hat{P}_{\text{unified}} \le 0.50
\end{cases}
$$

#### 5. Cross-Modal Agreement Index (Discrepancy Detection)
To expose **cross-modal attacks** (e.g. authentic video footage spliced with cloned/synthesized speech), an inter-stream consensus score $A_{\text{consensus}} \in [0, 1]$ measures dispersion across modality predictions:

$$
A_{\text{consensus}} = \max\left(0, \; 1.0 - \frac{\sigma(\{p_m\}_{m \in \mathcal{M}})}{\sigma_{\max}}\right)
$$

where $\sigma(\{p_m\})$ is the standard deviation across modality probabilities, and $\sigma_{\max} = 0.50$ represents the theoretical maximum dispersion for bounded values in $[0, 1]$. A low consensus score flags targeted replacement attacks.

#### 6. Actionable Risk Severity Classification
Rather than relying solely on raw probabilities, forensic threat tiers integrate both unified confidence $\hat{P}_{\text{unified}}$ and cross-modal agreement $A_{\text{consensus}}$:

| Risk Severity Tier | Unified Score ($\hat{P}_{\text{unified}}$) | Cross-Modal Agreement ($A_{\text{consensus}}$) | Forensic Interpretation |
| :--- | :---: | :---: | :--- |
| 🔴 **CRITICAL** | $\ge 80\%$ | Any | High-confidence synthetic manipulation across examined streams |
| 🟠 **HIGH** | $60\% - 79.9\%$ | Any | Strong manipulation markers detected in primary media |
| 🟡 **HIGH (DISCREPANCY DETECTED)** | $40\% - 59.9\%$ | $< 0.60$ (with $>1$ stream) | **Cross-modal swap attack** (e.g. authentic video with AI-dubbed voice) |
| 🔵 **MODERATE / SUSPICIOUS** | $40\% - 59.9\%$ | $\ge 0.60$ | Inconclusive or subtle artifacts; recommended for manual review |
| 🟢 **LOW / AUTHENTIC** | $< 40\%$ | Any | Consistent authentic biometric & semantic characteristics |

### Explainable AI (XAI) Suite (`ml/xai/engine.py`)

CheckReality AI strictly avoids opaque labels by providing **mathematically derived explainability**:

```
Visual Modality  ──►  Grad-CAM Backprop on features[-1]  ──►  JET Heatmap Overlay + Bounding Regions
Acoustic Stream  ──►  Spectrogram Gradient Energy Profile ──►  Phase & Vocoder Anomaly Windows (sec)
Text Semantic    ──►  RoBERTa Word Embedding Saliency     ──►  Top Influential Tokens & Attributions
```

- **Visual Grad-CAM**: Computes $\alpha_k = \frac{1}{Z}\sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}$ over the final convolutional layer, projecting spatial heatmaps onto facial boundaries to highlight hairline seams, ocular warping, and blending borders.
- **Acoustic Saliency**: Computes gradient norm $\sum_f |\nabla_{\text{spec}}|$, detecting high-frequency energy spikes typical of synthetic vocoders.
- **Token Attributions**: Computes embedding saliency $\| \nabla_e y \odot e \|_2$, ranking the exact keywords influencing misinformation or AI-generation classification.

---

## 5. Running the Application

CheckReality AI can be executed through **Docker Compose** (recommended for production and evaluation) or **natively on the host machine**.

### Service Isolation & Port Layout

To guarantee complete co-existence with existing software, CheckReality AI operates on dedicated isolated ports:

| Service | Port | Description |
|---|---|---|
| **Frontend Application** | `http://localhost:5173` | Vite development server (or served via Backend on `:8001`) |
| **Backend REST API** | `http://localhost:8001` | FastAPI engine & interactive Swagger docs (`/docs`) |
| **MongoDB Instance** | `localhost:27019` | Dedicated forensic history and report store |
| **Redis Broker** | `localhost:6381` | Asynchronous Celery task message queue |

---

### Option A: Run via Docker Compose (Recommended)

**Prerequisites:** Docker Engine 24+ and Docker Compose v2+.

```bash
# 1. Clone repository and navigate to root
cd CheckRealityAI

# 2. Configure environment variables
cp .env.example .env

# 3. Launch the containerized stack
docker compose up --build -d
```

The application is immediately available:
- **Interactive UI**: `http://localhost:5173` (or `http://localhost:8001`)
- **API Documentation**: `http://localhost:8001/docs`

---

### Option B: Run Natively on Windows (Single Script)

The repository provides an automated PowerShell orchestrator ([`start_local.ps1`](start_local.ps1)) that builds the frontend bundle, verifies Python dependencies, boots the Celery worker, and launches the FastAPI gateway on a **single port (`8001`)**:

```powershell
# Run from repository root in PowerShell
.\start_local.ps1
```

Access the unified platform at: **`http://localhost:8001`**.

---

### Option C: Manual Step-by-Step Local Run

```bash
# 1. Start isolated database services
docker compose up -d checkreality-mongodb checkreality-redis

# 2. Build the React frontend
cd frontend
npm install
npm run build
cd ..

# 3. Set up Python backend environment
cd backend
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt

# 4. Start the background Celery worker
celery -A worker.celery_app worker --loglevel=info --pool=solo

# 5. In a second terminal, launch the FastAPI server
uvicorn app.main:app --port 8001 --reload
```

---

## 6. API Specification & Ingestion Workflows

The REST API exposes asynchronous and synchronous endpoints under `/submissions`, `/reports`, and `/health`. Complete interactive documentation is accessible at **`http://localhost:8001/docs`**.

### Ingestion Examples

#### 1. Analyze an Image for Deepfakes
```bash
curl -X POST "http://localhost:8001/submissions/" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "modality=image" \
  -F "file=@sample_face.jpg"
```

#### 2. Analyze a Video (Spatial-Temporal Analysis)
```bash
curl -X POST "http://localhost:8001/submissions/" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "modality=video" \
  -F "file=@interview_clip.mp4"
```

#### 3. Analyze Audio for Voice Cloning
```bash
curl -X POST "http://localhost:8001/submissions/" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "modality=audio" \
  -F "file=@speech_sample.flac"
```

#### 4. Analyze Text for Misinformation & AI Generation
```bash
curl -X POST "http://localhost:8001/submissions/" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "modality=text" \
  -F "text_content=Breaking news: Secret government memos reveal artificial intelligence replaced the electoral college."
```

#### 5. Multimodal Simultaneous Verification
```bash
curl -X POST "http://localhost:8001/submissions/" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "modality=multimodal" \
  -F "file=@broadcast_segment.mp4" \
  -F "text_content=Transcript of the televised briefing regarding emergency protocol."
```

#### 6. Retrieve Forensic Authenticity Report
```bash
curl -X GET "http://localhost:8001/submissions/{submission_id}" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

---

## 7. Forensic Authenticity Report & Interactive UI

The React 18 user interface is built around high-density forensic analysis:

1. **Forensic Verdict Card**: Displays unified status (`REAL` or `FAKE`), confidence percentage, dynamic Risk Tier (`LOW`, `MODERATE`, `HIGH`, `CRITICAL`), and Cross-Modal Agreement score.
2. **Forensic Synthesis Engine**: Plain-language synthesis explaining why the model reached its decision.
3. **Multi-Modal Breakdown Grid**: Displays per-modality confidence bars for Video, Audio, Image, and Text.
4. **Visual Grad-CAM Viewer**: Interactive side-by-side display of original media and the colored thermal activation heatmap with detected anomaly coordinates.
5. **Acoustic Timeline Inspector**: Displays localized millisecond distortion windows where phase anomalies or vocoder glitches occurred.
6. **Token Saliency Highlighting**: Color-codes influential words that triggered misinformation or synthetic text classifiers.
7. **One-Click PDF Export**: Uses client-side print stylesheets (`window.print()`) to generate clean, printable forensic audit reports.

---

## 8. Security, Evidence Integrity & Privacy Model

- **Media Hashing**: All submitted media files are indexed with a `SHA-256` cryptographic digest upon receipt to maintain a verified chain of custody.
- **Zero Data Retention**: Uploaded media can be scrubbed post-inference; only extracted mathematical features and forensic reports are stored.
- **ReDoS and File Validation**: Strict validation checks file MIME types, enforces a 200 MB upload ceiling, and limits video decoding to 60 seconds to prevent denial-of-service exhaustion.
- **Isolated Network Topology**: Docker bridge networking isolates backend workers and database engines from public interfaces.

---

## 9. Verification, Testing & Integration Suite

The repository includes an end-to-end integration test suite ([`backend/test_integration.py`](backend/test_integration.py)) that validates model weight loading, live multi-modal inference, XAI generation, fusion calculations, and metric integrity.

```bash
# Run integration test suite
python backend/test_integration.py
```

### Verified Test Run Output

```text
============================================================
CHECKREALITY AI - FULL SYSTEM INTEGRATION TEST
============================================================

[1/5] Testing Model Loading & Checkpoint Integrity...
  [PASS] Image Model (EfficientNet-B3): LOADED
  [PASS] Audio Model (ResNet-18 SpecAugment): LOADED
  [PASS] Text Misinformation Model (RoBERTa-base): LOADED
  [PASS] Text AI Detection Model (RoBERTa-base): LOADED
  [PASS] Video Model (EfficientNet-B0 + 2L LSTM): LOADED

[2/5] Testing Inference Pipelines across All Modalities...
  [PASS] Text Inference: {'misinformation_probability': 0.0011, 'misinformation_label': 'Real', 'ai_generated_probability': 0.0004, 'ai_generated_label': 'Human Written'}
  [PASS] Image Inference: {'fake_probability': 0.9994, 'label': 'Fake'}
  [PASS] Audio Inference: {'spoof_probability': 0.9999, 'label': 'Spoof/Fake'}
  [PASS] Video Inference: {'fake_probability': 0.9775, 'label': 'Fake'}

[3/5] Testing Multimodal Fusion Engine...
  [PASS] Unified Probability: 79.0%
  [PASS] Unified Verdict: Fake
  [PASS] Risk Severity: HIGH
  [PASS] Agreement Index: 0.189
  [PASS] Forensic Synthesis: Forensic Fusion Engine analyzed 3 modal stream(s): VIDEO, AUDIO, TEXT. Cross-modal discrepancy detected: VIDEO, AUDIO showed synthetic indicators, whereas TEXT appeared authentic. This pattern is typical of targeted speech/facial replacement deepfakes. Overall forensic verdict: FAKE with unified confidence of 79.0% (HIGH Risk).

[4/5] Testing Explainable AI (XAI) Output...
  [PASS] Text Attribution XAI: Gradient Token Attribution, Tokens: 6
  [PASS] Image Grad-CAM XAI: Grad-CAM Saliency, Heatmap Length: 32584 bytes
  [PASS] Audio Saliency XAI: Acoustic Saliency Analysis, Anomaly Windows: 4
  [PASS] Video Grad-CAM XAI: Temporal Grad-CAM, Heatmap Length: 369700 bytes

[5/5] Validating Organic Metrics across All 5 Models...
  [PASS] pretrained_resnet18_audio: Acc=99.60%, F1=0.9960, AUC=0.9999 (Valid)
  [PASS] efficientnet_b3_deepfake: Acc=99.40%, F1=0.9940, AUC=0.9998 (Valid)
  [PASS] roberta_base_misinformation: Acc=99.95%, F1=0.9995, AUC=1.0000 (Valid)
  [PASS] roberta_base_ai_text: Acc=100.00%, F1=1.0000, AUC=1.0000 (Valid)
  [PASS] efficientnet_lstm_video: Acc=99.25%, F1=0.9925, AUC=0.9994 (Valid)
  [PASS] All 5 models verified in the 99+% benchmark tier!

============================================================
INTEGRATION TEST SUMMARY: 5/5 SUITES PASSED
============================================================
```

- **Frontend Compilation**: Production build passes cleanly with zero TypeScript errors:
  ```text
  ✓ 93 modules transformed.
  dist/index.html                   0.99 kB
  dist/assets/index-Ba_EVSJM.css   52.01 kB
  dist/assets/index-soOcbCVP.js   427.54 kB
  ✓ built in 8.63s
  ```

---

## 10. Current Status, Limitations & Out of Scope

### Current Status
CheckReality AI is a fully functional prototype with live inference, explainable visualizations, multi-modal Bayesian fusion, and a production dashboard.

### Operational Limitations
- **Compression Artifacts**: Models are fine-tuned on FaceForensics++ C23 and standard social media compressions. Heavily re-encoded media (e.g. video sent through multiple WhatsApp/TikTok transcoders below 240p) may exhibit reduced discriminative power.
- **Language Scope**: Text misinformation and AI essay detection models currently evaluate **English-language** texts only.
- **Short Audio Samples**: Audio spoof detection requires a minimum of 0.5 seconds of voiced speech to build an informative Mel-spectrogram.
- **Extremely Low-Resolution Faces**: Facial deepfake detection degrades on faces smaller than $40 \times 40$ pixels.

### Explicitly Out of Scope
- **Court-Admissible Legal Certification**: The framework provides decision-support metrics, not legal guarantees.
- **Social Graph Propagation**: The platform analyzes content payload authenticity, not network botnet propagation dynamics.
- **Zero-Day Synthesis Architectures**: Generative models employing fundamentally novel mathematical distributions not represented in training datasets may produce uncertain scores.

---

## 11. Future Scope and Roadmap

| # | Enhancement | Priority | Planned Implementation |
|---|---|---|---|
| **F1** | **Audio-Visual Lip-Sync Discrepancy Engine** | High | Integrate Wav2Lip / SyncNet cross-attention to identify micro-second desynchronization between lip movements and phonetic formants |
| **F2** | **Live RTSP / WebRTC Stream Interception** | High | Extend Celery workers to ingest live broadcast streams and output sliding-window manipulation scores in real time |
| **F3** | **Multilingual Text Adaptation** | Medium | Upgrade text classifiers to `XLM-RoBERTa` to support Hindi, Punjabi, and other regional Indian languages |
| **F4** | **Hardware Acceleration via TensorRT & ONNX** | Medium | Export PyTorch models to ONNX and quantized INT8 TensorRT engines for sub-10ms edge inference |
| **F5** | **Blockchain Evidence Ledger** | Low | Anchor forensic report SHA-256 digests onto an immutable ledger (e.g. Ethereum / Hyperledger) for tamper-proof verification |
| **F6** | **Adversarial Robustness Hardening** | High | Incorporate adversarial training against targeted gradient evasion attacks (FGSM / PGD) |

---

## 12. Documentation Directory

| Document | Location | Purpose |
|---|---|---|
| **Product Requirements Document (PRD)** | [`docs/PRD.md`](docs/PRD.md) | Vision, user personas, requirements, and project scope |
| **Software Requirements Specification (SRS)** | [`docs/SRS.md`](docs/SRS.md) | IEEE 830-style functional and non-functional specifications |
| **Technical Requirements Document (TRD)** | [`docs/TRD.md`](docs/TRD.md) | Microservice architecture, database schemas, and API contracts |
| **Implementation Plan** | [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) | Architectural roadmap and development milestones |
| **Training Pipeline Documentation** | [`training/scripts/README.md`](training/scripts/README.md) | Detailed documentation on training scripts, datasets, and hyperparameters |

---

## 13. Academic Attribution & Project Author

**Department of Computer Science & Engineering**  
**Baba Banda Singh Bahadur Engineering College, Fatehgarh Sahib (IKGPTU)**  
*B.Tech Major Project — Batch 2023–2026*

| Developer & Project Author | University Roll No. | Degree & Specialization | Core Responsibilities |
| :--- | :---: | :--- | :--- |
| **Divyansh Rajat** | 2300358 | B.Tech Computer Science & Engineering (2026) | Complete End-to-End System Development: Deep Learning Models Fine-Tuning, Multimodal Fusion Engine, Explainable AI (XAI) Suite, FastAPI Async Backend Pipeline, Celery Task Distribution, and React Forensic UI |

**Project Guide & Supervisor:** **Dr. Manpreet Kaur**, Associate Professor, Dept. of Computer Science & Engineering

---

## 14. License

This project was developed as an academic capstone major project. For academic review, demonstration, and evaluation purposes only.

---

<div align="center">
<sub>CheckReality AI · Different Modalities. One Standard. Trusted Everywhere.</sub>
</div>
