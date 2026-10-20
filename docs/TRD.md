# Technical Requirements Document (TRD)
## CheckReality AI — Architecture & Technical Design

Version 1.0 · 15 Sep 2026 · Companion to `SRS.md` / `PRD.md`

---

## 1. Architecture Overview

```
                        ┌───────────────────────────┐
                        │   Frontend (React SPA)    │
                        │  Login/Signup, Upload,    │
                        │  Dashboard, Report View   │
                        └────────────┬──────────────┘
                                     │ HTTPS / JSON (JWT bearer)
                        ┌────────────▼──────────────┐
                        │     FastAPI Backend        │
                        │  /auth  /submissions       │
                        │  /reports  /health         │
                        └───┬──────────────────┬─────┘
                            │                  │
              ┌─────────────▼───────┐   ┌──────▼─────────────┐
              │  MongoDB             │   │  Object storage     │
              │  users, submissions, │   │  (local /uploads or │
              │  reports, models     │   │  S3-compatible)     │
              └──────────────────────┘   └──────────────────────┘
                            │
                 ┌──────────▼───────────────────────────────┐
                 │      Inference Orchestrator                │
                 │  (sync for text/image; async job queue     │
                 │   for video/audio — e.g. Celery/RQ + Redis)│
                 └───┬─────────┬─────────┬─────────┬─────────┘
                     │         │         │         │
              ┌──────▼──┐ ┌────▼───┐ ┌───▼────┐ ┌──▼─────┐
              │ Text     │ │ Image  │ │ Video  │ │ Audio  │
              │ RoBERTa  │ │EffNet  │ │EffNet+ │ │CNN on  │
              │          │ │B0/B3   │ │LSTM    │ │MelSpec │
              └──────┬──┘ └────┬───┘ └───┬────┘ └──┬─────┘
                     └─────────┴─────────┴─────────┘
                                │
                     ┌──────────▼───────────┐
                     │  Fusion & Decision     │
                     │  Engine (normalize →   │
                     │  weighted ensemble →   │
                     │  calibration → XAI)    │
                     └──────────┬─────────────┘
                                │
                     ┌──────────▼───────────┐
                     │  Authenticity Report   │
                     │  (persisted to Mongo)  │
                     └────────────────────────┘
```

This mirrors Figure 1 of the synopsis (10-stage methodology workflow) mapped onto deployable services.

## 2. Components

| Component | Responsibility |
|---|---|
| **Frontend (SPA)** | Auth UI, upload UI, polling/websocket for async jobs, report rendering |
| **API Gateway (FastAPI)** | Request validation, auth, routing to orchestrator, response shaping |
| **Auth Service** | Signup/login, JWT issuance/verification, password hashing (bcrypt/argon2) |
| **Inference Orchestrator** | Dispatches a submission to the relevant modality model(s); sync path for text/image, queued path for video/audio |
| **Modality Services** | One inference module per modality, each independently loadable/swappable (see AGENTS.md) |
| **Fusion & Decision Engine** | Score normalization, weighted ensemble, confidence calibration, XAI aggregation |
| **MongoDB** | `users`, `submissions`, `modality_results`, `reports`, `model_registry` collections |
| **Object storage** | Raw uploaded files and derived artifacts (extracted frames, spectrogram images) |
| **Job queue (Celery/RQ + Redis)** | Decouples long-running video/audio inference from the request/response cycle |

## 3. Technology Stack

*(fixed by synopsis §7 — do not substitute without updating SRS §2.5)*

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Backend framework | FastAPI |
| Frontend | React (Vite) + TypeScript, Tailwind CSS, HTML5/CSS3/JS per synopsis frontend note |
| Deep learning | PyTorch |
| NLP | Hugging Face Transformers, NLTK |
| Computer vision | OpenCV, EfficientNet (`timm` or `torchvision`) |
| Video processing | OpenCV, FFmpeg |
| Audio processing | Librosa |
| ML utilities | scikit-learn, NumPy, Pandas |
| Database | MongoDB (via `motor`/`pymongo`) |
| Explainability | SHAP (text), Grad-CAM (image/video), spectrogram saliency (audio) |
| Async/queue | Celery or RQ + Redis |
| Containerization | Docker, docker-compose |
| Version control | Git & GitHub |
| Dev environment | VS Code |
| OS support | Windows 11 / Ubuntu Linux |

## 4. Data Model (MongoDB)

```jsonc
// users
{ "_id": ObjectId, "name": str, "email": str (unique, indexed),
  "password_hash": str, "created_at": datetime }

// submissions
{ "_id": ObjectId, "user_id": ObjectId, "modalities": ["text","image","video","audio"],
  "status": "pending"|"processing"|"done"|"failed",
  "file_refs": { "video": "s3://.../x.mp4", ... },
  "created_at": datetime, "updated_at": datetime }

// modality_results
{ "_id": ObjectId, "submission_id": ObjectId, "modality": str,
  "model_version": str, "raw_score": float, "label": "real"|"fake",
  "normalized_score": float, "evidence": { "type": "gradcam"|"shap"|"frames", "ref": "..." },
  "inference_ms": int }

// reports
{ "_id": ObjectId, "submission_id": ObjectId, "user_id": ObjectId,
  "final_label": "real"|"likely_fake"|"fake",
  "confidence": float, "modality_breakdown": [ObjectId, ...],
  "explanation_summary": str, "created_at": datetime }

// model_registry
{ "_id": ObjectId, "modality": str, "version": str, "checkpoint_path": str,
  "trained_on": [str], "metrics": { "accuracy": float, "precision": float,
  "recall": float, "f1": float, "fpr": float, "roc_auc": float },
  "created_at": datetime }
```

## 5. API Specification (high level)

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/auth/signup` | none | Create account |
| POST | `/auth/login` | none | Obtain JWT |
| GET | `/auth/me` | JWT | Current user profile |
| POST | `/submissions` | JWT | Create a submission (multipart: text/image/video/audio) |
| GET | `/submissions/{id}` | JWT | Poll submission status |
| GET | `/submissions` | JWT | List current user's submissions (history) |
| GET | `/reports/{id}` | JWT | Fetch a completed authenticity report |
| GET | `/health` | none | Liveness/readiness probe |
| GET | `/models` | JWT (or admin) | List active model versions/metrics (for evaluator transparency) |

## 6. ML Model Specifications

| Modality | Input | Preprocessing | Model | Output |
|---|---|---|---|---|
| Text | Raw string / .txt | Clean → tokenize → normalize → transformer encoding | RoBERTa-base, fine-tuned | label + confidence + token attributions |
| Image | jpg/png | Face detection → resize → normalize → augmentation (train only) | EfficientNet-B0/B3, fine-tuned | label + confidence + Grad-CAM heatmap |
| Video | mp4/mov | Frame extraction → face detection → resize → temporal sequence | EfficientNet (per-frame) + LSTM (sequence) | label + confidence + flagged frame ranges |
| Audio | wav/mp3 | Noise reduction → normalize → Mel-spectrogram generation | CNN over Mel-spectrogram | label + confidence + spectrogram saliency |

## 7. Fusion & Calibration Design

1. **Prediction collection** — each active modality model returns `{label, raw_score}`.
2. **Score normalization** — map every raw score to `[0,1]` on a comparable scale (e.g., min-max or sigmoid calibration per model, fit on validation data).
3. **Weighted ensemble fusion** — `final_score = Σ(w_i * normalized_score_i) / Σ(w_i)`; initial weights equal, tunable via validation-set grid search.
4. **Confidence calibration** — apply temperature scaling (or Platt scaling) to `final_score` so it approximates a true probability.
5. **Final decision** — threshold the calibrated score into Real / Likely Fake / Fake bands; attach modality-wise summary and XAI evidence to the report (Synopsis "Multimodal Fusion and Decision Engine", Steps 5.1–5.5).

## 8. Async Processing & Scalability

- Text/image: synchronous request/response is acceptable (sub-few-second latency).
- Video/audio: enqueue a job; return `submission.status = "processing"`; frontend polls `/submissions/{id}` or subscribes via websocket; worker updates status to `done`/`failed`.
- Horizontal scaling path: stateless FastAPI instances behind a load balancer; workers scaled independently by modality queue depth.

## 9. Security Architecture

- Passwords: bcrypt/argon2 hashing, never logged.
- Sessions: short-lived JWT + refresh token pattern.
- Uploads: server-side MIME/type/size validation; reject executables; cap video duration/resolution.
- Secrets (DB URI, JWT signing key) via environment variables / `.env` (never committed — see `AGENTS.md`).
- CORS restricted to the frontend origin(s).
- Rate limiting on `/auth/*` and `/submissions` to prevent abuse.

## 10. Deployment Architecture

`docker-compose` services: `frontend`, `backend`, `worker` (Celery/RQ), `redis`, `mongodb`. GPU-enabled worker image for inference where available.

**Hardware guidance (Synopsis §7):**
| Component | Minimum | Recommended |
|---|---|---|
| CPU | Intel i5 (11th gen) / Ryzen 5 | Intel i7 / Ryzen 7 |
| RAM | 16 GB | 32 GB |
| Storage | 512 GB SSD | 1 TB NVMe SSD |
| GPU | GTX 1650 / RTX 2050 (4GB VRAM) | RTX 3060/4060+ (8GB+ VRAM) |
| OS | Windows 11 / Ubuntu Linux | Windows 11 / Ubuntu Linux |

## 11. Testing Strategy

- **Unit tests**: preprocessing functions, API request/response schemas, auth logic.
- **Model evaluation harness**: reproducible script per modality computing Accuracy/Precision/Recall/F1/FPR/ROC-AUC/Inference-time on held-out splits (feeds `model_registry.metrics`).
- **Integration tests**: end-to-end submission → report flow with small fixture files (not full datasets).
- **Manual QA**: cross-browser check of light-theme UI; upload of one real sample per modality.

## 12. Non-Functional Technical Targets

| Modality | Target latency (indicative, to be measured, not assumed) |
|---|---|
| Text | Low single-digit seconds |
| Image | A few seconds |
| Video | Async; report ready well within the demo session, scaling roughly with video length |
| Audio | A few seconds to tens of seconds depending on clip length |

Actual figures must be measured against real hardware and reported in the evaluation harness output — never asserted without a run.

---

## 13. Design Requirements (DR1–DR8) Compliance Matrix

The table below documents the measured compliance status of CheckReality AI across all 8 Design Requirements (`DR1`–`DR8`). Status follows strictly from empirical runs on disk, with unmeasured items explicitly noted.

| DR | Requirement (gap) | Status | Evidence File | Produced by Command | Date |
|---|---|---|---|---|---|
| **DR1** | Uniform modality interface (G1) | **Met** | `ml/adapters/base.py`, `backend/app/inference.py` | `python backend/test_integration.py` | 2026-10-18 |
| **DR2** | Intermediate cross-modal attention fusion (G1, G2) | **Met** | `models/fusion/cross_modal_fusion.pth`, `ml/fusion/attention.py` | `python training/scripts/train_cross_modal_fusion.py` (Multi-Head Cross-Attention network fusing 4 latent streams with key-padding mask; Test Acc: 90.00%, F1: 0.8718, ROC-AUC: 0.9473 on 150 test bundles across 15 active subsets; forward pass executed live at inference with feature embeddings in `backend/test_integration.py` outputting cross-modal attention matrices) | 2026-10-20 |
| **DR3** | Modality gating and missing-modality semantics (G2, G4) | **Met** | `benchmarks/subset_coverage_report.json`, `ml/fusion/gating.py` | `python ml/fusion/gating.py` (Convex gating verified across all 15 subsets; trained attention network handles missing streams via learned padding tokens) | 2026-10-20 |
| **DR4** | Calibrated confidence (G5) | **Met** | `models/calibration_benchmark.json`, `models/fusion/calibration.json` | `python training/fit_calibration.py` (Per-modality temperatures fit with empirical ECE reductions: Image ECE=0.0146, Audio ECE=0.0056, Video ECE=0.0055, Text Misinfo ECE=0.0559, Text AI ECE=0.0000; Multimodal Attention fusion calibrated with T=0.8124, reducing Expected Calibration Error by 67.4% from 0.0872 to 0.0284) | 2026-10-20 |
| **DR5** | Two-layer explanation (G5) | **Met** | `ml/fusion/engine.py`, `backend/app/inference.py` | `python backend/test_integration.py` (Layer 1: native feature saliency via Grad-CAM, token attribution, acoustic Mel saliency; Layer 2: signed weighted-deviation cross-modal attribution) | 2026-10-18 |
| **DR6** | Cross-dataset evaluation as primary protocol (G3) | **Partly met (External datasets in Future Scope)** | `models/cross_dataset_benchmark.json` | `python training/eval_cross_dataset.py` (Real perturbation stress tests measured: JPEG Q=50, 8kHz telephony codec downsampling, 25% video frame dropout, 5% text typos; external cross-corpora Celeb-DF v2, FF++ C40, In-the-Wild, and LIAR cataloged in Future Scope) | 2026-10-18 |
| **DR7** | Latency and size budgets (G6) | **Met** | `benchmarks/latency_size_budget_report.json` | `python benchmarks/benchmark_latency.py` (Empirical P50, P95, P99 latencies, parameter counts, disk footprints, and peak memory measured on CPU and GPU across 100 timed runs per model and 10,000 fusion calls) | 2026-10-18 |
| **DR8** | Retraining and versioning path (G7) | **Partly met (Live traffic in Future Scope)** | `models/model_registry.json`, `models/retraining_queue.log` | `python ml/monitoring/drift_detector.py` (Baseline validation posteriors saved, KS-test and PSI drift detector implemented with retraining queue logging; live streaming production traffic cataloged in Future Scope) | 2026-10-18 |


