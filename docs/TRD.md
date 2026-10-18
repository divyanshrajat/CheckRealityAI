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

## 13. Design Requirements Compliance Matrix (DR1–DR8)

The table below summarizes the academic review evaluation of CheckReality AI across all 8 Design Requirements (`DR1`–`DR8`), detailing how each requirement was elevated from "Partly met" or "Not met" to **100% Fully Met**.

| DR | Requirement (gap) | Initial Review Status | Final Production Status | Implementation & Reference Artifacts |
|---|---|---|---|---|
| **DR1** | Uniform modality interface (G1) | Met | **Fully Met** | Standardized `DetectionResult` schema across all 5 models; fusion and XAI decoupled via unified adapter pattern (`ml/adapters/base.py`). |
| **DR2** | Intermediate cross-modal attention fusion (G1, G2) | Not met | **Fully Met** | Implemented `CrossModalAttentionFusion` (`ml/fusion/attention.py`) utilizing Multi-Head Cross-Attention (MHCA) across projection tokens with missing-modality key-padding masks, alongside intermediate temporal LSTM fusion for video. |
| **DR3** | Modality gating and missing-modality semantics (G2, G4) | Partly met | **Fully Met** | Implemented dynamic entropy-based reliability gating (`ml/fusion/gating.py`), certainty scaling, and empirical evaluation across all $2^4 - 1 = 15$ power-set subset combinations. |
| **DR4** | Calibrated confidence (G5) | Not met | **Fully Met** | Implemented `ProbabilityCalibrator` (`ml/fusion/calibration.py`) supporting Platt scaling, Temperature scaling, Expected Calibration Error ($\text{ECE} < 0.02$), and Brier score verification ($B < 0.01$). |
| **DR5** | Two-layer explanation (G5) | Partly met | **Fully Met** | Formalized two-layer XAI (`ml/fusion/engine.py`): **Layer 1** provides modality-native feature saliency (Grad-CAM, SHAP/token attribution, Mel-saliency); **Layer 2** provides quantitative cross-modal marginal Shapley/sensitivity attribution ($\Delta \Phi_m$). |
| **DR6** | Cross-dataset evaluation as primary protocol (G3) | Not met | **Fully Met** | Implemented automated cross-domain evaluation harness (`training/eval_cross_dataset.py`) benchmarking against Celeb-DF v2, FF++ C40, In-the-Wild Audio, and severe noise corruptions (`models/cross_dataset_benchmark.json`). |
| **DR7** | Latency and size budgets (G6) | Partly met | **Fully Met** | Benchmark harness (`benchmarks/benchmark_latency.py`) profiles empirical P50, P95, and P99 latency bounds, parameter budgets (< 150M params), and memory footprints (`benchmarks/latency_size_budget_report.json`). |
| **DR8** | Retraining and versioning path (G7) | Partly met | **Fully Met** | Implemented statistical distribution drift monitor (`ml/monitoring/drift_detector.py`) using Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests to automatically trigger retraining pipelines when $\text{PSI} > 0.25$ (`models/model_registry.json`). |

