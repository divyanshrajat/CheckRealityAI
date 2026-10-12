# TASK.md
## CheckReality AI — Build Backlog

Checklist form, organized by phase. See `IMPLEMENTATION.md` for the detailed runbook behind each phase, and `AGENTS.md` §7 for the definition of done.

## Phase 0 — Repo & Environment Setup
- [ ] Initialize monorepo structure per `AGENTS.md` §2
- [ ] `.gitignore` for `data/`, `models/checkpoints/`, `.env`, `node_modules/`, `__pycache__/`
- [ ] `.env.example` with all required variable names (Mongo URI, JWT secret, Redis URL, etc.)
- [ ] `docker-compose.yml` skeleton (frontend, backend, worker, redis, mongodb)
- [ ] `docs/` populated with PRD, SRS, TRD, AGENTS, TASK, IMPLEMENTATION
- [ ] CI stub (lint + test on push) — optional for an academic project, nice-to-have

## Phase 1 — Data Pipeline (per modality)
- [ ] Text: acquire/point to Kaggle Fake & Real News + AI vs Human Text datasets; write loader + split
- [ ] Image: acquire/point to 140k Real & Fake Faces; write loader + face-crop preprocessing
- [ ] Video: acquire/point to FaceForensics++; write frame-extraction + face-detection pipeline
- [ ] Audio: acquire/point to ASVspoof 2019; write Mel-spectrogram generation pipeline
- [ ] Create a small `data/subset/` sample of each dataset for fast dev iteration
- [ ] Document dataset licenses/usage terms in `ml/<modality>/README.md`

## Phase 2 — Model Fine-Tuning
- [ ] Text: fine-tune RoBERTa-base on Fake News + AI vs Human Text
- [ ] Image: fine-tune EfficientNet-B0/B3 on 140k Real/Fake Faces
- [ ] Video: fine-tune EfficientNet (frame) + LSTM (sequence) on FaceForensics++
- [ ] Audio: fine-tune CNN on Mel-spectrograms from ASVspoof 2019
- [ ] Evaluation harness per modality: Accuracy, Precision, Recall, F1, FPR, ROC-AUC, Inference Time
- [ ] Register each trained checkpoint + metrics in `model_registry`
- [ ] Smoke-test every model on the `data/subset/` sample before a full training run

## Phase 3 — Fusion, Calibration, Explainability
- [ ] Score normalization across modalities
- [ ] Weighted ensemble fusion implementation
- [ ] Confidence calibration (temperature/Platt scaling) fit on validation data
- [ ] SHAP integration for text token attributions
- [ ] Grad-CAM integration for image/video
- [ ] Spectrogram saliency for audio
- [ ] Fused-system evaluation on multimodal test cases

## Phase 4 — Backend API
- [ ] FastAPI project scaffold + settings/config
- [ ] MongoDB connection + collections (`users`, `submissions`, `modality_results`, `reports`, `model_registry`)
- [ ] Auth: signup, login, JWT issuance/verification, password hashing
- [ ] `POST /submissions` — accept multipart upload, validate, persist, dispatch inference
- [ ] Sync inference path (text/image)
- [ ] Async inference path (video/audio) via Celery/RQ + Redis worker
- [ ] `GET /submissions/{id}` status polling
- [ ] `GET /reports/{id}` — assemble fused report from `modality_results`
- [ ] `GET /submissions` — user history
- [ ] `GET /health`
- [ ] Backend unit + integration tests

## Phase 5 — Frontend
- [ ] Design tokens: light theme, white background, logo, background graphics (see `ANTIGRAVITY_PROMPT.md`)
- [ ] Landing page
- [ ] Signup page + validation
- [ ] Login page + validation
- [ ] Auth state management (JWT storage, protected routes)
- [ ] Upload page (unified text/image/video/audio surface)
- [ ] Processing/loading state for async jobs
- [ ] Report view (verdict, confidence, per-modality breakdown, explanation, disclaimer)
- [ ] History/dashboard page
- [ ] Responsive check (desktop + mobile widths)

## Phase 6 — Integration & Evaluation
- [ ] End-to-end test: sign up → upload each modality → receive report
- [ ] End-to-end test: multimodal submission (video with audio) → fused report
- [ ] Full-dataset evaluation run recorded for the viva/report (Stage 1 + Stage 2 per `TRD.md` §11)
- [ ] Cross-dataset generalization check (Synopsis §7.4)
- [ ] Fix graceful-degradation path: one modality's model fails → others still return

## Phase 7 — Deployment & Docs
- [ ] `docker-compose up` runs the full stack from a clean checkout
- [ ] README with setup instructions
- [ ] Final evaluation report (numbers from Phase 6, not estimates)
- [ ] Viva/demo script and slide alignment with `PRD.md` release plan
- [ ] Update `AGENTS.md`/`TRD.md` for any deviations made during the build
