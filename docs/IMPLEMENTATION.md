# IMPLEMENTATION.md
## CheckReality AI — Build Runbook

This is the step-by-step execution order behind `TASK.md`. It assumes `AGENTS.md` conventions and the folder layout in `AGENTS.md` §2.

## Stage A — Foundations

1. `git init`, create the folder tree from `AGENTS.md` §2.
2. Add `.gitignore`:
   ```
   data/
   models/checkpoints/
   .env
   __pycache__/
   *.pt
   *.ckpt
   node_modules/
   dist/
   ```
3. Write `.env.example`:
   ```
   MONGO_URI=
   JWT_SECRET=
   JWT_EXPIRES_MIN=60
   REDIS_URL=
   MAX_UPLOAD_MB=200
   MAX_VIDEO_SECONDS=60
   ```
4. Stand up `docker-compose.yml` with empty/placeholder service defs for `frontend`, `backend`, `worker`, `redis`, `mongodb` — fill in as each is built.

## Stage B — Data Pipelines (`ml/<modality>/dataset.py`)

For each modality, in this order (cheapest/fastest first, to get the pipeline pattern right before tackling video):

1. **Text** — load Kaggle Fake & Real News + AI vs Human Text CSVs → clean → tokenize with RoBERTa tokenizer → train/val/test split (stratified).
2. **Image** — load 140k Real & Fake Faces → face-detect (if not pre-cropped) → resize to EfficientNet input size → normalize → augment (train split only).
3. **Audio** — load ASVspoof 2019 (protocol files + flac/wav) → noise reduction → Mel-spectrogram (Librosa) → cache spectrograms as tensors to avoid recomputing every epoch.
4. **Video** — load FaceForensics++ → extract frames at a fixed rate → face-detect/crop per frame → build fixed-length frame sequences → cache to disk (this is the heaviest pipeline; build it last, after the pattern is proven on text/image).

At each step, also generate `data/subset/<modality>/` — a few hundred examples — so training scripts can be smoke-tested in minutes, not hours (see `CLAUDE_CODE_PROMPT.md`'s `--subset` flag convention).

## Stage C — Model Fine-Tuning (`ml/<modality>/train.py` + `eval.py`)

Order: text → image → audio → video (video depends on the image encoder pattern, and benefits from the other three pipelines' scaffolding being proven first).

Each `train.py` should:
- Load config (dataset paths, hyperparameters, `--subset` flag) from a YAML file.
- Fine-tune the specified backbone (RoBERTa-base / EfficientNet-B0-B3 / EfficientNet+LSTM / CNN-on-melspec).
- Save checkpoints to `models/checkpoints/<modality>/<version>/`.
- Log training curves (loss/accuracy per epoch) to a simple CSV or TensorBoard log.

Each `eval.py` should:
- Load a checkpoint + the held-out test split.
- Compute Accuracy, Precision, Recall, F1, FPR, ROC-AUC, and average Inference Time.
- Write results to `model_registry` (or a local JSON the backend seeds into Mongo).

## Stage D — Fusion Engine (`ml/fusion/`)

1. Implement `normalize(raw_score, model_stats) -> float in [0,1]`.
2. Implement `weighted_ensemble(scores: dict[modality, float], weights: dict[modality, float]) -> float`.
3. Implement `calibrate(score: float, calibrator) -> float` (fit temperature/Platt scaling on a validation set with known labels).
4. Implement `decide(calibrated_score, thresholds) -> "real"|"likely_fake"|"fake"`.
5. Wire XAI: `ml/xai/text_shap.py`, `ml/xai/image_gradcam.py`, `ml/xai/video_gradcam.py`, `ml/xai/audio_saliency.py` — each returns evidence in the shape the report expects (`modality_results.evidence`).

## Stage E — Backend (`backend/app/`)

1. FastAPI scaffold: `main.py`, `routers/{auth,submissions,reports}.py`, `db.py` (Mongo client), `security.py` (JWT + password hashing), `config.py` (env settings).
2. Implement `/auth/signup`, `/auth/login`, `/auth/me`.
3. Implement `/submissions` (POST): validate upload → persist file → create `submissions` doc → for text/image call inference synchronously; for video/audio enqueue a Celery/RQ task.
4. Implement the worker (`backend/app/worker.py` or a separate `worker/` service) that consumes queued jobs, runs the relevant `ml/<modality>` inference module, writes `modality_results`, and on completion (if all modalities done) triggers the fusion engine to write a `reports` doc.
5. Implement `/submissions/{id}` (status polling) and `/reports/{id}`.
6. Implement `/submissions` (list, for history) and `/health`.
7. Write tests: auth flow, upload validation, a mocked-model end-to-end submission.

## Stage F — Frontend (`frontend/src/`)

1. Scaffold with Vite + React + TypeScript + Tailwind; set light-theme tokens (white background, accent color from the project's shield/lock motif — see `ANTIGRAVITY_PROMPT.md` for exact palette guidance).
2. Build `AuthContext` (JWT storage, login/logout, protected-route wrapper).
3. Pages: `Landing`, `Signup`, `Login`, `Upload`, `Report/:id`, `History`.
4. `src/api/client.ts` — single axios/fetch wrapper attaching the JWT bearer token.
5. Upload page: modality tabs or a single drag-and-drop zone that detects file type; text also accepts a paste box.
6. Report page: verdict banner, confidence meter, per-modality cards, explanation panel (render Grad-CAM images / highlighted text spans), decision-support disclaimer footer.
7. History page: list of past submissions linking to their reports.

## Stage G — Integration, Evaluation, Deployment

1. Run the full `docker-compose up` stack from a clean checkout; fix any missing env vars or service wiring.
2. End-to-end manual test: one real sample per modality + one multimodal (video-with-audio) sample.
3. Run the full evaluation harness on the real (not subset) test splits; record final numbers in `model_registry` and summarize in the project report/viva materials — never assert numbers that weren't produced by this run.
4. Write the top-level `README.md`: setup, run, and where to find each doc.
5. Final pass on `AGENTS.md`/`TRD.md` to correct any spec drift introduced during the build.
