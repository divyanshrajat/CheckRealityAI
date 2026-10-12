# AGENTS.md
## Conventions for any AI coding agent (Claude Code, Antigravity, or other) working in this repository

This file is the first thing an agent should read. It does not repeat requirements — see `PRD.md` / `SRS.md` / `TRD.md` for those — it only covers **how to work in this repo**.

## 1. Read order before doing anything

1. `PRD.md` — what we're building and why
2. `SRS.md` — exact functional/non-functional requirements
3. `TRD.md` — architecture, API, data model, model specs
4. `TASK.md` — current backlog / what's already done
5. `IMPLEMENTATION.md` — build sequence and folder layout
6. This file — how to behave while building it

## 2. Repository layout (target)

```
checkreality-ai/
├── docs/                     # PRD, SRS, TRD, AGENTS, TASK, IMPLEMENTATION
├── data/
│   ├── raw/                  # untouched downloaded datasets (gitignored)
│   ├── subset/               # small dev-sized subset per dataset (gitignored or LFS)
│   └── processed/            # preprocessed tensors/features (gitignored)
├── ml/
│   ├── text/                 # RoBERTa fine-tuning: train.py, eval.py, dataset.py
│   ├── image/                # EfficientNet fine-tuning
│   ├── video/                # EfficientNet+LSTM fine-tuning
│   ├── audio/                # CNN on Mel-spectrogram fine-tuning
│   ├── fusion/                # weighted ensemble + calibration
│   ├── xai/                  # SHAP / Grad-CAM / saliency helpers
│   └── common/                # shared config, metrics, checkpoint utils
├── models/
│   └── checkpoints/<modality>/<version>/   # trained weights (gitignored, tracked via model_registry doc/DB instead)
├── backend/
│   ├── app/                  # FastAPI app: routers, services, db, auth
│   └── tests/
├── frontend/
│   ├── src/                  # React app: pages, components, api client
│   └── public/assets/        # logo, background graphics
├── docker-compose.yml
└── .env.example
```

## 3. Non-negotiable guardrails

- **Never commit datasets, raw media uploads, or model checkpoints.** Use `.gitignore`; reference them by path/registry entry instead.
- **Never commit secrets** (DB URI, JWT signing key, API keys). Use `.env`, and keep `.env.example` up to date with variable *names* only.
- **Never fabricate evaluation numbers.** Any accuracy/precision/recall/F1/FPR/ROC-AUC quoted in code comments, docs, or the UI must come from an actual run of the evaluation harness (`TRD.md` §11) and be stored in `model_registry`. If a number hasn't been measured yet, say "not yet evaluated," not a guess.
- **Keep the decision-support disclaimer** (SRS FR-XAI-2) visible in the report UI — do not remove it while iterating on styling.
- **Keep modalities decoupled.** A failure or absence of one modality's model must not break the others (SRS "Reliability"). Each `ml/<modality>/` module should be runnable and testable independently.
- **English-only text v1** — do not silently expand scope to other languages without updating `SRS.md`.

## 4. Coding conventions

**Python (ml/, backend/):**
- Format with `black`, lint with `ruff`, type-check with `mypy` where practical.
- Type hints on public functions.
- Config via `pydantic-settings` or a single `config.py` reading from environment variables — no hardcoded paths/ports.
- One `requirements.txt` (or `pyproject.toml`) per top-level app (`ml/`, `backend/`) so training deps don't bloat the API image.

**Frontend (React/TypeScript):**
- Format with `prettier`, lint with `eslint`.
- Functional components + hooks; Tailwind for styling (light theme tokens defined once in `tailwind.config.js`, not inlined per-component).
- API calls centralized in `src/api/` — no `fetch()` scattered through components.

**Git:**
- Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`, `test:`).
- One logical change per commit; don't mix a model change with an unrelated UI tweak.

## 5. How to run locally (fill in exact commands as they're implemented)

```bash
# Backend
cd backend && cp ../.env.example .env && pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend && npm install && npm run dev

# Full stack
docker-compose up --build

# Model training (example — see CLAUDE_CODE_PROMPT.md)
cd ml/text && python train.py --config configs/roberta_fake_news.yaml --subset   # fast dev run
cd ml/text && python train.py --config configs/roberta_fake_news.yaml           # full run
```

## 6. Adding a new modality or swapping a model

1. Add `ml/<modality>/{dataset.py,train.py,eval.py}` following the pattern of an existing modality.
2. Register the new checkpoint in `model_registry` with its measured metrics.
3. Add a corresponding inference adapter in `backend/app/services/inference/`.
4. Update `TRD.md` §6 (model specs) and `SRS.md` §3 (functional requirements) — docs and code stay in sync.
5. Add/extend the fusion engine's weighting to include the new modality (`ml/fusion/`).

## 7. Definition of done (for any TASK.md item)

- Code merged with passing tests.
- If it touches a model: evaluation harness re-run, metrics recorded in `model_registry`, no fabricated numbers.
- If it touches an API contract: `TRD.md` §5 updated to match.
- If it touches UI copy about the verdict: decision-support disclaimer still present.
- `TASK.md` checkbox ticked in the same PR/commit.
