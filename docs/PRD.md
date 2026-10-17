# Product Requirements Document (PRD)
## CheckReality AI — A Multimodal Deepfake & Misinformation Detection System for Digital Content Authenticity

| | |
|---|---|
| **Version** | 1.0 |
| **Date** | 15 Sep 2026 |
| **Project type** | B.Tech CSE Major Project |
| **Institution** | Baba Banda Singh Bahadur Engineering College, Fatehgarh Sahib (IKGPTU) |
| **Author / Developer** | Divyansh Rajat (Roll No. 2300358) |
| **Guide** | Dr. Manpreet Kaur |
| **Session** | July–Dec 2026 |
| **Source docs** | `Project-2_Synopsis.pdf`, `Project-2_Synopsis.pptx` |

---

## 1. Vision

Give any user — a student, journalist, moderator, or ordinary person forwarded a suspicious file — a single place to upload **text, image, video, or audio** and get back a **Real / Fake verdict, a confidence score, and a plain-language explanation** of why, instead of having to trust a single-modality checker or their own judgment.

## 2. Problem Statement

Generative AI has made fake text, deepfake images/video, and cloned voices cheap to produce and hard for ordinary users to detect. Existing tools are modality-specific (only text, or only video), rarely explain their verdicts, generalize poorly to new manipulation techniques, and are often closed-source or paywalled. There is no lightweight, explainable, single platform covering all four modalities. (Full detail: Synopsis §1, §3, §4.)

## 3. Goals

| # | Goal | Synopsis reference |
|---|---|---|
| G1 | Detect AI-generated/fake text and fake news | Objective 1 |
| G2 | Detect manipulated/synthetic facial images | Objective 2 |
| G3 | Detect deepfake video via spatial + temporal analysis | Objective 3 |
| G4 | Detect synthetic/cloned voices | Objective 4 |
| G5 | Fuse all four modality outputs into one authenticity verdict | Objective 2 / Methodology Step 4 |
| G6 | Produce a confidence score and human-readable explanation (XAI), not just a label | Objectives 5–6 |
| G7 | Ship as a scalable, deployable web platform | Objective 7 |

**Non-goals (explicitly out of scope, per Synopsis §8 Limitations):**
- Legal/forensic-grade evidence generation — the product is a **decision-support tool**, not a replacement for expert verification.
- Source attribution, content-provenance tracking, or social-media misinformation-propagation analysis.
- Multilingual text detection (v1 is English-only).
- Guaranteed detection of manipulation techniques not present in training data.

## 4. Target Users / Personas

| Persona | Need | Primary flow |
|---|---|---|
| **Student / researcher** (evaluator persona for the project itself) | Verify the system's claims, inspect per-modality scores | Upload → per-modality breakdown → report |
| **General public user** | "Is this WhatsApp/social video real?" | Upload → simple verdict + confidence |
| **Journalist / content moderator** | Needs the *why*, not just the label, before acting | Upload → authenticity report with explanation |
| **Project guide / evaluator (Dr. Manpreet Kaur / viva panel)** | Wants to see accuracy, methodology rigor, and a working demo | Dashboard, evaluation report, live demo |

## 5. Core Features (Product-Level)

1. **Account system** — signup, login (email + password, JWT session), logout.
2. **Content upload** — one unified upload surface accepting text (paste or file), image, video, or audio.
3. **Modality-specific analysis** — dedicated model per modality returns a label + confidence.
4. **Multimodal fusion** — when more than one modality is present (e.g., a video with audio), combine evidence into one verdict.
5. **Confidence-calibrated verdict** — Real / Likely Fake / Fake with a calibrated 0–100% score, not a raw model logit.
6. **Explainable authenticity report** — which modality/evidence drove the decision (e.g., "unnatural blinking detected in frames 40–52", "voice spectrogram shows synthetic artifacts").
7. **History / dashboard** — a signed-in user can see their past submissions and reports.
8. **Downloadable report** (nice-to-have, v1.1) — export the authenticity report as PDF.

## 6. Representative User Stories

- *As a user*, I can sign up and log in so my submission history is saved to my account.
- *As a user*, I can upload a video and see, within a reasonable wait, whether it's likely a deepfake, with a confidence score.
- *As a user*, I can paste a news article and find out if it looks AI-generated or matches known fake-news patterns.
- *As a user*, I can see **why** the system flagged content as fake (e.g., highlighted frames, flagged sentences), not just a bare label.
- *As a returning user*, I can view my past checks without re-uploading.
- *As the project guide*, I can see per-modality and overall accuracy/precision/recall/F1 figures from a documented evaluation run.

## 7. Success Metrics

**Model quality (Stage 1 — per-modality, Synopsis §7):**
| Modality | Metric | Target for viva/demo (initial) |
|---|---|---|
| Text | Accuracy / F1 | Reported on held-out test split, ≥ published baseline for RoBERTa fine-tunes on the same datasets |
| Image | Accuracy / F1 | Reported on 140k Faces test split |
| Video | Accuracy / F1 / ROC-AUC | Reported on FaceForensics++ test split |
| Audio | Accuracy / EER-equivalent (FPR) | Reported on ASVspoof2019 eval split |

No target numbers are invented here — actual figures must come from real evaluation runs (see TRD §11, TASK.md). This PRD intentionally does not assert unverified accuracy claims.

**Stage 2 — integrated system:** overall accuracy, precision, recall, F1, false-positive rate, and inference time end-to-end, per Synopsis §7.2–7.4.

**Product metrics:**
- Successful signup → first upload conversion.
- Median time-to-report per modality (see TRD §12 latency budgets).
- Report is judged "explains the verdict clearly" by test users (qualitative, guide/panel feedback).
- Zero crashes on the four supplied benchmark datasets during the live demo.

## 8. Release Plan

| Phase | Scope | Maps to |
|---|---|---|
| **Alpha** | Backend + one working modality (text) end-to-end, auth, basic UI | TASK.md Phase 0–4 (partial) |
| **Beta** | All four modalities individually working + fusion + confidence calibration | TASK.md Phase 2–3 |
| **RC / Pre-viva** | XAI explanations, polished UI (light theme, auth, history), full evaluation report | TASK.md Phase 4–6 |
| **Final** | Deployed demo, documentation, references, viva-ready presentation | TASK.md Phase 7 |

## 9. Risks & Mitigations

| Risk (from Synopsis §8) | Mitigation |
|---|---|
| Dataset quality/coverage limits generalization | Report limitations honestly in evaluation; keep dataset table visible in report |
| Compute cost for real-time video/audio | Cap upload length in v1; process asynchronously with a progress indicator |
| Missing/corrupted modality input | Graceful degradation — fuse only the modalities actually present |
| Black-box residual behavior despite XAI | Frame XAI output as "supporting evidence," never "full explanation," in UI copy |
| Evolving generative techniques outdate the model | Version model checkpoints; document retraining process in AGENTS.md |
| English-only text detector | State this constraint in-product (tooltip/disclaimer) |

## 10. Dependencies

- Public datasets: Kaggle Fake & Real News, AI vs Human Text, FaceForensics++, 140k Real & Fake Faces, ASVspoof 2019 (see SRS Appendix A for sizes).
- Pretrained backbones: RoBERTa-base, EfficientNet-B0/B3, CNN (Mel-spectrogram), LSTM.
- GPU access for fine-tuning (see TRD §10 hardware table).
- Guide sign-off (Dr. Manpreet Kaur) at each evaluation milestone.

## 11. Related Documents

`SRS.md` (requirements) · `TRD.md` (architecture) · `AGENTS.md` (agent/dev conventions) · `TASK.md` (backlog) · `IMPLEMENTATION.md` (build runbook) · `CLAUDE_CODE_PROMPT.md` · `ANTIGRAVITY_PROMPT.md`
