# Software Requirements Specification (SRS)
## CheckReality AI — Multimodal Deepfake & Misinformation Detection System

Version 1.0 · 15 Sep 2026 · Based on IEEE 830-style structure
Author / Developer: Divyansh Rajat (Roll No. 2300358) · Guide: Dr. Manpreet Kaur

---

## 1. Introduction

### 1.1 Purpose
This SRS specifies the functional and non-functional requirements for CheckReality AI, a web platform that analyzes text, image, video, and audio content and returns an authenticity verdict, confidence score, and explanation. It is the requirements source of truth for `TRD.md` (design), `TASK.md` (backlog), and all AI-agent build prompts.

### 1.2 Scope
The system accepts one or more content modalities from a signed-in user, runs modality-specific deep-learning classifiers, fuses their outputs, calibrates confidence, generates an explainable report, and persists results to the user's history. It is a **decision-support** tool, not a forensic or legal authority (Synopsis §8).

### 1.3 Definitions, Acronyms, Abbreviations
| Term | Meaning |
|---|---|
| XAI | Explainable AI |
| FPR | False Positive Rate |
| ROC-AUC | Receiver Operating Characteristic – Area Under Curve |
| Fusion Engine | Module combining per-modality predictions into one verdict |
| Modality | One of: text, image, video, audio |
| Bonafide | Genuine (non-spoofed) audio class, per ASVspoof terminology |
| JWT | JSON Web Token, used for session auth |

### 1.4 References
Synopsis document (`Project-2_Synopsis.pdf`, 21 pp., IKGPTU B.Tech CSE synopsis); guide info-doc (`CheckReality_InfoDoc.pdf`); synopsis references [1]–[33] therein.

---

## 2. Overall Description

### 2.1 Product Perspective
Standalone web application: browser client (frontend) → REST API (backend) → ML inference layer → database + object storage. No dependency on an existing product; new development.

### 2.2 Product Functions (summary)
Account management · content upload · text authenticity detection · image forgery detection · deepfake video detection · synthetic voice detection · cross-modal fusion · confidence calibration · explainable reporting · submission history.

### 2.3 User Classes and Characteristics
| Class | Technical level | Primary use |
|---|---|---|
| Anonymous visitor | Any | View landing page, sign up |
| Registered user | Low–medium | Upload content, view own reports/history |
| Evaluator / guide | Medium–high | Review evaluation metrics, methodology, demo |
| (Future) Admin | High | Model version management, usage monitoring — **out of scope for v1**, noted for extensibility |

### 2.4 Operating Environment
Modern evergreen browsers (Chrome, Edge, Firefox) on desktop; backend runs on Linux (Ubuntu) or Windows 11 per Synopsis §7 hardware table; GPU host required for training and recommended for inference.

### 2.5 Design and Implementation Constraints
- Must use the tech stack specified in the synopsis: Python 3.11+, FastAPI, PyTorch, Hugging Face Transformers, OpenCV, FFmpeg, Librosa, scikit-learn, MongoDB, SHAP/Grad-CAM, Git.
- Must fine-tune the specific pretrained backbones named in the synopsis (RoBERTa-base; EfficientNet-B0/B3; EfficientNet+LSTM; CNN on Mel-spectrograms) rather than substituting different architectures, to remain consistent with the approved synopsis.
- English-language text only in v1.
- Academic-project compute budget — see TRD §10 hardware table (RTX 3060/4060-class GPU, 8GB+ VRAM recommended).

### 2.6 Assumptions and Dependencies
- The five benchmark datasets (or a representative subset of each) are available locally or downloadable with credentials the team holds.
- Guide approval gates each major milestone.
- No dedicated production SLA is required (academic project), but the system should behave correctly under demo conditions.

---

## 3. System Features (Functional Requirements)

Priority key: **M** = must-have for viva/demo, **S** = should-have, **C** = could-have (stretch).

### 3.1 Authentication (AUTH)
| ID | Requirement | Priority |
|---|---|---|
| FR-AUTH-1 | System shall allow a new user to sign up with name, email, password. | M |
| FR-AUTH-2 | System shall hash and salt passwords before storage (never store plaintext). | M |
| FR-AUTH-3 | System shall allow a registered user to log in and receive a session token (JWT). | M |
| FR-AUTH-4 | System shall reject invalid/expired tokens on protected endpoints. | M |
| FR-AUTH-5 | System shall allow logout (client-side token invalidation at minimum). | S |
| FR-AUTH-6 | System shall show clear validation errors (weak password, duplicate email, malformed email). | S |

### 3.2 Content Upload (UPLOAD)
| ID | Requirement | Priority |
|---|---|---|
| FR-UP-1 | System shall accept text via paste or `.txt` upload. | M |
| FR-UP-2 | System shall accept image files (jpg/png/webp) up to a configured max size. | M |
| FR-UP-3 | System shall accept video files (mp4/mov) up to a configured max size/duration. | M |
| FR-UP-4 | System shall accept audio files (wav/mp3) up to a configured max size/duration. | M |
| FR-UP-5 | System shall validate file type/size client- and server-side and reject unsupported files with a clear message. | M |
| FR-UP-6 | System shall show upload progress and an analysis-in-progress state for long-running (video/audio) jobs. | S |

### 3.3 Text Analysis (TXT)
| ID | Requirement | Priority |
|---|---|---|
| FR-TXT-1 | System shall classify submitted text as Real/Fake news or Human/AI-generated using a fine-tuned RoBERTa-base model. | M |
| FR-TXT-2 | System shall return a confidence score with the text verdict. | M |
| FR-TXT-3 | System shall surface which spans/sentences most influenced the verdict (XAI, e.g., SHAP token attributions). | S |

### 3.4 Image Analysis (IMG)
| ID | Requirement | Priority |
|---|---|---|
| FR-IMG-1 | System shall detect manipulated/synthetic faces using a fine-tuned EfficientNet model. | M |
| FR-IMG-2 | System shall return a confidence score with the image verdict. | M |
| FR-IMG-3 | System shall provide a visual explanation (e.g., Grad-CAM heatmap over the face) alongside the verdict. | S |

### 3.5 Video Analysis (VID)
| ID | Requirement | Priority |
|---|---|---|
| FR-VID-1 | System shall extract frames and detect faces prior to classification. | M |
| FR-VID-2 | System shall classify video authenticity using EfficientNet (spatial) + LSTM (temporal) over the frame sequence. | M |
| FR-VID-3 | System shall return an overall verdict, confidence, and flag the frame range(s) most indicative of manipulation. | S |
| FR-VID-4 | System shall process video asynchronously and notify the user/UI when the report is ready. | M |

### 3.6 Audio Analysis (AUD)
| ID | Requirement | Priority |
|---|---|---|
| FR-AUD-1 | System shall convert audio to Mel-spectrogram representation. | M |
| FR-AUD-2 | System shall classify audio as Bonafide/Spoof (synthetic/cloned) using a fine-tuned CNN over the spectrogram. | M |
| FR-AUD-3 | System shall return a confidence score with the audio verdict. | M |

### 3.7 Fusion & Decision Engine (FUS)
| ID | Requirement | Priority |
|---|---|---|
| FR-FUS-1 | When a submission contains more than one modality (e.g., a video's visual + audio track), system shall normalize per-modality confidence scores to a common scale. | M |
| FR-FUS-2 | System shall combine normalized scores via a weighted ensemble into one authenticity score. | M |
| FR-FUS-3 | System shall calibrate the fused confidence score (e.g., temperature/Platt scaling) rather than exposing raw model output. | S |
| FR-FUS-4 | System shall degrade gracefully to single-modality output when only one modality is present. | M |

### 3.8 Explainability & Reporting (XAI / RPT)
| ID | Requirement | Priority |
|---|---|---|
| FR-XAI-1 | System shall generate a human-readable authenticity report containing: verdict, overall confidence, per-modality breakdown, and supporting evidence/reasons. | M |
| FR-XAI-2 | System shall clearly label the tool's output as decision-support, not a forensic/legal determination. | M |
| FR-RPT-1 | System shall persist each submission's report to the user's account. | M |
| FR-RPT-2 | System shall let a user browse and reopen past reports (history/dashboard). | S |
| FR-RPT-3 | System shall allow exporting a report as PDF. | C |

### 3.9 Evaluation & Admin (EVAL)
| ID | Requirement | Priority |
|---|---|---|
| FR-EVAL-1 | System shall include an offline evaluation harness producing accuracy/precision/recall/F1/FPR/ROC-AUC/inference-time per modality and for the fused system, on held-out test splits. | M |
| FR-EVAL-2 | System shall record which model checkpoint/version produced each stored report (for reproducibility). | S |

---

## 4. External Interface Requirements

### 4.1 User Interfaces
- Light theme, white background, clean single-column upload flow; dedicated Login and Signup pages; Dashboard/History; Report view with per-modality cards and an explanation panel. Full spec in `ANTIGRAVITY_PROMPT.md`.

### 4.2 Hardware Interfaces
None beyond the host machine's CPU/GPU/storage (see TRD §10).

### 4.3 Software Interfaces
- Backend exposes a REST/JSON API (see TRD §5) consumed by the frontend.
- Backend integrates PyTorch model artifacts loaded from local checkpoint storage.
- MongoDB is the system of record for users, submissions, and reports.

### 4.4 Communications Interfaces
HTTPS for all client–server traffic; JWT bearer tokens for authenticated requests.

---

## 5. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Performance** | Text/image verdicts should return in low single-digit seconds; video/audio may run asynchronously (see TRD §12 latency budgets). |
| **Scalability** | Architecture should not hard-code single-request assumptions; a queue-based design for heavy (video/audio) jobs is required so load can scale beyond a demo (Objective 7 / Synopsis §7 "Scalable"). |
| **Security** | Passwords hashed; JWT auth on all non-public endpoints; uploaded files validated by type/size before processing; secrets via environment variables, never committed. |
| **Reliability** | A failure in one modality's model must not crash the whole submission; return partial results with an error flag for the failed modality. |
| **Usability** | Non-technical users must understand the verdict and the "why" without reading documentation. |
| **Maintainability** | Each modality's model, preprocessing, and API route should be independently swappable (see AGENTS.md "adding a new modality/model"). |
| **Portability** | Must run on both Windows 11 and Ubuntu Linux dev machines per Synopsis §7. |
| **Explainability/Transparency** | Every verdict must ship with at least one piece of supporting evidence, per FR-XAI-1. |
| **Ethical/Compliance** | UI must state the tool is not a substitute for expert forensic/legal verification (Synopsis §8). |

---

## 6. Other Requirements

### Appendix A — Datasets
| Dataset | Modality | Approx. size | Classes | Purpose |
|---|---|---|---|---|
| Kaggle Fake & Real News | Text | ~44k articles | Fake / Real | Fake news detection |
| AI vs Human Text | Text | ~487k+ samples | AI / Human | AI-generated text detection |
| FaceForensics++ | Video | ~1,000 original + manipulated videos | Real / Fake | Deepfake video detection |
| 140k Real & Fake Faces | Image | ~140,000 images | Real / Fake | Face forgery detection |
| ASVspoof 2019 | Audio | ~250k+ utterances (train/dev/eval) | Bonafide / Spoof | Synthetic/cloned voice detection |

### Appendix B — Pretrained Models
| Task | Pretrained model | Fine-tune on |
|---|---|---|
| Text | RoBERTa-base | Fake News + AI vs Human Text |
| Image | EfficientNet-B0/B3 | 140k Real/Fake Faces |
| Video | EfficientNet + LSTM | FaceForensics++ |
| Audio | CNN (Mel-Spectrograms) | ASVspoof 2019 |

### Appendix C — Evaluation Metrics
Accuracy, Precision, Recall, F1-score, False Positive Rate, ROC-AUC, Inference Time, Computational Efficiency, Cross-dataset Generalization (Synopsis §6 Step 7, §7).
