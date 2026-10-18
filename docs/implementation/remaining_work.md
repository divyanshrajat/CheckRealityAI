# Prioritized Implementation & Research Backlog
## CheckReality AI — Multimodal Deepfake and Misinformation Detection System

**Document Version:** 1.0  
**Audit Date:** 18 October 2026  
**Document Purpose:** Prioritized implementation checklist, pending tasks, dependency graphs, and concrete acceptance criteria  
**Repository State:** Verified against Git Commit `4ac9f2f`

---

## 1. Prioritization Framework

Tasks are strictly prioritized according to academic urgency and project compliance requirements:

- **Priority 1 (Correctness & Reproducibility):** Zero data leakage, verified model checkpoints, deterministic test suites, elimination of hardcoded/unmeasured numbers. *(Status: COMPLETED)*
- **Priority 2 (Explicit Synopsis Commitments):** Core modality pipelines, decision gating, explainability, web application, and baseline comparisons. *(Status: COMPLETED / VERIFIED)*
- **Priority 3 (SLR Design Requirements DR1–DR8):** Calibration, perturbation stress testing, latency profiling, and drift monitoring. *(Status: SUBSTANTIALLY COMPLETED)*
- **Priority 4 (Academic Deliverables for Viva & Paper):** Formal report documentation, compliance matrices, claim mappings, and viva slide deck alignment. *(Status: IN PROGRESS)*
- **Priority 5 (Post-B.Tech Strategic Research Extensions):** Joint intermediate cross-attention training, external benchmark ingestion, edge quantization. *(Status: FUTURE SCOPE)*

---

## 2. Comprehensive Implementation & Experimental Backlog

| Task ID | Component / Area | Task Description | Priority | Dependencies | Status | Concrete Completion / Acceptance Criteria | Blockers / Required Resources |
|---|---|---|---|---|---|---|---|
| **TASK-COR-01** | Test Suite | Execute end-to-end integration test verifying all 5 models, inference routes, fusion engine, and XAI outputs. | P1 | Local checkpoints | **DONE** | `backend/test_integration.py` exits with code 0 (5/5 test suites passing). | None (Verified 18 Oct 2026). |
| **TASK-COR-02** | Integrity Audit | Replace hardcoded benchmark claims with real empirical JSON logs or explicit `NOT_MEASURED` markers. | P1 | Benchmark runners | **DONE** | Commit `4ac9f2f` verified; zero simulated numbers in documentation. | None. |
| **TASK-DOC-01** | Compliance | Create formal `synopsis_scope_alignment.md` tracing Objectives 1–4 and methodology. | P2 | Synopsis document | **DONE** | File created with complete 10-column traceability matrix. | None. |
| **TASK-DOC-02** | Compliance | Create `dr1_dr8_status.md` auditing all 8 SLR design requirements with mathematical formulas. | P2 | SLR requirements | **DONE** | File created with status, evidence files, and limitations. | None. |
| **TASK-DOC-03** | Compliance | Create `current_limitations.md` detailing hardware, data, model, and operational boundaries. | P2 | Empirical audits | **DONE** | File created with clear non-goals and constraint registers. | None. |
| **TASK-DOC-04** | Research | Create `future_scope_and_future_work.md` with strict demarcation of incomplete scope vs future research. | P2 | Scope matrix | **DONE** | File created detailing 9 strategic future research directions. | None. |
| **TASK-DOC-05** | Research | Create `paper_claims_and_evidence.md` mapping paper statements to empirical evidence. | P2 | Benchmark JSONs | **DONE** | File created with conservative academic phrasing and caveat registers. | None. |
| **TASK-DOC-06** | Evaluation | Create `experimental_protocol.md` formalizing splits, preprocessing, metrics, and seed controls. | P2 | Codebase scripts | **DONE** | File created with complete mathematical formulations. | None. |
| **TASK-DOC-07** | Evaluation | Create `results_summary.md` compiling verified numbers from all benchmark JSON files. | P2 | Benchmark JSONs | **DONE** | File created with exact P50/P95 latencies, ECE, and AUC tables. | None. |
| **TASK-DOC-08** | Implementation | Create `remaining_work.md` prioritizing backlog and pre-viva deliverables. | P2 | Master prompt | **DONE** | File created and maintained. | None. |
| **TASK-VIV-01** | Viva Readiness | Update presentation slides and oral defense script to align with empirical findings (Agreement Index = 0.189, CPU latency ~470ms). | P2 | `results_summary.md` | **PENDING** | Oral notes match verified empirical numbers exactly; no discrepancies in viva slides. | None (Immediate next action). |
| **TASK-EXT-01** | External Data | Ingest Celeb-DF v2 and In-the-Wild audio corpora for true cross-dataset transfer evaluation. | P5 | Institutional DUA | **DEFERRED** | Empirical cross-dataset transfer ROC-AUC reported in `models/cross_dataset_benchmark.json`. | Requires institutional DUA and ~50GB storage. |
| **TASK-EXT-02** | Feature Fusion | Train intermediate cross-modal attention module (`ml/fusion/attention.py`) end-to-end. | P5 | 4-modality corpus | **DEFERRED** | Loss converges on joint multimodal classification task; weights saved to checkpoint. | Requires curated 4-modality aligned dataset. |
| **TASK-EXT-03** | Conformal Pred | Fit split conformal calibration to produce prediction sets with $1-\alpha = 95\%$ coverage guarantees. | P5 | Calibration split | **DEFERRED** | Empirical coverage evaluated on held-out test split with set size reported. | Requires paired multimodal calibration split. |
| **TASK-EXT-04** | Quantization | Convert PyTorch backbones to ONNX / TensorRT INT8 precision and profile latency. | P5 | TensorRT toolkit | **DEFERRED** | Latency reduced by $> 2.5\times$ with $< 0.5\%$ loss in validation ROC-AUC. | Requires NVIDIA TensorRT environment. |

---

## 3. Dependency Graph & Immediate Viva Milestones

```
[TASK-COR-01: Integration Tests (DONE)] ──┐
[TASK-COR-02: Integrity Audit (DONE)]    ├──► [TASK-DOC-01 to 08: Core Documentation (DONE)]
                                         │
                                         └──► [TASK-VIV-01: Viva Slide Deck Alignment (PENDING)]
                                                        │
                                                        ▼
                                              [FINAL VIVA DEFENSE]
                                                        │
                                                        ▼
                                              [POST-B.TECH RESEARCH]
                                              ├── TASK-EXT-01: Celeb-DF v2 Evaluation
                                              ├── TASK-EXT-02: Joint Attention Training
                                              └── TASK-EXT-04: TensorRT INT8 Optimization
```

---

## 4. Acceptance Criteria for Viva Examination & Thesis Submission

1. **Zero Discrepancy Standard:** Every metric cited in the Project Report, PPT slides, and research paper must match the serialized JSON reports in `benchmarks/` and `models/`.
2. **Conservative Language Compliance:** The term "truth detector" or "universal deepfake detector" must not appear in any official deliverable. The system must be presented as a *decision-support authenticity classifier*.
3. **Transparent Non-Measurement Disclosure:** Examiners must be explicitly informed that independent cross-dataset transfer and streaming production drift are `NOT_MEASURED` due to data access and offline lab constraints.
4. **Reproducibility Guarantee:** Any evaluator executing `python backend/test_integration.py` must observe 5/5 passing test suites without external network calls or manual parameter intervention.
