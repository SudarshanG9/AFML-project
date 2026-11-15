# AVFuse-CausalQ

## IO-Aware, Precision-Controlled Audio-Visual Fusion for Deepfake Detection

**A controlled experimental study of how simulated low-precision attention affects the forensic decisions of a frozen, pretrained audiovisual deepfake detector—and whether selectively preserving precision improves those decisions at a measured computational cost.**

AVFuse-CausalQ combines a real pretrained **DiMoDif** detector, **LAV-DF** whole-video data, four precisely defined attention modes, source-aware evaluation, CPU benchmarking, reproducible artifacts, and an eight-page **Streamlit research dashboard**. The accepted scientific study is complete; this repository also documents its negative findings and limitations rather than presenting the hoped-for optimization as a success.

> **Main finding:** On the selected LAV-DF cohort, simulated INT8-style rounding of joint-attention activations caused small numerical changes, but **overall precision-induced degradation was not statistically established**. Restoring precision in **V and the attention output** was **not shown to outperform restoring Q**. The measured costs are for floating-point fake quantization on cached features, **not native INT8 acceleration**.

**Project status (November 2026):** Three-member backend, frontend, accepted research evidence, and CI are integrated on [`main`](https://github.com/SudarshanG9/AFML-project). Software tests pass; a complete manual browser walkthrough and an independently demonstrated real-checkpoint run through the Streamlit form remain disclosed validation limitations. See the [Phase 5 readiness report](docs/integration/PHASE_5_RELEASE_READINESS_REPORT.md).

### Contents

1. [Problem statement and research question](#1-problem-statement-and-research-question)
2. [Research hypotheses and objectives](#2-research-hypotheses-and-objectives)
3. [Technical choices, models, and datasets](#3-technical-choices-models-and-datasets)
4. [System architecture and complete data flow](#4-system-architecture-and-complete-data-flow)
5. [Research design and evaluation methodology](#5-research-design-and-evaluation-methodology)
6. [Project structure and team ownership](#6-project-structure-and-team-ownership)
7. [Research journey: experiments, findings, and decisions](#7-research-journey-experiments-findings-and-decisions)
8. [Final experimental results](#8-final-experimental-results)
9. [Conclusions and answers to the hypotheses](#9-conclusions-and-answers-to-the-hypotheses)
10. [Using the system and reproducing the evidence](#10-using-the-system-and-reproducing-the-evidence)
11. [Limitations and future directions](#11-limitations-and-future-directions)
12. [References, datasets, and further reading](#12-references-datasets-and-further-reading)

---

## 1. Problem statement and research question

### The problem in simple terms

Deepfake detection becomes difficult when a video and its soundtrack can be manipulated **independently**. A clip might show a real person speaking with generated audio; an authentic voice might accompany an altered face; or both modalities may be fake. Inspecting only the image or only the sound can miss evidence present in the other stream.

An audiovisual (AV) deepfake detector jointly processes video and audio features. A transformer-style **attention** mechanism decides which parts of those sequences are relevant to one another. But the attention operation must move and calculate numerical values, which can be expensive on constrained hardware.

A common efficiency idea is to **reduce numerical precision**—for example, representing some values on an 8-bit grid instead of preserving their full 32-bit floating-point resolution. This raises a forensic question: **Could a seemingly small rounding error erase a subtle clue that distinguishes authentic from manipulated content?** And if so, could we preserve precision only in the important part of attention rather than everywhere?

It is not sufficient to show that an attention tensor looks similar or that a model runs faster. We must measure whether the **true-label detection decisions** change, separately for audio-only, video-only, and combined manipulation, and quantify the cost of each choice.

### Research question

> Under a source-disjoint, manipulation-stratified evaluation, does reducing numerical precision inside audiovisual fusion measurably change deepfake detection decisions? If damage occurs, can keeping higher precision in the **value/attention-output path** recover more of it than keeping higher precision in the **query path**, at a comparable cost measured on real hardware?

**Scope:** We investigate *one frozen detector and its attention arithmetic*. We do **not** claim to invent deepfake detection, train a new model, develop a new quantization algorithm, build a custom FlashAttention alternative, or demonstrate deployment-ready INT8 speedups.

For context, see [research foundations](docs/01_RESEARCH_FOUNDATIONS.md) and [literature/novelty review](docs/02_LITERATURE_REVIEW_AND_NOVELTY.md).

## 2. Research hypotheses and objectives

The project began with three explicit hypotheses. They describe **what we expected**, not what the final data necessarily proved.

| ID | Original expectation | How we tested it | Final assessment |
| --- | --- | --- | --- |
| **H1 — IO-aware engineering** | Real attention shapes would not justify writing a new custom tiled attention kernel; an existing optimized implementation should suffice for this scope. | Inspect real Q/K/V shapes, masks, analytical dense-memory size, dense-versus-SDPA correctness, and recorded runtime. | **Supported as an engineering scope decision**, not proof that no device could have a memory bottleneck. |
| **H2 — Precision sensitivity** | Rounding Q/K/V and the attention output to a simulated INT8 grid would measurably worsen label-grounded detection, potentially differently for audio, video, and combined fakes. | Compare paired FP32 and low-precision outputs using BCE, AUROC, AP, recall, and FPR, overall and by fake type. | **Overall degradation not supported**; manipulation-dependent degradation **inconclusive**. |
| **H3 — Selective precision restoration** | Preserving V and the attention output should recover more detection quality than preserving Q, ideally at comparable extra cost. | Compare V/output restoration, Q restoration, and low precision on the *same clips*, using paired loss intervals and measured runtime. | **Overall V/output advantage not supported**; comparable restoration cost **not established**. |

Our intended contribution was a **falsifiable, reproducible answer**, even if the optimal precision policy we hoped to discover did not exist for this detector. This principle is important: *a well-executed experiment with an unsupported hypothesis is still a research result.*

### What we set out to deliver

- A frozen pretrained detector operating on the **same cached video/audio features** in four configurations.
- Per-video fake probabilities, with input and checkpoint identities preserved.
- Accuracy and probability-quality measurements against actual labels, **by manipulation type**.
- Paired, **source-grouped 95% confidence intervals**, rather than relying on tiny differences between rounded scores.
- Measured operator and model costs with explicit boundaries and hardware details.
- Reusable code, experiment/provenance records, a CLI, and an interactive research dashboard.

### 2.1 How a user experiences the end system

**Think of AVFuse-CausalQ as a research laboratory you can inspect, rather than a website where you upload any video and instantly get a fake/real verdict.** The end system has two distinct uses: (A) explore the **already completed, accepted experiment**, which needs no private checkpoint or dataset; and (B) run a **new, explicitly authorized validation-only experiment** through the real pretrained backend, which requires compatible external assets. The dashboard clearly separates those two sources of evidence.

```mermaid
flowchart TD
    U[User opens the Streamlit research dashboard] --> C{What would you like to do?}
    C -->|Understand the completed study| A[Read Overview and inspect the dataset and fusion diagnostics]
    A --> E[Open Evaluation: accepted metrics and paired confidence intervals]
    E --> B[Open Benchmarks: saved CPU timing measurements]
    B --> F[Interpret hypotheses, findings and limitations]
    C -->|Run a permitted new validation experiment| V[Supply authorized DiMoDif checkout, checkpoint, manifest and cached AV features]
    V --> Q[Confirm inputs and choose precision modes]
    Q --> R[Streamlit calls the actual Member 3 validation and run CLI]
    R --> M[Member 1 loads frozen DiMoDif and computes fake probabilities]
    M --> P[Member 3 saves predictions, benchmarks and provenance]
    P --> I[Inspect outputs and explicitly verify run integrity]
    I --> G[Use Member 2 evaluation separately if an authorized report is needed]
```

#### Path A — Explore the completed research (recommended first)

1. **Launch the dashboard.** Follow [Section 10](#10-using-the-system-and-reproducing-the-evidence) or the [frontend launch guide](frontend/README.md). The application opens locally; merely viewing a page **does not load the pretrained model or rerun the held-out test**.
2. **Start at _Overview_.** Read the objective, implementation status and available research artifacts. A successful software-interface check means the relevant Python code imports; it does **not** mean an external checkpoint was loaded or real inference completed.
3. **Visit _Dataset & features_.** Learn how whole LAV-DF videos are grouped into real, audio-fake, video-fake and both-fake categories, and why cached audio/video encoder features are used instead of raw media at inference time.
4. **Open _Fusion diagnostics_.** Inspect saved attention, masking, ablation and memory evidence wherever present. These explain *where* numerical interventions happen; they are not themselves proof of better detection.
5. **Open _Evaluation_ → _Saved research results_.** Inspect the accepted four-mode AUROC, average precision, BCE, recall, false-positive rate, manipulation-specific results and paired 95% confidence intervals. These are **original saved measurements**, not numbers generated when the page opens. Pay particular attention to the paired BCE comparisons: intervals spanning zero do not establish overall precision damage or a V/output restoration advantage.
6. **Open _Benchmarks_.** Compare the saved CPU model runtimes across the four modes and three workloads. These measure **cached-feature model calls**, including applicable simulated-quantization work—not the time to decode, preprocess and classify a raw video, or a native INT8 hardware speedup.
7. **Visit _Runs & system_ and _Artifact import_ if you want to audit more evidence.** You can inspect discovered run folders, explicitly request an integrity check, or select supported local/uploaded artifacts. A file's presence or a previous integrity receipt is not proof of a completed, scientifically accepted experiment.

**What you should understand by the end of Path A:** what the team hypothesized, what changed mathematically in the frozen detector, how the results were measured, **what the data actually support**, and where the limits of that conclusion lie. No model download or new experiment is needed to read the accepted results.

#### Path B — Execute the real backend on authorized validation inputs (optional)

This path is for a developer/researcher who has the **pinned external DiMoDif source and LAV-DF checkpoint**, an appropriate **validation-only manifest**, and the **complete compatible precomputed feature cache**, with permission to use them. The dashboard does **not** accept arbitrary uploaded MP4s for raw-video deepfake inference; its model input consists of encoder-generated audio/video tensors.

1. Open **_Demo & predictions_** and select the approved LAV-DF model, feature format, device and one or more of the four precision modes. Enter a **new run ID**, provide the server-authorized filesystem paths, and explicitly confirm the inputs. The UI checks for validation-only use and enforces the agreed checkpoint identity.
2. The frontend calls `python -m avfuse.member3_runtime.cli validate-config`, followed—only when validation succeeds—by the actual `run` command. This is a real backend subprocess, **not a mock detector hidden behind the interface**.
3. **Member 3** loads the approved cached features; **Member 1** constructs the frozen DiMoDif model and runs the selected reference/precision-intervention paths. The runtime writes computed fake-score rows, scoped benchmarks, execution summaries and provenance. Missing inputs, incompatible modes, incomplete runs or failures are reported instead of replaced with fictional predictions.
4. Return to **_Demo & predictions_** or **_Runs & system_** to inspect discovered outputs. Perform explicit run-integrity verification where appropriate. If you need new **label-grounded evaluation metrics and statistical comparisons**, run the **separate Member 2 evaluator** with the permitted validation evidence and then inspect its report; completing inference alone does not automatically calculate a new AUROC or approve a research result.

**Important boundary:** The dashboard's **synthetic smoke test** checks software wiring only; it must not be presented as a checkpoint-backed prediction. The completed held-out study remains frozen, and no casual dashboard interaction should reopen or retune it. At the documented Phase 5 handoff, the frontend-to-backend integration passed automated tests, but a full live real-model validation run through the UI was **not independently demonstrated** with the complete authorized feature cache.

For full setup instructions, prerequisites and reproducibility limits, read [Section 10](#10-using-the-system-and-reproducing-the-evidence), the [frontend guide](frontend/README.md), and the [Phase 5 readiness report](docs/integration/PHASE_5_RELEASE_READINESS_REPORT.md).

## 3. Technical choices, models, and datasets

### 3.1 Detector selection: DiMoDif instead of AVFF

The original proposal favored **AVFF (Audio-Visual Feature Fusion)** because of its emphasis on audiovisual correspondence. A runnable official AVFF checkpoint could not be verified in time for the feasibility gate; a newly trained fusion model would also have introduced a different training problem. The team therefore adopted the accessible pretrained **DiMoDif** detector and its matching LAV-DF checkpoint.

**What actually runs:** DiMoDif's existing audiovisual deepfake detection architecture, loaded from the [official source](https://github.com/mever-team/dimodif), with its original trained weights and scoring head intact. Our code adds controlled **activation rounding within its joint-attention computation**; it does not retrain the detector.

| Component | Final choice | Why it matters |
| --- | --- | --- |
| Pretrained detector | **DiMoDif** (`AVDeepFakeDetector`, LAV-DF DFD checkpoint) | Real joint audiovisual inference with an inspectable model and frozen weights. |
| Visual feature encoder | auto-AVSR visual **`LRS3_V_WER19.1`** | Produces speech-related visual tokens compatible with the pretrained detector. |
| Audio feature encoder | auto-AVSR audio **`LRS3_A_WER1.0`** | Produces corresponding speech-related audio tokens. |
| Visual preprocessing | **MediaPipe** landmarks/tracking and upstream-compatible cropping | Aligns inputs with the encoder's preprocessing expectations. |
| Model framework | **PyTorch** | Model execution, SDPA, tensor operations and simulated quantization. |
| Attention implementations | **PyTorch SDPA** for execution; **dense oracle** for correctness checks | Reuse a tested attention path rather than building a new kernel. |
| Quantization | **Dynamic symmetric, per-tensor fake INT8** on selected activations | Isolates rounding sensitivity without claiming real integer hardware execution. |
| Statistical evaluation | Member 2's **source-group bootstrap and paired analysis** | Separates observed changes from uncertainty and source dependence. |
| Benchmarking | **CPU cached-feature model and operator timing** | Measures a named, limited cost boundary; excludes video decoding/feature extraction. |
| Application | **Streamlit** + existing **Member 3 Python CLI** | Inspects real artifacts and can request guarded validation-only backend execution. |

The adapter supports another checkpoint name at the software interface, but **only the LAV-DF checkpoint is the evaluated model in this study**. AVFF, AuViRe, AASIST-based alternatives, and other models discussed during selection were **not** additional trained/evaluated detectors in the final experiment.

### 3.2 Dataset selection and what was actually used

**Final evaluated dataset: [LAV-DF — Localized Audio Visual DeepFake Dataset](https://github.com/ControlNet/LAV-DF)**, using **selected whole videos** converted into a binary real/fake task and then stratified by manipulation type. Although LAV-DF also supports *temporal localization*, this project tests **whole-video classification**, not localization accuracy.

| Dataset or data source | Role in this project | Used for final model evaluation? | Access |
| --- | --- | --- | --- |
| **LAV-DF** | The actual video/metadata source for validation and held-out testing, plus the matching pretrained detector checkpoint. | **Yes: selected 1,200 validation and 2,000 test whole videos.** | [Official repository / terms](https://github.com/ControlNet/LAV-DF) · [Hugging Face access page](https://huggingface.co/datasets/ControlNet/LAV-DF) |
| **FakeAVCeleb** | Initial planned dataset because its audio/video manipulation categories matched the question. Access was delayed; the plan pivoted to LAV-DF. | **No.** Do not attribute final metrics to FakeAVCeleb. | [Official dataset repository and request instructions](https://github.com/DASH-Lab/FakeAVCeleb) |
| **LRS3-related pretrained encoder assets** | Upstream training context for auto-AVSR speech encoders consumed during feature extraction. | **No separate LRS3 evaluation** was performed in this project. | [auto-AVSR upstream models/instructions](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages) |

LAV-DF access is subject to the provider's terms and noncommercial license conditions. The repository deliberately **does not distribute the licensed videos, private complete feature cache, or original checkpoint weights**.

**Final selected cohort:**

| Split | Authentic | Audio fake | Video fake | Both fake | Total | Source groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 300 | 300 | 300 | 300 | **1,200** | 1,114 |
| Held-out test | 500 | 500 | 500 | 500 | **2,000** | 1,738 |
| **Total** | **800** | **800** | **800** | **800** | **3,200** | — |

The historical short-window pipeline processed an earlier selection (9,200 proposed windows; 9,143 usable). Those **superseded window counts are not the final study cohort**. We switched to whole videos because a short window can miss a localized fake edit while still inheriting the video's fake label.

**Leakage caveat:** Source-disjointness was audited, including zero selected-source overlap between validation/test and between selected evaluation sources and the recorded training sources. **Speaker-disjointness was not proven** because verified speaker IDs were unavailable. See the [dataset and methodology chapter](docs/05_DATASET_AND_EXPERIMENTAL_METHODOLOGY.md).

### 3.3 The four frozen numerical configurations

The study uses a **single pretrained model** with four attention modes—not four separately trained models.

| Mode (actual code name) | Query Q | Key K | Value V | Attention output O | Interpretation |
| --- | --- | --- | --- | --- | --- |
| `reference` | FP32 | FP32 | FP32 | FP32 | Original unmodified detector. |
| `low_precision` | Fake INT8 | Fake INT8 | Fake INT8 | Fake INT8 | Tests broad rounding sensitivity. |
| `restore_value_output` | Fake INT8 | Fake INT8 | FP32 | FP32 | Keeps the hypothesized information-carrying path at higher precision. |
| `restore_query` | FP32 | Fake INT8 | Fake INT8 | Fake INT8 | Control that preserves query precision instead. |

**Important:** “Fake INT8” means **round to a symmetric 8-bit grid and dequantize back to floating point**. The model's weights, original projections, softmax, residual/normalization paths, and detection head remain floating point. No native INT8 matrix-multiplication kernel, reduced integer storage, or production acceleration is claimed.

The V and output boundaries are restored **together**, and the same policy is applied **cumulatively to all five attention layers**. Consequently, this experiment cannot isolate the effect of V alone, O alone, or one particular layer.

## 4. System architecture and complete data flow

### 4.1 End-to-end research architecture

The system is **artifact-driven**: an offline preparation pipeline makes compatible features; Member 3 runs Member 1's real model and saves scores; Member 2 performs label-based evaluation separately; Streamlit displays accepted evidence and can request new *validation-only* development runs.

```mermaid
flowchart TB
    subgraph OFFLINE["OFFLINE PREPARATION — MEMBER 2"]
        A["LAV-DF videos and metadata"] --> B["Whole-video manifest and source audit"]
        A --> C["MediaPipe preprocessing and auto-AVSR encoders"]
        C --> D["Cached audio/video features — FP32, width 768"]
        B --> E["Canonical clip IDs, splits and source groups"]
    end

    subgraph RUNTIME["CONTROLLED INFERENCE — MEMBERS 3 + 1"]
        F["Member 3 CLI: validate-config / run / predict"]
        G["Feature loader: NPZ or compatible PT"]
        H["Member 1: frozen DiMoDif adapter"]
        I["Five joint self-attention layers + selected precision mode"]
        J["Original feature pyramid and detection head"]
        K["Predictions: run ID / clip ID / split / mode / score / checkpoint hash"]
        L["Validation CPU timing, run summaries and artifact hashes"]
        F --> G --> H --> I --> J --> K
        F --> L
        I --> L
    end

    subgraph ANALYSIS["EVIDENCE AND PRESENTATION"]
        M["Member 2: schema checks and label-based evaluation"]
        N["Metrics / source-group bootstrap / paired confidence intervals"]
        O["Accepted saved results + figures"]
        P["Streamlit: 8-page research dashboard"]
        K --> M --> N --> O --> P
        L --> P
        K --> P
    end

    D --> G
    E --> F
    E --> M
    Q["Withheld labels — evaluation only"] --> M
    P -->|"Explicit, confirmed validation request"| F
```

**The diagram shows several cooperating commands, not a single automatic raw-video-to-prediction API.** The offline extractor and statistical evaluator are separate from the UI's live inference command. Labels are joined only for evaluation; they are not features supplied to the detector.

### 4.2 What happens at each stage

1. **Select and audit whole videos.** Member 2 derives binary labels and `audio_fake` / `video_fake` / `both_fake` categories, and assigns stable IDs and source groups.
2. **Extract once, reuse many times.** MediaPipe and the upstream auto-AVSR encoders generate video and audio representations. Keeping the same features across modes prevents repeated decoding from confounding the comparison.
3. **Load validated cached features.** DiMoDif-compatible NPZ files contain `video_features` and `audio_features` arrays of shape `[T_v,768]` and `[T_a,768]`. The runtime checks finite floating-point values, dimensions, and required keys. It also supports a separately validated `shared_pt` contract.
4. **Reconstruct the original pretrained detector.** `DiMoDifAdapter` loads the upstream architecture and checkpoint with strict weight matching, calls `eval()`, and freezes parameters. It is **not a mock detector**.
5. **Prepare joint tokens.** The adapter truncates both feature sequences to the shared valid length, at most 600, pads each modality to 600, projects to 64 dimensions, and adds two special tokens. This yields **1,202 positions**.
6. **Apply the attention intervention.** Five existing joint self-attention layers use four heads. A typical Q/K/V tensor at the padded boundary has shape **`[1,4,1202,16]`**. The chosen mode rounds selected activations while preserving the original masking, learned weights, and output projection.
7. **Produce original-model fake scores.** The upstream feature-pyramid/detection head generates a probability per video. Each prediction row records `run_id`, `clip_id`, `split`, `config`, `score_fake`, and `checkpoint_sha256`—**no labels**.
8. **Evaluate independently.** Member 2 aligns predictions with canonical labels, applies the fixed threshold, computes metrics, and produces paired source-grouped intervals. This does **not** happen automatically every time Streamlit runs inference.
9. **Preserve and present artifacts.** Member 3 records runtime/provenance/integrity. Streamlit reads those artifacts as well as the **already accepted** aggregate CSVs and figures. The dashboard does not recompute or overwrite held-out results on launch.

### 4.3 Inside one joint-attention layer

For one attention head:

```text
Video tokens + audio tokens
          |
       Q, K, V  (original learned projections)
          |
    score = Q × Kᵀ / √16 + mask
          |
       softmax(score)
          |
     weighted values: P × V
          |
    attention output O
          |
    original output projection
          |
  residual, normalization and later layers
          |
     original detection head
```

The intervention's dynamic per-tensor scaling is:

`scale = max(max(abs(x)), tiny) / 127`

`fake_int8(x) = scale * clip(round(x / scale), -127, 127)`

The rounded output is **still a floating-point tensor**. Full technical equations and masking limitations are in [Chapter 4](docs/04_FINAL_SYSTEM_ARCHITECTURE.md).

### 4.4 How the frontend connects to the actual backend

The eight-page dashboard is implemented in `frontend/app.py` and `frontend/avfuse_ui/`. When a user explicitly requests an authorized validation run, the UI calls `run_cached_feature_experiment()` in `backend_status.py`. That adapter invokes the real Member 3 commands through **Python subprocesses**:

```text
Streamlit form
    → validate local/authorized paths and validation-only manifest
    → python -m avfuse.member3_runtime.cli validate-config ... --json
    → python -m avfuse.member3_runtime.cli run ... --split val --require-lavdf-hash --json
    → Member 3 builds Member 1's genuine DiMoDifAdapter
    → cached-feature predictions and scoped benchmark records
    → newly discovered artifact files and run-status UI
```

There is **no parallel dummy inference implementation** in the production UI. An optional CLI `--smoke` command uses a clearly labeled toy model **only to test infrastructure**, never to supply scientific results. Real inference requires the external checkpoint, matching configuration, authorized validation manifest, and complete compatible cache. The final Phase 5 report does not claim that this last real-asset browser demonstration was performed.

**Dashboard pages:** Overview · Dataset & features · Fusion diagnostics · Demo & predictions · Evaluation · Benchmarks · Runs & system · Artifact import.

## 5. Research design and evaluation methodology

| Element | Accepted procedure |
| --- | --- |
| Research unit | One selected **whole video**, scored under every precision mode. |
| Label | **0** authentic; **1** fake. Fake videos additionally have audio/video/both manipulation categories. |
| Data pairing | The same IDs, features, pretrained checkpoint and masks are used for all four configurations. |
| Operating threshold | **`0.6627789735794068`**, chosen using **reference validation** scores to target no more than 5% authentic FPR; then frozen for every mode. |
| Test isolation | Final policy/checkpoint/threshold frozen before accepted held-out scoring; historical earlier exposure to test-derived summaries is disclosed, not denied. |
| Primary loss | **Binary cross-entropy (BCE)**: lower is better; sensitive to wrong-label confidence, not just ranking. |
| Other metrics | **AUROC**, **average precision (AP/AUPRC)**, recall at the fixed threshold, and real false-positive rate (FPR). |
| Uncertainty | **1,000 source-group bootstrap resamples**, seed **0**, percentile **95% confidence intervals**. |
| Comparisons | **Paired differences** between modes on the same clips, with shared resampled source groups. |
| Manipulation analysis | Each fake subtype's **500 test fakes versus the 500 real videos**; subsets share real controls. |
| Runtime | Three validation anchors (short/typical/long); CPU, batch 1, 10 warmups and 50 timed repetitions per mode. |
| Prediction integrity | Expected IDs/modes, finite scores, one checkpoint identity, explicit run statuses and artifact hashes. |

**Reading a paired result:** All saved comparisons use `delta = mode_b − mode_a`. For BCE, a **positive** low-precision-minus-reference delta suggests damage; a **negative** restoration-minus-low delta suggests improvement. A **95% interval spanning zero does not establish a directional effect**, and it does **not prove equivalence**.

The original research has **80 saved paired intervals** across contrasts, metrics, and categories. These are nominal intervals, **not corrected for multiple comparisons**. For more on the protocol, source groups, metrics and caveats, see [Chapter 5](docs/05_DATASET_AND_EXPERIMENTAL_METHODOLOGY.md).

## 6. Project structure and team ownership

```text
AFML-project/
├── README.md                       # This project overview
├── pyproject.toml                  # Python package and dependencies
├── .github/workflows/ci.yml        # Python 3.10 / 3.12 integration CI
│
├── src/avfuse/
│   ├── member1_fusion/             # MEMBER 1 — pretrained model and attention
│   │   ├── dimodif_adapter.py      # Real DiMoDif architecture/checkpoint adapter
│   │   ├── dimodif_intervention.py # Temporary interventions in original attention
│   │   ├── attention.py            # Dense oracle and SDPA path
│   │   ├── precision.py            # Fake-INT8 numerical operations
│   │   └── model_adapter.py        # Stable backend prediction interface
│   │
│   ├── member2_data/               # MEMBER 2 — data and scientific evaluation
│   │   ├── wholevideo.py           # Whole-video cohort preparation
│   │   ├── dimodif_features.py     # Compatible cached-feature preparation
│   │   ├── manifest.py / splits.py # IDs, split grouping and contracts
│   │   ├── predictions.py          # Prediction verification
│   │   ├── evaluate.py / paired.py # Metrics and paired source bootstrap
│   │   └── shortcut_control.py     # Trailing-silence validity control
│   │
│   ├── member3_runtime/            # MEMBER 3 — inference/benchmark orchestration
│   │   ├── cli.py                   # validate-config/run/predict/benchmark/inspect
│   │   ├── experiment.py            # Real adapter + run orchestration
│   │   ├── feature_loader.py        # NPZ/PT contract loaders
│   │   ├── runner.py                # Four-mode model execution
│   │   ├── model_benchmark.py       # Scoped cached-feature CPU timing
│   │   ├── artifacts.py             # Run manifests and integrity
│   │   └── provenance.py            # Input/output and environment lineage
│   └── shared/                     # Shared precision-mode and JSONL contracts
│
├── frontend/
│   ├── app.py                       # Eight-page Streamlit application
│   ├── avfuse_ui/
│   │   ├── backend_status.py        # Real Member 3 CLI subprocess bridge
│   │   ├── data_access.py           # Real schema and saved-result readers
│   │   ├── run_inspection.py        # Explicit run integrity verification
│   │   └── ui.py                    # Page layouts and user interactions
│   ├── tests/                       # Frontend and integration tests
│   └── README.md                    # UI setup and artifact handling
│
├── scripts/                         # Audits, whole-video extraction, evaluation CLI
├── contracts/schemas/               # Manifest/prediction/benchmark schemas
├── configs/                         # Benchmark examples and frozen experiment records
├── tests/                           # Member-specific and cross-module tests
├── data/                            # Documentation and small historical pilot assets
├── artifacts/                       # Documented structure and limited historical records
├── docs/
│   ├── 01_... through 10_...        # Scientific chapters (see reading guide below)
│   ├── results/                    # Preserved aggregate CSV extracts
│   ├── member1/ / member2/         # Member evidence and original result figures
│   └── integration/                # Phases 1–5 acceptance/verification records
└── third_party/dimodif             # Historical gitlink; external checkout required
```

**Boundaries between teammates:** Member 1 owns the **model and controlled numerical intervention**; Member 2 owns **which clips, labels, metrics and confidence intervals are trustworthy**; Member 3 owns **running the frozen model, timing it, and preserving verifiable artifacts**. The dashboard consumes these interfaces—it does not replace the backend.

Detailed package documentation: [Member 1](src/avfuse/member1_fusion/README.md) · [Member 2](src/avfuse/member2_data/README.md) · [Member 3](src/avfuse/member3_runtime/README.md) · [Architecture](docs/04_FINAL_SYSTEM_ARCHITECTURE.md).

## 7. Research journey: experiments, findings, and decisions

The following sequence describes **what was actually done**, including changes from the original plan. “Day B,” “C9,” and “C10” are experiment/release identifiers, not an assertion that every item was finished on a literal consecutive calendar day.

| Stage | Investigation and evidence | Decision or outcome |
| --- | --- | --- |
| **1. Research framing** | Identified the gap between generic attention efficiency and **label-grounded audiovisual forensic correctness** under precision reduction. | Narrowed scope to one pretrained detector, four modes, manipulation-stratified tests and measured cost. |
| **2. Model feasibility** | AVFF was considered; an executable official checkpoint was not verified at the decision gate. DiMoDif source/config/weights were available for inspection. | **Selected pretrained DiMoDif** instead of training a new AV fusion model. |
| **3. Dataset and preprocessing** | FakeAVCeleb access was delayed. An initial short-window LAV-DF selection retained 9,143 usable windows but could miss localized edits. | **Pivoted to whole-video LAV-DF**, preserving 1,200 validation and 2,000 test clips. |
| **4. Joint-attention feasibility** | Audited five layers, QKV shape `[1,4,1202,16]`, original masks, a **22.046 MiB** analytical FP32 score matrix/layer and dense-vs-SDPA parity. | **No custom attention kernel**; retained the dense correctness oracle plus SDPA. No measured universal memory-no-bottleneck claim. |
| **5. Precision-policy implementation** | Built dynamic per-tensor fake INT8 and four modes over Q/K/V/output. Pilot zero-audio/zero-video ablations changed detector outputs. | Frozen pretrained weights; preserved original head; defined precise, testable interventions. |
| **6. Validation baseline and comparisons** | All four modes evaluated on the same 1,200 validation videos. Low-minus-reference BCE = **−0.000043144** with an interval spanning zero. | **Could not establish damage requiring recovery**; retained the Q control and avoided claiming a restoration win. |
| **7. C9 scientific freeze** | Fixed threshold, four configurations, checkpoint, cohort and a reference-preferred reporting policy. | Continued to held-out testing **without selecting a test-score-favored policy**. Recorded historical test-summary exposure and other qualifications. |
| **8. Held-out execution and recovery** | Parent C10 failed from a missing input; R1 was interrupted; accepted R2 completed **2,000 predictions per mode**. | Preserved earlier failure/recovery history and the completed **8,000-row** prediction handoff. |
| **9. Final evaluation and validity** | Member 2 performed paired source bootstrap on the accepted R2 outputs; a reference-only trailing-silence matched control was retained. | Overall H2/H3 effects remained unsupported. Reported the one exploratory subtype interval without elevating it to a universal claim. |
| **10. Benchmarks and system integration** | Measured three CPU model workloads, validated artifact contracts, integrated the eight-page frontend, tested Python 3.10/3.12. | Released an **artifact-driven research system** with reproducibility checks, documented limits and a conditional human-browser sign-off. |

Historical details and the original failure ledgers are in [Chapter 3](docs/03_RESEARCH_DECISIONS_AND_PIVOTS.md), [Chapter 7](docs/07_FAILURES_LIMITATIONS_AND_HYPOTHESES.md) and the [integration reports](docs/integration/).

## 8. Final experimental results

**These are accepted, saved original results—not new scores computed by opening this README or the dashboard.** The source-linked CSV extracts are in [`docs/results/`](docs/results/). Unrounded values and original source hashes are preserved there.

### 8.1 Held-out test: all four modes

**Test set:** 2,000 whole LAV-DF videos (500 real and 1,500 fake), 1,738 source groups. **Fixed threshold:** `0.6627789735794068`.

| Mode | AUROC ↑ | Average precision ↑ | BCE ↓ | Fake recall ↑ | Real FPR ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| **FP32 reference** | **0.998044000** | **0.999352103** | 0.067792265 | 0.992 | 0.042 |
| Low precision | 0.998033333 | 0.999348271 | 0.068121211 | 0.992 | 0.042 |
| V/output restored | 0.998037333 | 0.999349608 | 0.068176098 | 0.992 | 0.042 |
| Q restored | 0.998040000 | 0.999350411 | **0.067693347** | 0.992 | 0.042 |

**How to read this:** All four configurations retain high aggregate ranking performance on *this selected in-dataset cohort*. Their BCE values are not identical even when recall and FPR are the same at the fixed threshold. A slightly lower observed BCE is **not proof of a statistically better mode**. The reference-preferred policy was frozen earlier rather than reselected from these test values.

### 8.2 Paired loss comparisons: the main hypothesis tests

| Paired test-BCE contrast (`b − a`) | Observed difference | Source-grouped 95% CI | What it means |
| --- | ---: | --- | --- |
| **Low precision − reference** | +0.000328946 | [−0.000237371, +0.000962465] | Slight positive damage estimate, but **not established**; CI crosses zero. |
| **V/output restored − low precision** | +0.000054887 | [−0.000386235, +0.000493064] | No demonstrated recovery; estimate is slightly worse. |
| **Q restored − low precision** | −0.000427864 | [−0.000977544, +0.000092703] | Suggestive recovery point estimate, **not established**. |
| **V/output restored − Q restored** | +0.000482751 | [−0.000164867, +0.001185028] | No demonstrated V/output advantage; point estimate favors Q, with uncertainty. |

All four **overall** paired confidence intervals contain zero. **We cannot conclude that low precision causes a consistent overall loss, that restoration reliably repairs it, or that V/output beats Q.** We also **cannot conclude equivalence**: no equivalence margin or equivalence trial was specified.

### 8.3 What happened by manipulation type?

The reference detector's selected held-out results are:

| Fake category (each vs 500 real clips) | Reference AUROC | Reference fake recall | Interpretation |
| --- | ---: | ---: | --- |
| **Audio fake** | 0.999028 | 0.994 | High observed performance. |
| **Video fake** | **0.995472** | **0.984** | Descriptively the hardest subtype for this detector/cohort. |
| **Both fake** | 0.999632 | 0.998 | Highest observed reference AUROC among the three. |

These baseline differences do **not** establish that quantization harms the categories by different amounts. The one nominal zero-excluding paired interval among 80 contrasts concerns **audio-fake BCE(V/output − Q) = +0.001167749**, 95% CI **[+0.000152772, +0.002482882]**, favoring Q for that subset/metric. Because the comparisons are correlated and not multiplicity-adjusted, this is **exploratory**, not proof of reliable Q superiority.

### 8.4 Shortcut check: is trailing silence doing the work?

A reference-only control matched real and fake clips within ten trailing-silence bins. It retained **912** selected videos:

| Metric | Original 2,000 | Silence-matched 912 |
| --- | ---: | ---: |
| Trailing-silence-only AUROC | 0.659703333 | 0.506073984 |
| Frozen reference detector AUROC | 0.998044000 | 0.997234726 |

The simple silence signal became nearly uninformative in the matched subset, while the detector retained strong ranking there. This weakens a **trailing-silence-only explanation** for that selected result. It does **not** establish independence from duration, speaker identity, codec artifacts, other shortcuts or all precision modes.

### 8.5 CPU benchmark: numerical sensitivity is not hardware acceleration

**Accepted environment:** Intel Xeon CPU @ 2.20 GHz, batch 1, resident cached features, 10 warmups, 50 timed repeats per mode/workload; three validation anchor workloads; 12 result records and 600 timing observations. The following are *model-level* p50 latencies, not video-upload-to-result times.

| Mode | Short p50 (ms) | Typical p50 (ms) | Long p50 (ms) |
| --- | ---: | ---: | ---: |
| FP32 reference | 115.842 | **118.454** | 121.196 |
| Low precision | 120.460 | **161.434** | 158.258 |
| V/output restored | 117.882 | **119.918** | 121.531 |
| Q restored | 171.228 | **167.474** | 157.677 |

In the **typical** recorded workload, fake low precision was *slower*, not faster, than the FP32 reference. That is unsurprising: the simulation adds reductions/rounding/dequantization while retaining floating-point model math. Workloads have different valid input lengths but are all padded to 1,202 positions. Timing noise, unequal restoration boundaries and the lack of CPU peak-memory measurements prevent an equal-cost restoration or memory-savings claim.

**Not measured:** native INT8 kernels, true low-bit storage, energy efficiency, reliable CPU peak tensor memory, or complete raw-video end-to-end latency. See [full results and original benchmark scope](docs/06_EXPERIMENTS_AND_RESULTS.md).

### Original figures

![Original held-out four-mode ROC and precision–recall curves](docs/member2/results/test_figures/curves.png)

![Original held-out manipulation-specific recall](docs/member2/results/test_figures/recall_by_type.png)

**Evidence trail:** [Overall metrics](docs/results/test_overall_metrics.csv) · [Manipulation metrics](docs/results/test_manipulation_metrics.csv) · [All 80 paired comparisons](docs/results/test_paired_comparisons.csv) · [Validation metrics](docs/results/validation_overall_metrics.csv) · [CPU timing summaries](docs/results/cpu_model_runtime.csv) · [All 600 CPU model timing samples](docs/results/cpu_model_timing_samples.csv) · [Evidence registry](docs/results/evidence_claims.csv).

## 9. Conclusions and answers to the hypotheses

**Direct answer:** For this frozen DiMoDif/LAV-DF detector and this dynamic fake-INT8 activation scheme, the observed precision effects on aggregate forensic metrics were small. The accepted paired analysis **did not demonstrate overall damage**, nor a **reliable V/output restoration advantage over Q**. The equal-cost part of the original optimization hypothesis was not achieved, and the experiment does not establish native low-precision acceleration.

- **H1:** Reusing SDPA and a dense oracle was the defensible engineering choice given the verified shapes, parity and scope. A general memory-bottleneck-free claim was not tested.
- **H2:** The proposed overall degradation was **not supported**. Manipulation-specific sensitivity remains **inconclusive**; this does not mean quantization is universally harmless.
- **H3:** The proposed superiority of preserving V/output was **not supported**. A valid equal-cost comparison was **not established**.

### What this project genuinely contributes

1. A frozen pretrained detector with **four transparent attention-precision interventions** rather than differently trained baselines.
2. A carefully identified **whole-video, manipulation-stratified cohort** and separate source-aware label evaluation.
3. A negative or inconclusive finding made interpretable by **paired uncertainty**, rather than only rounded leaderboard scores.
4. Real but **strictly scoped CPU costs** that prevent unsupported low-bit speedup claims.
5. **Provenance, recovery records, software tests and a connected dashboard** that make both the findings and their limitations inspectable.

The study did not solve audiovisual deepfake detection universally; it answered a narrower question about the precision behavior of one detector under one controlled protocol. [Detailed conclusion and future work](docs/08_FINAL_CONCLUSIONS_AND_FUTURE_WORK.md).

## 10. Using the system and reproducing the evidence

### 10.1 Install from a clean clone

```bash
git clone https://github.com/SudarshanG9/AFML-project.git
cd AFML-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pip install -r frontend/requirements.txt
```

The installed package and frontend allow you to explore source code, run synthetic correctness checks, and view saved results **without obtaining restricted media or checkpoints**. The extraction environment for raw video is a separate, heavier setup.

### 10.2 Start the research dashboard

```bash
cd frontend
PYTHONPATH=../src:. python -m streamlit run app.py --server.address 127.0.0.1
```

Visit the local URL printed by Streamlit (normally `http://localhost:8501`). The frontend's eight pages let you inspect original result tables and figures, imported artifacts, software status, validation predictions and run integrity. The application **does not load a pretrained model or rerun the accepted study on launch**.

Run the backend smoke check from the repository root:

```bash
python -m avfuse.member3_runtime.cli --help
python -m avfuse.member3_runtime.cli --smoke
```

`--smoke` uses a **toy diagnostic path**. Its scores and timings are **not scientific detector results**.

### 10.3 What real validation inference requires

Real model execution requires **all** of the following: (1) trusted external DiMoDif source at the documented revision, (2) matching LAV-DF checkpoint/config, (3) approved validation-only manifest, (4) complete compatible cached features, (5) explicitly chosen device and precision modes, and (6) permission to use the assets.

The historical `third_party/dimodif` gitlink lacks `.gitmodules`; **do not assume `git clone --recursive` installs the detector**. Follow the [reproducibility guide](docs/09_REPRODUCIBILITY_AND_USER_GUIDE.md) for the external source checkout, checkpoint SHA-256 checks, manifest identity and **validation-only** CLI examples. The accepted checkpoint identity is:

`e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e`

The intended DiMoDif NPZ cache layout is `feature_root/val/<clip_id>/features.npz`, with `video_features` and `audio_features` keys. Arbitrary video uploads or arbitrary embedding tensors are **not** supported as detector input. The Streamlit form invokes the actual Member 3 backend and enforces validation-only access and the LAV-DF checkpoint hash.

**Verification boundary:** Phase 5 verified interface wiring and synthetic tests, but **did not execute a new real-checkpoint inference through the frontend** with a complete authorized cache. The accepted held-out results come from the preserved original experiment, not a fabricated dashboard run. Never reopen the original test cohort merely to demonstrate UI functionality.

### 10.4 Tests and recorded software status

```bash
# From repository root
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:frontend \
  python -m pytest -q tests frontend/tests
```

The [Phase 5 report](docs/integration/PHASE_5_RELEASE_READINESS_REPORT.md) records **1,181 local tests passed, 22 skipped, 0 failures/errors** (19 historical private-fixture-dependent cases and three unavailable ffmpeg/ffprobe cases). The Python **3.10 and 3.12 GitHub CI jobs passed** on the integration commit; the merged `main` workflow also passed. Skipped real-fixture tests are not counted as demonstrated real-model execution.

[CI workflow](.github/workflows/ci.yml) · [Merged `main` workflow run](https://github.com/SudarshanG9/AFML-project/actions/runs/43721523552) · [Frontend setup](frontend/README.md) · [Phase 5 verification](docs/integration/PHASE_5_RELEASE_READINESS_REPORT.md).

## 11. Limitations and future directions

### Important limitations of these findings

- **Single detector / dataset / selected cohort:** Generalization to other checkpoints, manipulation generators, speakers or cross-dataset conditions was not measured.
- **Fake quantization, not actual INT8 deployment:** Tensor storage, integer accumulators, kernels, transfer costs and power efficiency were not established.
- **Cumulative and bundled changes:** All five layers are intervened on; V and output are restored together. No isolated single-layer or single-boundary causal claim follows.
- **Source groups are not speaker IDs:** Recording-level leakage audits do not establish speaker-level independence or exclude all training/pretraining exposure.
- **Development selection and historical access:** Upstream checkpoint selection exposed validation; historical test-derived summaries were seen before freeze. The accepted chain discloses this and did not select a new mode on final scores.
- **Statistical interpretation:** Small effects and zero-crossing CIs do not prove equivalence. Eighty nominal paired intervals were not multiplicity-adjusted.
- **Incomplete independent extraction audit:** Directory/metadata coverage reached 3,200 selected clips; the separately recorded complete-array audit stopped at 2,800 before an interruption. Later accepted inference is not retroactive byte-for-byte extraction verification.
- **Benchmark scope:** Resident cached-feature CPU timings do not measure end-to-end video preprocessing or prove equal restoration costs, memory savings or native speedup.
- **Application verification:** Automated tests and CI are green; complete human UI upload/visual checks and live checkpoint-backed Streamlit execution were not claimed in Phase 5.

### What we hope future work can achieve

The original objective—*protect forensic quality while reducing practical hardware cost*—remains worth testing with a **new, independently authorized experiment**:

1. Implement **native low-bit execution** with explicit activation/weight/calibration policies and evaluate genuine speed, energy and peak memory alongside detection quality.
2. Separate V from O, and test **individual layers/boundaries** while controlling upstream hidden activations.
3. Verify **cost fairness and timing uncertainty** before claiming quality recovered per unit of overhead.
4. Extend to **independent datasets, multiple checkpoints and verified speaker-disjoint sources**, avoiding unsupported transfer claims.
5. Pre-register a **practically important effect size, equivalence margin and multiplicity policy**; test silence, duration, codec and identity shortcuts more broadly.

These are **future research directions**, not capabilities or results already demonstrated by AVFuse-CausalQ.

## 12. References, datasets, and further reading

### Official datasets and model/software sources

- **LAV-DF** — [Official project, data access and license](https://github.com/ControlNet/LAV-DF) · [Hugging Face gated dataset](https://huggingface.co/datasets/ControlNet/LAV-DF). **This is the final evaluated dataset.** Access and redistribution are subject to provider terms (CC BY-NC 4.0 project license).
- **FakeAVCeleb** — [Official repository and access-request instructions](https://github.com/DASH-Lab/FakeAVCeleb). **Originally proposed; not used in the final evaluation.**
- **DiMoDif** — [Official source code and model instructions](https://github.com/mever-team/dimodif). **The actual pretrained detection model used.**
- **auto-AVSR** — [Visual Speech Recognition for Multiple Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages). Upstream audio and visual encoder family used for feature extraction; LRS3-related model assets are not an independently evaluated AVFuse dataset.

### Primary research papers

1. **Koutlis & Papadopoulos.** [DiMoDif: Discourse Modality-information Differentiation for Audio-visual Deepfake Detection and Localization](https://arxiv.org/abs/2411.10193) — pretrained model and joint processing adopted here.
2. **Cai et al.** [Glitch in the Matrix: A Large Scale Benchmark for Content Driven Audio-Visual Forgery Detection and Localization](https://arxiv.org/abs/2305.01979); [original dataset/method paper](https://arxiv.org/abs/2204.06228) — LAV-DF and its modality-manipulation setting.
3. **Oorloff et al.** [AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection](https://arxiv.org/abs/2406.02951) — original architectural inspiration, **not reproduced**.
4. **Khalid et al.** [FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset](https://arxiv.org/abs/2108.05080) — originally considered dataset; **not evaluated**.
5. **Dao et al.** [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — IO-aware attention motivation and prior art.
6. **Zhang et al.** [SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration](https://arxiv.org/abs/2410.02367) — true low-bit attention context; **not implemented here**.
7. **Oh, Kwon & Lee.** [QFlash: Bridging Quantization and Memory Efficiency in Vision Transformer Attention](https://ijcai.org/proceedings/2026/167) · [code](https://github.com/EfficientCompLab/qflash) — quantized fused-attention prior art; **not reproduced**.
8. **Li et al.** [DeFakeQ: Enabling Real-Time Deepfake Detection on Edge Devices via Adaptive Bidirectional Quantization](https://arxiv.org/abs/2604.08847) — selective precision as established adjacent prior art.
9. **Cai et al.** [Explicit Correlation Learning for Generalizable Cross-Modal Deepfake Detection](https://arxiv.org/abs/2404.19171) — content-level AV correspondence beyond simple lip synchronization.
10. **Smeu et al.** [Circumventing Shortcuts in Audio-visual Deepfake Detection Datasets with Unsupervised Learning](https://arxiv.org/abs/2412.00175) — shortcut validity motivation.
11. **Jain & Wallace.** [Attention is not Explanation](https://aclanthology.org/N19-1357/) — why attention-map similarity is not a substitute for label-grounded evidence.
12. **Vaswani et al.** [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — foundational transformer-attention formulation.

A broader, qualified bibliography—including AuthentiQ, AV-DeepFake-Net, AuViRe, Sensors-driven fusion, DeepFake Doctor and lightweight quantized detectors—is in [Chapter 10](docs/10_EVIDENCE_AND_REFERENCES.md). Referenced authors' published results are **not** measurements produced by this repository.

### Full research-documentation index

| Chapter | Read this for... |
| --- | --- |
| [1 — Research foundations](docs/01_RESEARCH_FOUNDATIONS.md) | Problem, concepts, formal hypotheses H1/H2/H3. |
| [2 — Literature and novelty](docs/02_LITERATURE_REVIEW_AND_NOVELTY.md) | Prior art, candidate models and carefully bounded contribution. |
| [3 — Research decisions and pivots](docs/03_RESEARCH_DECISIONS_AND_PIVOTS.md) | Model/dataset changes, validation freeze and execution recovery. |
| [4 — Final system architecture](docs/04_FINAL_SYSTEM_ARCHITECTURE.md) | Exact attention tensors, formulae, masks, interfaces and data flow. |
| [5 — Dataset and methodology](docs/05_DATASET_AND_EXPERIMENTAL_METHODOLOGY.md) | Whole-video cohort, extraction, thresholds, bootstrap and benchmarking. |
| [6 — Experiments and results](docs/06_EXPERIMENTS_AND_RESULTS.md) | Complete validation/test comparisons, confidence intervals, subtype effects and timing. |
| [7 — Failures and limitations](docs/07_FAILURES_LIMITATIONS_AND_HYPOTHESES.md) | Why findings are qualified and how each hypothesis was assessed. |
| [8 — Conclusions and future work](docs/08_FINAL_CONCLUSIONS_AND_FUTURE_WORK.md) | Direct scientific answer and next experiments. |
| [9 — Reproducibility and user guide](docs/09_REPRODUCIBILITY_AND_USER_GUIDE.md) | Setup, authorized external assets, CLI, schema and error handling. |
| [10 — Evidence and references](docs/10_EVIDENCE_AND_REFERENCES.md) | Provenance hashes, full bibliography and source verification limits. |

**Research transparency:** Original scientific evidence, its hashes, frozen policies, recovery records and failure histories have been preserved. The code and dashboard help inspect the completed work; they do not turn an unsupported hypothesis into a positive result.
