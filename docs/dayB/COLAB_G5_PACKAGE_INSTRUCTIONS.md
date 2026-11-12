# G5 CPU execution package — manual Colab route

**Approved device: existing Colab CPU runtime.** Expected Torch is 2.14.1+cpu from the accepted G4 evidence. The agent has not accessed Colab, mounted Drive, loaded cache arrays or executed G5. All commands below are for the user to execute manually. The helpers never mount Drive, authenticate, access accounts, upload anything or launch G6.

## Approved scope and inputs

G5 only: all 1,200 reference-validation predictions, score orientation, provisional reference-validation 5% real-FPR calibration, baseline grouped-bootstrap metrics (1,000 resamples, seed 0), and reference cached-model timings. The dataset, labels, model and precision semantics remain unchanged. No inference over held-out test features; canonical test metadata is used only to audit splits.

Approved benchmark selection from G4 metadata, without using scores:

| Role | Validation ID | Valid tokens |
| --- | --- | ---: |
| short | 6d8daea9cc33fe63 | 78 |
| typical cost anchor | 7cea6ad75b4ebf41 | 192 |
| long | 891214bf9e5f17d0 | 493 |

All later precision runs and matched cost comparisons must retain CPU and the exact typical anchor. Model attention is padded to 600 positions per modality; these unpadded lengths characterize input/preparation variation. Their timing results must remain separate workloads.

## 1. Upload and extract locally in Colab

Upload the delivered G5 ZIP using Colab's Files sidebar. Do not upload the feature cache. Extract this archive to a fresh directory (substitute only the ZIP filename printed in the final delivery):

```python
from pathlib import Path
import zipfile
archive = Path(input('Absolute path to the uploaded G5 code ZIP: ').strip())
assert archive.is_file()
with zipfile.ZipFile(archive) as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    top = {Path(n).parts[0] for n in z.namelist()}
    assert len(top) == 1
    package_root = Path('/content') / next(iter(top))
    assert not package_root.exists(), 'Use a fresh extraction; preserve earlier files'
    z.extractall('/content')
import os
os.chdir(package_root)
print(package_root)
```

```python
!python -B g5/verify_package.py
!python -m pip install -r requirements-g5-cpu.txt
```

Confirm package verification and dependency installation succeed. Do not install, upgrade or downgrade Torch. The CPU environment check will stop if it differs from the accepted G4 Torch version. Dependencies are pinned to NumPy 2.1.3, jsonschema 4.23.0, scikit-learn 1.6.1, matplotlib 3.10.0 and pytest 8.3.4. No editable project installation is needed: helpers supply the bundled source paths explicitly. Model/environment reproducibility is recorded in the run context; these local pins alone are not a remote model parity pass.

## 2. Existing mounted cache and checkpoint

Keep or manually restore your own approved Drive mount. No mounting code is supplied. The included, unchanged G4 ready manifest retains the previously verified absolute paths. All 1,200 val paths must exist; the helper refuses to rewrite them if the mount changed. If unavailable, stop and report the changed mount rather than substituting a pilot or regenerated manifest.

Required checkpoint weights, NOT included in the ZIP:

`dfd_lavdf_reduceonplateau_64_4_1_5_True.pth`, 4,566,082 bytes.

Required SHA-256: `e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e`.

The local original is `/Users/sudarshan/Documents/GitHub/AFML-project/third_party/dimodif/ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.pth`. Reuse its already-uploaded approved Colab copy if available, otherwise upload it manually via the Files sidebar. The matching config JSON, original models.py and license are included. Preparation copies these three small detector assets into a new runtime folder INSIDE COLAB; it copies no feature cache and downloads nothing to the Mac.

```python
import subprocess, sys
from pathlib import Path
checkpoint = Path(input('Absolute path to the approved .pth in Colab: ').strip())
assert checkpoint.is_absolute() and checkpoint.is_file()
subprocess.run([sys.executable, '-B', 'g5/prepare_runtime.py',
                '--checkpoint', str(checkpoint),
                '--output-root', '/content/g5_results'], check=True)
```

Copy the printed G5_RUN_DIR value into RUN below. Keep that SAME run directory for later stages. Preparation validates the exact G4 evidence/manifests, existence and val/ confinement of all 1,200 paths, and approved benchmark input hashes. It records the CPU environment and package identity. It performs no inference.

## 3. New-runtime reference parity — execute FIRST, then STOP

```python
RUN = Path(input('Paste the printed absolute G5_RUN_DIR: ').strip())
assert RUN.is_absolute() and RUN.is_dir()
subprocess.run([sys.executable, '-B', 'g5/check_parity.py', '--run-dir', str(RUN)], check=True)
```

This runs only the approved typical validation clip through native and reference-instrumented CPU inference. It records all five per-layer errors, 20 unchanged precision-boundary records, logits/scores, checkpoint SHA and unchanged frozen weights. Existing tolerances remain logits/layers atol 1e-5/rtol 1e-4 and scores atol 1e-6/rtol 1e-5. No lowp/restoration/query inference or full-cohort inference runs in this step.

**STOP HERE. Return `G5_review_runtime_parity.zip` for review before running Step 4.** A passing program must not be interpreted as automatic human approval. The ZIP excludes runtime checkpoint weights and features. If anything fails, return the full cell traceback and available run files; preserve them, do not loosen tolerances or change the device/model.

---

## 4. Full reference prediction integration — ONLY after parity review authorizes it

```python
subprocess.run([sys.executable, '-B', 'g5/full_reference.py', '--run-dir', str(RUN)], check=True)
```

This calls the existing dayB_inference.py with --mode reference --device cpu over the exact G4 ready manifest, then validates exactly 1,200 IDs/rows, six-field schema, approved checkpoint, val split and finite probabilities. It runs the existing m2_check_predictions.py against manifest_whole.jsonl. Outputs and stdout/stderr are retained under RUN. No score flipping, exclusion or cohort reduction is allowed.

**STOP. Return `G5_review_reference_orientation.zip` and the orientation result for human review before calibration.** If the checker fails/INVERTED, return its report/logs and do not continue. If unclear, return it for interpretation; the metrics helper will not proceed automatically. Neither a pilot nor this scalar range proves forensic accuracy.

## 5. Provisional calibration and baseline metrics — ONLY after orientation review authorizes it

```python
subprocess.run([sys.executable, '-B', 'g5/baseline_metrics.py', '--run-dir', str(RUN)], check=True)
```

This uses the existing Member 2 tools unchanged. Calibration uses reference VALIDATION at target real FPR 0.05. Evaluation uses the same canonical manifest, fixed provisional threshold, --split val --n-boot 1000 --seed 0. It validates total counts, per-type real-versus-fake metrics, hashes and finite CIs. If the threshold lies outside [0,1] or an interval is undefined/NaN, stop and return the existing outputs without changing the protocol.

The calibration file is provisional Day B validation evidence, NOT a Day 9 final freeze. Preserve `reports/dayB_validation_calibration.json` and its SHA for G6/G8; never recalibrate per mode. Plots and summaries are retained. Return `G5_review_baseline_metrics.zip` for interpretation. No lowp or restored predictions are generated.

## 6. Reference CPU benchmark — approved inputs only

After the preceding reviews permit continuation within G5:

```python
subprocess.run([sys.executable, '-B', 'g5/reference_benchmarks.py', '--run-dir', str(RUN)], check=True)
```

Runs the new integration using the existing benchmark engine and v0.2 validated record writer, only --modes reference --device cpu --warmup 10 --repeats 50, separately for each approved input. The typical anchor is `benchmarks/dayB_model_reference.jsonl`; short/long records have distinct filenames. Private workload metadata records clip IDs, token lengths and source hashes separately. Hashing/loading are outside timing; model preparation/padding, attention and head are inside. No disk I/O, decode or host/device copy is included in the timed model call. CPU peak memory is null/not_measured; no GPU-memory or INT8-speed claim is made.

**STOP AT G5. Return `G5_review_reference_benchmarks.zip` plus preceding evidence.** This final review ZIP includes prior reports/predictions/logs, excluding runtime assets, other ZIPs and all features. It is created locally in Colab, not uploaded automatically. Do not run G6 or any low-precision full-validation command.

## Execution and preservation details

Existing command scripts remain unchanged; each subprocess argv, exit code, stdout and stderr is saved. New outputs use an exclusive, timestamped run directory. Existing command logs/outputs are never silently overwritten. A failed stage must be reviewed before retrying; use a new run for a complete rerun. If a Python/helper assertion fails before a child command log is written, return the notebook traceback and existing execution_context.json. The agent will not mount Drive or execute any of these stages remotely.

G1 provenance/source-proxy/dev-selection limitations persist. The optional restore_query pilot was not generated by this agent; a local file has since appeared but has not been reviewed, and no Day 8 claim is made. The package does not execute it.
