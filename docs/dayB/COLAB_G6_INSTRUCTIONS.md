# G6 CPU add-on — user-run, stop at G6

G5 has been acknowledged. This add-on runs G6 only. Keep the existing approved Colab CPU runtime, G5 code package, checkpoint, G4 ready manifest, mounted validation cache and G5 results. Do not reinstall or upgrade packages. No agent login, Drive mount, account operation or remote execution is involved. No new cache upload/download is needed.

## Existing dependencies

- G5 code root: `/content/AFML_G5_CPU_Colab_20261111T180139Z`
- G5 results: `/content/g5_results/G5_CPU_20261111T181935Z_422cf72f`
- Original checkpoint remains in that run's `runtime/dimodif/ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.pth`, SHA-256 `e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e`.
- No installation command is needed in the continuing runtime. Required versions remain Python 3.13.18; Torch 2.14.1+cpu; NumPy 2.1.3; jsonschema 4.23.0; scikit-learn 1.6.1; matplotlib 3.10.0; pytest 8.3.4. Preserve one Torch thread and one interop thread.
- The exact existing G4 feature paths must still resolve in Colab. The helper verifies those paths rather than inventing a new cache root or rewriting the manifest.

If that runtime, mount or run directory no longer exists, stop and return that information. Do not substitute a different device/runtime or rerun/recalibrate G5 silently.

## 1. Upload the add-on ZIP and extract it

Upload `AFML_G6_CPU_Addon.zip` manually using Colab's Files sidebar, then run:

```python
from pathlib import Path
import zipfile
import subprocess
import sys

archive = Path('/content/AFML_G6_CPU_Addon.zip')
ADDON = Path('/content/AFML_G6_CPU_Addon')
assert archive.is_file()
assert not ADDON.exists(), 'Preserve an existing extraction; do not overwrite it'
with zipfile.ZipFile(archive) as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    assert {Path(n).parts[0] for n in z.namelist()} == {'AFML_G6_CPU_Addon'}
    z.extractall('/content')
print(ADDON)
```

## 2. Run G6 only

```python
G5_ROOT = Path('/content/AFML_G5_CPU_Colab_20261111T180139Z')
G5_RUN = Path('/content/g5_results/G5_CPU_20261111T181935Z_422cf72f')
assert G5_ROOT.is_dir() and G5_RUN.is_dir()
subprocess.run([
    sys.executable, '-B', str(ADDON / 'g6/run_g6.py'),
    '--g5-root', str(G5_ROOT), '--g5-run', str(G5_RUN),
    '--output-root', '/content/g6_results',
], check=True)
```

The helper prints `G6_RUN_DIR=...`, followed by `REVIEW ZIP: .../G6_review.zip`. It creates a fresh run and refuses to overwrite earlier outputs. It verifies both package manifests, the accepted CPU environment and runtime assets, G4 provenance, full validation membership, approved benchmark hashes, exact reference prediction hash, and exact provisional calibration hash before proceeding.

It executes the original, unchanged inference/evaluation/benchmark scripts from the accepted G5 code package:

1. All 1,200 low-precision predictions. The original adapter checks logits and probabilities for finiteness on every call. No clip skipping, subsampling, score inversion or threshold refitting.
2. Reference/lowp prediction checks and orientation. An inverted or unclear orientation stops execution for human interpretation.
3. Overall/per-type evaluation and paired BCE damage at the exact G5 threshold, with 1,000 source-group bootstrap resamples and seed 0. An additional report uses the same existing bootstrap engine for per-type paired AUROC/AUPRC/BCE/recall/FPR differences.
4. Fresh reference+lowp cached-model benchmarks for the three previously approved inputs, CPU, 10 warmups / 50 repeats. The typical clip remains the matched cost anchor. Fake quant/dequant stays inside the timed callable.
5. Isolated attention reference+lowp measurements on the original Day A Q/K/V tensors with SDPA and dense-oracle backends, 10/50. The included capture is 6.4 MiB, from the existing pilot; it is not the validation feature cache. These results mean **simulated INT8, no integer kernels**. CPU peak memory is unmeasured/null.

The existing Day A capture and provenance are included byte-for-byte. It was captured in an earlier environment; G6 operator timing executes on the approved current CPU runtime. Model and operator timing scopes remain separate.

On failure, preserve the traceback. After run creation the helper still prints a failure-review ZIP location; return that ZIP and the notebook traceback. Failures before run creation require the notebook traceback. Do not delete evidence or silently retry with changed settings. A subsequent explicit rerun creates a new run.

## 3. Download the printed ZIP manually and return it

Use Colab's Files sidebar to download the exact printed `G6_review.zip`. It contains reference/lowp predictions, the unchanged calibration, full metrics/plots, paired reports, benchmark records, logs, runtime provenance, status and output hashes. It excludes the checkpoint, feature cache and Q/K/V tensors.

**STOP AT G6.** Return the ZIP for the H2 interpretation and G6 acknowledgement. No G7/G8 helper or restored/query/test inference is included. A negative or absent quantization effect remains a valid finding; the helper does not authorize a positive restoration narrative.

The research limitations remain open: validation-based checkpoint selection, source groups without speaker identities, training overlap not independently verified, and outstanding shortcut risks including trailing silence. G7 additionally requires explicit review of cumulative five-layer restoration versus separately matched-upstream causal diagnostics. G9 requires a freeze/handoff review; running test is not authorized here.
