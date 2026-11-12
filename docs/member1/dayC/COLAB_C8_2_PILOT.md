# C8.2 pilot — existing approved Colab CPU runtime

The user explicitly approved: **Approve existing Colab CPU; prepare the local pilot package**. This is a narrow exception to the earlier project-local runtime restriction for the recorded existing CPU/Drive arrangement. Agent-generated source/configuration/tests/packages remain in the local clone. The agent has not accessed Colab/Drive or run a real pilot. The user runs this package manually.

## Run once

1. Keep the existing accepted **CPU** Colab runtime and previously authorized Drive mount. Do not reset it, install packages, upload new model weights or rewrite manifests. The original G5 package/run and recorded cache paths must still exist.
2. Manually upload `AFML_DayC_C8_2_CPU_Pilot.zip` from the clone's `artifacts/dayC/member1/colab/` folder into `/content` using Colab's Files sidebar.
3. Paste the cell below. It verifies the exact archive SHA, extracts into a new directory, and runs only the C8.2 pilot. No service account, mount, download, Git or installation command is included.
4. Return the printed **C8_2_pilot_review.zip** for audit, including if the pilot fails. If failure occurs before a bundle can be created, return the traceback. Preserve the runtime/evidence; do not rerun, modify guards or change paths to force success.

```python
from pathlib import Path
import hashlib, os, stat, subprocess, sys, zipfile

archive = Path('/content/AFML_DayC_C8_2_CPU_Pilot.zip')
addon = Path('/content/AFML_DayC_C8_2_CPU_Pilot')
assert archive.is_file(), 'Upload the pilot ZIP manually first'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '184cfe198d62e1f5e68d7ba2e16431f67fd897625175b22fb75646582a60b8f8', 'Wrong or changed ZIP'
assert not addon.exists(), 'Preserve the existing extraction and evidence; do not rerun'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    assert {Path(n).parts[0] for n in z.namelist()} == {addon.name}
    assert all(not stat.S_ISLNK(info.external_attr >> 16) for info in z.infolist())
    z.extractall('/content')
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
           PYTHONPATH=str(addon / 'src') + os.pathsep + str(addon),
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
           NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg', GIT_OPTIONAL_LOCKS='0')
subprocess.run([sys.executable, '-B',
    str(addon / 'scripts/dayC_member1/colab_query_pilot.py')],
    cwd=addon, env=env, check=True)
```

## Exact package identity

- Archive: `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayC/member1/colab/AFML_DayC_C8_2_CPU_Pilot.zip`
- SHA-256: `184cfe198d62e1f5e68d7ba2e16431f67fd897625175b22fb75646582a60b8f8`
- Bytes: 125715; files: 53.
- Original frozen G5 source/schema files: **41**, each compared byte-for-byte to the approved package manifest. Original attention, precision, intervention and adapter math are unchanged.
- Package manifest SHA-256: `5763fb138c14d0067096af35a09aa359f64d1a324bfbf96358d11e6a4d317b27`.
- Includes new diagnostic helpers/runner, pinned release/context and actual user C0/runtime approvals. **No weights, features, media or dataset rows** are included.

## What the pilot checks and executes

Before loading a feature: exact Python/platform and six package versions; CPU with one Torch and one interop thread; package integrity; original G5 context; checkpoint/model-config/model-source hashes; canonical-manifest hash only; validation/ready-manifest hashes and full 1,200 matching IDs with 300 per category; all 1,200 recorded feature and metadata paths; fixed typical clip feature/metadata hashes. It never reads canonical test rows or enters a held-out cache directory.

The fixed pilot is **7cea6ad75b4ebf41**, the approved typical Day B anchor. The checkpoint is `e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e`. The cache root remains exactly the recorded Drive shortcut root. No path rebasing or replacement manifest is created.

The runner constructs one frozen adapter and loads that one validation feature once. It performs one native reference call, one instrumented reference call for parity, and one call each for low_precision, restore_value_output and restore_query. It checks 20 trace rows per diagnostic mode across all five layers, finite original fake probabilities, tensor/model integrity, original method restoration and final input hashes. The parity helper exports scores already computed by those reference calls; no extra reference call is added just to print a score.

It records all 1,200 cache files' presence, but **does not claim to have rehashed the entire cache**. The pilot feature/metadata are rehashed against accepted evidence. Model invariants, trace summaries and source-only independent oracles have distinct evidential scopes.

A new exclusive pilot lock prevents a second actual pilot attempt using the same extracted package. Failures preserve evidence; there is no automatic retry. All new runtime outputs belong under the newly extracted package's `artifacts/dayC/member1/day8/` directory. Existing G5 code, checkpoint, manifests, caches and Day B predictions remain read-only.

## Runtime and stop conditions

Accepted environment: Python 3.13.18, Linux-6.6.159+-x86_64-with-glibc2.39, PyTorch 2.14.1+cpu, NumPy 2.1.3, jsonschema 4.23.0, scikit-learn 1.6.1, matplotlib 3.10.0, pytest 8.3.4, CUDA unavailable. The fresh subprocess explicitly reproduces the approved one-thread settings. Any other mismatch stops; there is no installation or runtime-override flag.

**STOP after the pilot and return its review ZIP.** A passing pilot is not Day 8 completion, evidence of improved accuracy, equal cost or H3 support. C8A review is still required before the full 1,200-clip query run. No full-validation scores, benchmarks, Day 9 freeze, test labels or held-out inference are produced by this package.

## Local validation

**189 Member 1 tests passed, 0 failed, 0 skipped.** Tests cover unchanged numerical oracles, synthetic five-layer traces, package integrity/tampering, authorization, asset exclusion, cohort downscaling/duplicates, wrong split, cache escape/missing files, environment mismatch and exclusive destinations. A local Mac smoke test of the extracted package stopped during read-only preflight and generated failure evidence without constructing a model, reading a feature or creating a pilot attempt lock. That smoke test is not a research pilot.

Evidence: `artifacts/dayC/member1/colab/PREP_20261112T092520Z_8fb36d/member1.xml`, `tests.json`, `package_build.json` and `PACKAGE_DELIVERY.json`.
