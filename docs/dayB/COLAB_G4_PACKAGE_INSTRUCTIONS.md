# User-executed G4 only

This package was prepared locally; G4 has NOT been executed. No Drive access, mounting, account connection or upload was performed by the agent. The user mounts Drive manually and supplies the verified cache root. No G5 or inference command follows this workflow.

## Contents and scope

Current project src/avfuse, scripts, Python tests, JSON contracts/configs and original pyproject.toml; relevant protocol/contract docs; both unchanged approved metadata manifests; original DiMoDif models.py, license and matching detector configuration JSON. Full-manifest test metadata is included only for source-group split auditing. No features, raw media, predictions, .git, virtualenv, credentials, datasets or checkpoint weights are bundled. The current source snapshot is 92337cef4cda8a94b45068c4bf2ccfc036cbb931; per-file hashes identify the actual bytes.

## Required separate upload

Upload **dfd_lavdf_reduceonplateau_64_4_1_5_True.pth** separately from:

`/Users/sudarshan/Documents/GitHub/AFML-project/third_party/dimodif/ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.pth`

Size: 4,566,082 bytes. Required SHA-256:

`e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e`

The corresponding `dfd_lavdf_reduceonplateau_64_4_1_5_True.json` and original `third_party/dimodif/src/models.py` are already included. No separate upload of those is needed. G4 hashes the weights; it does not deserialize them or run the detector. The config/model source are included to account for the original detector dependencies; the complete detector/GPU runtime will be reviewed separately before G5.

## 1. Upload and extract the code ZIP manually

Use Colab's Files sidebar to upload `AFML_G4_Colab_20261111T153141Z.zip`. Run:

```python
from pathlib import Path
import zipfile
archive = Path('/content/AFML_G4_Colab_20261111T153141Z.zip')
assert archive.is_file(), 'Upload the code ZIP using the Files sidebar first'
with zipfile.ZipFile(archive) as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    destination = Path('/content/AFML_G4_Colab_20261111T153141Z')
    assert not destination.exists(), 'Use a fresh extraction directory; preserve earlier work'
    z.extractall('/content')
```

```python
%cd /content/AFML_G4_Colab_20261111T153141Z
!python -B g4/verify_package.py
```

Expect PACKAGE INTEGRITY PASS. This checks packaged file hashes only, without feature access.

## 2. Install the G4 audit dependencies

```python
!python -m pip install -r requirements-g4.txt
```

Check installation completes successfully before proceeding. The G4 pins are NumPy 2.1.3 and jsonschema 4.23.0, matching the locally tested preflight-unit environment. The original project metadata is preserved for provenance. G4 does not need an editable project install: the wrapper and its child process explicitly use the bundled src/ tree. This avoids installing unused model/evaluation dependencies or generating files inside the verified package. Torch and scikit-learn are not needed to execute this preflight, and the existing Colab Torch build is not changed. These pins do not establish a final G5 model/GPU environment.

## 3. Mount Drive manually and identify the cache root

Use your own manual Drive-mount flow. This package contains no Drive login, mount, account or remote-download commands. Identify the actual existing root with this layout:

```
<verified-cache-root>/val/<clip_id>/features.npz
<verified-cache-root>/val/<clip_id>/meta.json
```

There is deliberately no assumed cache-root value below. Upload the approved .pth separately using the Files sidebar, or identify its already-existing approved location. Do not upload/download the cache to the Mac. Do not open the sealed test subtree.

## 4. Execute G4 against your supplied paths

Run this Python cell only after your manual mount and path verification:

```python
from pathlib import Path
import subprocess, sys
cache_text = input('Paste the VERIFIED absolute cache root containing val/: ').strip()
checkpoint_text = input('Paste the absolute path to the approved uploaded .pth: ').strip()
assert cache_text and checkpoint_text
cache_root = Path(cache_text).expanduser()
checkpoint = Path(checkpoint_text).expanduser()
assert cache_root.is_absolute() and checkpoint.is_absolute()
assert (cache_root / 'val').is_dir() and checkpoint.is_file()
subprocess.run([
    sys.executable, '-B', 'g4/run_g4.py',
    '--cache-root', str(cache_root),
    '--checkpoint', str(checkpoint),
    '--output-root', '/content/g4_results',
], check=True)
```

The wrapper runs the **unchanged** scripts/day6_preflight.py with --expected-count 1200 and --expected-per-type 300. It checks the exact source/checkpoint hashes, preserves metadata, confines resolved feature paths to val/, and writes to a new unique /content/g4_results/G4_<UTC>_<id>/ directory. It never walks test/. It runs CPU NumPy integrity checks, not a GPU benchmark or inference. No count override is exposed by the wrapper.

Output: preflight_report.json, dayB_val_ready.jsonl (only generated by a successful existing preflight), command.json, preflight.log, execution_environment.json, verification_summary.json and output_hashes.json. On failure, retain any failure report/logs; do not lower counts, exclude clips, fabricate a manifest or delete evidence. Failure before the wrapper's output-directory creation appears in the cell traceback; return that traceback with the supplied paths.

The direct existing-preflight command executed by the wrapper is recorded exactly in command.json, using your supplied paths and new output directory. No subprocess launches G5.

## 5. Return evidence and stop

Successful acceptance requires PASS, 1200/1200 verified, 300 per category, zero errors and the approved SHA. The wrapper verifies these conditions plus ready-manifest ID/metadata identity. Download the small generated G4 output directory through Colab's Files sidebar and return its files for review. Do not include feature tensors. The ready manifest's absolute paths apply only to that mounted environment; do not expect them to resolve on the Mac.

**STOP AT G4.** Do not run inference, calibration, benchmarks or G5. Await explicit user G4 acknowledgement after review. The optional restore_query G3 pilot remains not generated; this package does not run it.
