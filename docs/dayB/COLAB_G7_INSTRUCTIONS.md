# Approved G7 CPU execution — stop after G7

The user approved the separate single-clip, one-target-at-a-time diagnostic: use validation clip `7cea6ad75b4ebf41`, restore V/output at one target layer, and keep every other layer, including the suffix, low precision. This supplements the main cumulative five-layer audit; it does not replace any main configuration.

Upload **AFML_G7_CPU_Addon_APPROVED.zip** manually using Colab's Files sidebar. Keep the accepted existing CPU runtime and existing G5/G6 files. No installation, Drive mounting, checkpoint upload or cache copy is needed when that runtime is still available.

Required versions remain Python 3.13.18; Torch 2.14.1+cpu; NumPy 2.1.3; jsonschema 4.23.0; scikit-learn 1.6.1; matplotlib 3.10.0; pytest 8.3.4. Keep one Torch thread and one interop thread. If the accepted runtime, original cache mount, G5 code/checkpoint or G5/G6 results are missing or changed, stop and report it. Do not reinstall, silently recreate results, rewrite the ready manifest or change the scientific protocol.

Run this cell after upload. Replace the checksum placeholder with the SHA-256 in G7_APPROVED_PACKAGE_DELIVERY.json or the final message.

```python
from pathlib import Path
import hashlib, zipfile, subprocess, sys

archive = Path('/content/AFML_G7_CPU_Addon_APPROVED.zip')
addon = Path('/content/AFML_G7_CPU_Addon_APPROVED')
expected_sha256 = 'PASTE_DELIVERED_ARCHIVE_SHA256'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected_sha256
assert not addon.exists(), 'Preserve the existing extraction'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    assert {Path(n).parts[0] for n in z.namelist()} == {addon.name}
    z.extractall('/content')

subprocess.run([
    sys.executable, '-B', str(addon / 'scripts/dayB_g7/run_g7.py'),
    '--g5-root', '/content/AFML_G5_CPU_Colab_20261111T180139Z',
    '--g5-run', '/content/g5_results/G5_CPU_20261111T181935Z_422cf72f',
    '--g6-run', '/content/g6_results/G6_CPU_20261111T194903Z_6a2bf3c1',
    '--matched-scope', 'one-target-lowp-prefix-and-suffix',
], check=True)
```

The helper validates its manifest, accepted package/runtime context and assets, original reference/calibration bytes, G6 evidence hashes, accepted G6 lowp prediction hash, and selected validation feature hash. It then runs only:

1. Standard cumulative whole-model upstream audit for lowp versus five-layer V/output restoration.
2. Original full-model pilot restoration trace: all five layers, Q/K changed and V/attention_output unchanged at the defined boundary.
3. Separate matched-upstream diagnostic: one all-lowp baseline and five one-target-restored arms, including per-target upstream equality, projected/quantized QK equality, target-output difference after FP32 output projection, downstream score change, unchanged weights and method/hook cleanup. The lowp baseline must match both the original adapter and the accepted G6 score for this clip.

The code prints **G7_RUN_DIR=...** and **REVIEW ZIP: .../G7_review.zip**. Download that printed ZIP manually from Colab's Files sidebar and return it for review. It contains upstream tables, trace, separate diagnostic, context, command logs, output hashes and status. It contains no weights, feature cache, raw tensors or dataset.

No full 1,200-clip restored inference, G8, query inference, held-out test access, training or threshold recalibration is run. A diagnostic score change on one clip is not forensic accuracy evidence. G6's absence of established positive H2 damage remains part of interpretation.

On failure, preserve the notebook traceback and review ZIP. After run creation the helper still bundles failure evidence; before that, return the traceback. Do not bypass guards, overwrite previous results or silently change the runtime/model. A rerun creates a fresh output directory. Preserve an already extracted package rather than overwriting it.

**STOP AT G7.** Returned artifacts require G7 PASS/FAIL review and explicit ACK G7 before G8. Scientific wording and limitations are in G7_RESEARCH_REVIEW.md.
