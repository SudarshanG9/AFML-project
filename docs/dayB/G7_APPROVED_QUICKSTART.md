# Approved G7 Colab cell

Upload `AFML_G7_CPU_Addon_APPROVED.zip` manually to the existing Colab CPU runtime, then paste this exact cell. No dependency installation or new checkpoint upload is needed if the accepted runtime is intact. Download the printed `G7_review.zip` and return it for review. Stop at G7.

```python
from pathlib import Path
import hashlib, zipfile, subprocess, sys

archive = Path('/content/AFML_G7_CPU_Addon_APPROVED.zip')
addon = Path('/content/AFML_G7_CPU_Addon_APPROVED')
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '81e645efc0888b37ab25963379f11b3fc6015b15051e720ac117197e01b4d5bf'
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
