# One-shot G8/G9 Colab execution

G7 passed and your acknowledgement is recorded. Upload `AFML_DayB_G8_G9_CPU.zip` manually into the **existing approved CPU Colab runtime**, then paste this one cell. No intermediate upload is required; return only the printed `DayB_G8_G9_review.zip` when finished. The cache remains remote.

```python
from pathlib import Path
import hashlib, zipfile, subprocess, sys

archive = Path('/content/AFML_DayB_G8_G9_CPU.zip')
addon = Path('/content/AFML_DayB_G8_G9_CPU')
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '9909583f48a94305608a69944503c1f00fc0f0bc2136be4255b0b2f842b9068b'
assert not addon.exists(), 'Preserve the existing extraction'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    assert {Path(n).parts[0] for n in z.namelist()} == {addon.name}
    z.extractall('/content')
subprocess.run([
    sys.executable, '-B', str(addon / 'scripts/dayB_finish/run_g8_g9.py'),
    '--g5-root', '/content/AFML_G5_CPU_Colab_20261111T180139Z',
    '--g5-run', '/content/g5_results/G5_CPU_20261111T181935Z_422cf72f',
    '--g6-run', '/content/g6_results/G6_CPU_20261111T194903Z_6a2bf3c1',
    '--g7-run', '/content/g7_results/G7_CPU_20261111T202044Z_ce6a0493',
], check=True)
```

This runs G8 and assembles G9 evidence/handoffs. It does not execute Day 8 query control, a policy freeze or held-out testing. The agent will audit the single returned ZIP before claiming Day B complete. On a computational failure, return the printed failure bundle and traceback; do not bypass the guard.
