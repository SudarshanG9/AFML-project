# B1 / C8.2 returned evidence: missing original G5 context

The returned ZIP SHA-256 is `0240c2a3959711fa30c9489bffef6f4974dab37af98ffd3c95d5c8babb48293b`. Its two recorded JSON hashes match and the approved version-drift comparison passes. The reported failure is `FileNotFoundError` for:

`/content/g5_results/G5_CPU_20261111T181935Z_422cf72f/execution_context.json`

Expected SHA-256: `c03488cb787b1230bd86274dc7a36d39983574c7e2bc996c931dcfc8a835bd7c`.

The reported stage is `read_only_preflight`; `model_constructed=false` and `feature_contents_read=false`. There is no `preflight.json`, pilot trace, reference consistency result or completed pilot in the ZIP. The ZIP does not record the current run ID or executed package/source hash, so no full executed-code provenance is inferred from this artifact alone. No C8A request is ready.

## One read-only diagnostic action

Run this cell in the same Colab session and return the printed output. It checks the entire recorded input set before another attempt, without loading feature tensors or opening any test rows. No replacement context or manifest is created and no data is moved. The packaged original-context copy is identified only as a reference; it is not silently substituted for a missing original run.

```python
from pathlib import Path
import hashlib, json

addon = Path('/content/AFML_DayC_C8_2_CPU_Pilot_Drift_v2')
c = json.loads((addon/'configs/dayC/member1/colab_pilot_release.json').read_text())
def sha(path):
    d = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            d.update(chunk)
    return d.hexdigest()

# Existing files only: no installs, copies, manifests, mounts or inference.
checks = [(key, Path(c[key]), c[key+'_sha256']) for key in
          ('g5_context','canonical_manifest','validation_manifest','ready_manifest','g4_report')]
checks += [('runtime_asset', Path(item['path']), item['sha256']) for item in c['runtime_assets']]
for label, path, expected in checks:
    if not path.is_file():
        print('MISSING', label, str(path))
    else:
        actual = sha(path)
        print('MATCH' if actual == expected else 'HASH_MISMATCH', label, str(path), actual)
val_root = Path(c['cache_root'])/'val'
print('RECORDED_VAL_CACHE_ROOT', str(val_root), 'EXISTS', val_root.is_dir())
ready_path = Path(c['ready_manifest'])
if ready_path.is_file() and sha(ready_path) == c['ready_manifest_sha256']:
    rows = [json.loads(line) for line in ready_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1200 and all(row['split'] == 'val' for row in rows)
    present = 0
    for row in rows:
        path = Path(row['feature_path'])
        assert path == val_root/row['clip_id']/'features.npz'
        if path.is_file():
            present += 1
    print('RECORDED_VALIDATION_FEATURE_FILES_PRESENT', present, 'OF', len(rows))
else:
    print('FULL_CACHE_COVERAGE_UNVERIFIED: original ready manifest unavailable or changed')
# This is the package's reference copy, not proof the original G5 run exists.
reference = addon/'assets/accepted_g5_context.json'
print('PACKAGED_REFERENCE_CONTEXT', str(reference), sha(reference),
      'MATCHES_ORIGINAL', sha(reference) == c['g5_context_sha256'])
# Filename-only inspection of existing Colab-local folders; never recurse Drive/test data.
print('TOP_LEVEL_COLAB_DIRECTORIES', sorted(p.name for p in Path('/content').iterdir() if p.is_dir()))
for base in (Path('/content/g5_results'), Path(c['g5_root'])):
    if base.is_dir():
        print('EXISTING_LOCAL_CHILDREN', str(base), sorted(p.name for p in base.iterdir()))
print('READ_ONLY CHECK COMPLETE. Return this output. Do not rerun the pilot yet.')
```

The full original G5 run or its assets may need a user-managed restoration, or its actual existing location must be identified and reviewed. A missing context file does not prove that the checkpoint, manifests or cache are present. Do not upload only a context file and rerun before this inventory is reviewed. No install or additional version approval is needed for the exact observed runtime values, which match the approved amendment.

**Stop at C8.2.** No model inference, full validation, Day 9 or Day 10 is authorized by this diagnostic cell.
