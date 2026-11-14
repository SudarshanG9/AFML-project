# B1 C8.2 — mapped validation cache, verified v3 handoff

Status: **BLOCKED: AWAITING USER-RUN COLAB EVIDENCE**. All independent B1 local implementation/tests are complete. The new root is user-reported accessible; its exact identity remains unverified until this manual run returns evidence. No C8A authorization is inferred.

Upload only `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayC/member1/colab/AFML_DayC_C8_2_CPU_Pilot_Mapped_v3_Verified.zip` to `/content`, keep the fresh Drive mount connected, and run the single cell below once in the approved Colab CPU runtime. Keep every old folder and result. The final upload is **Mapped_v3_Verified.zip**, not the retained intermediate Mapped_v3.zip. Extracted directory: `/content/AFML_DayC_C8_2_CPU_Pilot_Mapped_v3`.

SHA-256: `8cb453863ffad1f688318729ca158497b04e2b15161f752faf1521f6a09bc33a`  
Size: 133979 bytes; 56 files; 41 original G5 source/schema files verified against their frozen manifest. No checkpoint, cached features or dataset rows are packaged. Package manifest SHA-256: `1a90298829c6d5d0b6665cfa616c0135a9b30d0a65ce1b3dd98dbc04a75918b4`.

## Exact inputs and verification

| Input | Approved original Colab path | Required SHA-256 |
|---|---|---|
| canonical_manifest | `/content/AFML_G5_CPU_Colab_20261111T180139Z/data/manifests/manifest_whole.jsonl` | `8339b2476a71e62c6c6fca7d937ba4cc7a704b09d26f08506c20d1f10de8f3dd` |
| validation_manifest | `/content/AFML_G5_CPU_Colab_20261111T180139Z/data/manifests/manifest_whole_val.jsonl` | `22d1d9e02403cfdbb05f2116da16bc4015f540547a4ac00bf6b7127c119e359c` |
| ready_manifest | `/content/AFML_G5_CPU_Colab_20261111T180139Z/data/g4_evidence/dayB_val_ready.jsonl` | `c79ad908517878ebf0f3aa0e0dac053a01d9d5c4eb264583db26096985890481` |
| g4_report | `/content/AFML_G5_CPU_Colab_20261111T180139Z/data/g4_evidence/preflight_report.json` | `559445575ff51495dbb79ae175737ad1596fcc41afab058734d138bd01fb9c01` |
| g5_context | `/content/g5_results/G5_CPU_20261111T181935Z_422cf72f/execution_context.json` | `c03488cb787b1230bd86274dc7a36d39983574c7e2bc996c931dcfc8a835bd7c` |
| model input | `/content/g5_results/G5_CPU_20261111T181935Z_422cf72f/runtime/dimodif/ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.pth` | `e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e` |
| model input | `/content/g5_results/G5_CPU_20261111T181935Z_422cf72f/runtime/dimodif/ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.json` | `bea128dc7dad6b72bc0487dcc45d03ae3322e48fd597de1a5cd83ad036f30651` |
| model input | `/content/g5_results/G5_CPU_20261111T181935Z_422cf72f/runtime/dimodif/src/models.py` | `e74b5e3b5fcee24353f9940dddd48d9ac1982ced725664db2518f63a6d853778` |

The nine restored files were previously reported matching by the user; this runner checks the required original inputs again. Their corresponding local originals were independently hash-verified during preparation. The original G5 context is preserved byte for byte, including its historical paths. It is not regenerated.

The ready manifest retains `/content/drive/.shortcut-targets-by-id/1yKf-S0Yyi8ZU7uLx6JbYKZpL95Kn3Mb7/dimodif_emb`. The in-memory resolver accepts only its exact `val/<approved_clip_id>/features.npz` layout and resolves reads to `/content/afml_drive/MyDrive/AFML_Project/dimodif_emb/val/<approved_clip_id>/features.npz`. It never creates an old mount, rewrites a manifest, copies data, changes metadata, or walks the test directory.

Before model construction, the runner reconciles the original 1200 validation/ready/G4 IDs, all 300-per-category strata, unchanged validation metadata, and G4 source/manipulation identity. It streams SHA-256 over **all 1200 feature files and all 1200 meta.json files**, matching every recorded original hash. Exact metadata-file hashes establish original metadata identity without parsing or rewriting their contents. Missing hashes/files, ID differences, path deviations, symlinks and byte mismatches stop before model loading. Extra unrelated folders do not establish or alter cohort membership. Feature arrays are not loaded during this verification. Progress prints every 100 clips; Drive hashing may take time.

After all input/runtime checks pass, the same unchanged adapter loads one fixed clip `7cea6ad75b4ebf41` once. It runs native/instrumented reference parity and the fixed Day B score check, followed by low_precision, restore_value_output and restore_query with 20 boundary traces each. State, input, pilot feature and metadata preservation are checked. Full-cache hashes describe preflight identity, not continuous storage locking during inference.

The runtime keeps Torch 2.14.1+cpu unchanged, CPU and one Torch/interop thread. Exact original Python/platform and other dependencies remain required. Only the already-approved NumPy 2.4.6, jsonschema 4.26.0 and pytest 8.4.2 drift is allowed. The v2 runtime approval remains bound to its original release SHA; v3 separately records the user's path-mapping instruction. Numerical definitions, model and checkpoint are unchanged. No installs, mounts, network access, extraction of data, training, or full-validation inference occurs.

## One manual Colab cell

```python
from pathlib import Path
from google.colab import files
import hashlib, json, os, re, stat, subprocess, sys, zipfile

archive = Path('/content/AFML_DayC_C8_2_CPU_Pilot_Mapped_v3_Verified.zip')
addon = Path('/content/AFML_DayC_C8_2_CPU_Pilot_Mapped_v3')
assert archive.is_file(), 'Upload the Verified ZIP into /content first'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '8cb453863ffad1f688318729ca158497b04e2b15161f752faf1521f6a09bc33a', 'Archive hash mismatch'
assert not addon.exists(), 'Preserve existing evidence. Do not reuse or rerun this package folder.'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    names = z.namelist()
    assert len(names) == len(set(names))
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
    assert {Path(n).parts[0] for n in names} == {addon.name}
    assert all(not stat.S_ISLNK(i.external_attr >> 16) for i in z.infolist())
    z.extractall('/content')
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
    PYTHONPATH=str(addon/'src') + os.pathsep + str(addon),
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg', GIT_OPTIONAL_LOCKS='0')
print('Verifying all 1200 validation feature + metadata hashes; then one approved pilot.')
print('No install, mount, manifest rewrite, test access or full-validation inference.')
lines = []
with subprocess.Popen([sys.executable, '-B', str(addon/'scripts/dayC_member1/colab_query_pilot.py')],
        cwd=addon, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1) as process:
    for line in process.stdout:
        lines.append(line)
        print(line, end='', flush=True)
    returncode = process.wait()
print('Exit code:', returncode)
matches = re.findall(r'^RETURN REVIEW ZIP: (.+)$', ''.join(lines), flags=re.MULTILINE)
if matches:
    review = Path(matches[-1].strip()).resolve()
    assert review.is_file() and review.is_relative_to(addon.resolve())
    with zipfile.ZipFile(review) as z:
        hashes = json.loads(z.read('output_hashes.json'))
        assert set(z.namelist()) == set(hashes) | {'output_hashes.json'}
        for name, expected in hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
        if 'failure.json' in z.namelist():
            print('HALTED:', json.loads(z.read('failure.json')))
        else:
            proof = json.loads(z.read('cache_verification.json'))
            pilot = json.loads(z.read('pilot.json'))
            assert proof['status'] == 'PASS' and proof['verified_count'] == len(proof['records']) == 1200
            assert pilot['status'] == 'PILOT_COMPLETE_PENDING_C8A'
            print('1200 exact validation identities verified; single pilot complete, pending review.')
    print('REVIEW_ZIP_SHA256:', hashlib.sha256(review.read_bytes()).hexdigest())
    files.download(str(review))
else:
    print('No review ZIP. Return this complete output; preserve the directory and do not rerun.')
print('STOP: return the review ZIP even on failure. Await evidence audit and C8A before B2.')
```

Return **C8_2_pilot_review.zip**, even on failure. It contains output hashes, observed runtime, either failure evidence or full-cohort per-file identity proof, preflight source hashes, reference consistency and real pilot traces/scores. Actual result hashes are printed from the run, never invented in advance. Download is attempted for an ordinary preflight/model failure as well as success. If Colab interrupts or no ZIP is produced, return the complete cell output without rerunning a model attempt.

## Local checks and limits

**244 Member 1 tests passed; 0 failed; 0 skipped.** Synthetic tests cover exact 1200 identities, feature and metadata corruption, ID/metadata mismatches, obsolete-root and sealed-test access guards, missing files/hashes, symlink/path escapes, hidden protocol changes, runtime and mapping approval chains, and all three packaged entry points rejecting the local Mac before model/cache access. Evidence: `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayC/member1/day8/MAPPED_V3_LOCAL_VERIFICATION`. The earlier 242-test run and intermediate package are preserved.

No real-model inference or measured Day C scientific finding has been produced locally. The full 1200-clip inference, measured four-mode costs, H3 analysis and Day 9/10 remain gated. After the returned pilot passes audit, the next mandatory research gate is `ACK DAYC C8A: AUTHORIZE FULL VALIDATION QUERY CONTROL`. The pilot-only version amendment does not establish population/timing equivalence with old Day B runs. Query vs V/output equal cost remains unmeasured.
