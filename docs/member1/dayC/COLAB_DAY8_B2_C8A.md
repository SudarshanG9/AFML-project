# B2 C8A release — one manual Colab execution handoff

C8A acknowledgement accepted: `ACK DAYC C8A: AUTHORIZE FULL VALIDATION QUERY CONTROL`. Approved context SHA-256 `4b0063628fe44161aceae1a8b782c8f65240c41feb97157cd24fc94ce027933e`. This advances B2; it does not approve C8B, Day 9 freeze or held-out work.

All independent B2 source preparation is complete. **EXTERNAL HANDOFF REQUIRED: user-run Colab inference and cost evidence.** The agent has not accessed Colab/Drive or run a real model. B2 remains in progress until real outputs and required Member 2/3 evidence pass review. This handoff is not an additional block-review request.

## Upload once

Upload `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayC/member1/colab/AFML_DayC_B2_C8A_Release.zip` into `/content`. Use **AFML_DayC_B2_C8A_Release.zip** only; earlier locally retained Full_Validation/Verified archives are preparation candidates, not the current release. Expected SHA-256 `dbd44345b383645b12f6a03dae7778165ec0f82df6018dfec4cd27e7706b0882`; 250928 bytes, 76 files. Package manifest SHA-256 `c301d27c47e57f00c2a356ee83c5e68739c97ae5c579c1c0db09b626515fa408`. Existing baseline prediction bytes and fixed calibration are included at their original repository-relative paths; all are validation-only. Checkpoint, feature arrays, raw media and test rows are excluded.

Fresh extraction root: `/content/AFML_DayC_B2_Full_Validation`. Keep all old Day B/Day C folders and evidence. Existing restored original inputs remain at their historical locations; leave the fresh `/content/afml_drive` mount connected. No mounts, installations, remote downloads or data-copy/rebuild operations are performed by the cell. Do not run it locally.

## Exact inputs

- `/content/AFML_G5_CPU_Colab_20261111T180139Z/data/manifests/manifest_whole.jsonl` — SHA-256 `8339b2476a71e62c6c6fca7d937ba4cc7a704b09d26f08506c20d1f10de8f3dd`
- `/content/AFML_G5_CPU_Colab_20261111T180139Z/data/manifests/manifest_whole_val.jsonl` — SHA-256 `22d1d9e02403cfdbb05f2116da16bc4015f540547a4ac00bf6b7127c119e359c`
- `/content/AFML_G5_CPU_Colab_20261111T180139Z/data/g4_evidence/dayB_val_ready.jsonl` — SHA-256 `c79ad908517878ebf0f3aa0e0dac053a01d9d5c4eb264583db26096985890481`
- `/content/AFML_G5_CPU_Colab_20261111T180139Z/data/g4_evidence/preflight_report.json` — SHA-256 `559445575ff51495dbb79ae175737ad1596fcc41afab058734d138bd01fb9c01`
- `/content/g5_results/G5_CPU_20261111T181935Z_422cf72f/execution_context.json` — SHA-256 `c03488cb787b1230bd86274dc7a36d39983574c7e2bc996c931dcfc8a835bd7c`
- Checkpoint/config/models.py remain at `/content/g5_results/G5_CPU_20261111T181935Z_422cf72f/runtime/dimodif` and their three original relative paths, with original hashes pinned in the C8A context.
- Read-only cache mapping: `/content/drive/.shortcut-targets-by-id/1yKf-S0Yyi8ZU7uLx6JbYKZpL95Kn3Mb7/dimodif_emb` → `/content/afml_drive/MyDrive/AFML_Project/dimodif_emb`. Only approved `val/<clip_id>/features.npz` and companion `meta.json` reads are allowed; the manifest itself is never rewritten. All 1200 feature and 1200 metadata hashes are verified before model loading, and each feature/meta identity is checked again when loaded by the child.
- The canonical metadata is read by the unchanged Member 2 manifest/split audit to reconcile validation labels and connected source/identity groups. No test feature, ready-test manifest or test prediction is opened. Canonical labels never enter prediction rows.
- Same approved CPU environment as the successful pilot: Torch 2.14.1+cpu unchanged, Python 3.13.18 and exact recorded Linux platform, one Torch/interop thread, NumPy 2.4.6/jsonschema 4.26.0/pytest 8.4.2, scikit-learn 1.6.1/matplotlib 3.10.0. Any other drift stops; no automatic replacements.

## Execution scope

1. Validate actual C0/C8A records, exact source hashes, original checkpoint/config/source/manifests/context/calibration and old three prediction hashes. Refuse an existing query output or prior attempt lock. Reconcile all 1200 IDs, 300 per category, and source/identity split audit.
2. Invoke the original Day B validation loop once in a fresh one-thread subprocess for `restore_query`. The read-only bridge adapts only `load_features` path resolution and verifies original file identity; the model, original loop, feature loading/FP32 conversion and precision math are unchanged. A 1200-clip synthetic integration test proves this delegation. No replacement manifest or raw-feature extraction.
3. Validate all 1200 saved schema-valid probabilities, same IDs, single unique run ID, val/config/checkpoint and unchanged protected inputs. Save only the six existing prediction schema fields at `artifacts/predictions/restore_query_val.jsonl` inside the extraction root. Preserve execution context, input/source hashes, pilot reference, command/log, completion and persistent attempt status.
4. Invoke the unchanged benchmark runner on the three approved workloads, one record per four modes, 10 warmups/50 measured repetitions each. Planned outputs: 12 records and 600 raw measured samples, plus warmups; no sample reduction. Validate schema, identical within-workload runtime/input identity, approved lengths and CPU memory honesty. Save `four_modes_short.jsonl`, `four_modes_typical.jsonl`, `four_modes_long.jsonl`; keep typical primary.
5. Bundle outputs on success or an ordinary preflight/model failure. No automatic retry and no silent partial completion. Member 2 evaluation is not launched while its complete-plan resampling/per-stratum requirements remain unresolved. The research reader and sign/undefined-value checks are locally implemented and tested, ready for owner-supplied evidence.

## One Colab cell

```python
from pathlib import Path
from google.colab import files
import hashlib, json, os, re, stat, subprocess, sys, zipfile

archive = Path('/content/AFML_DayC_B2_C8A_Release.zip')
addon = Path('/content/AFML_DayC_B2_Full_Validation')
assert archive.is_file(), 'Upload the C8A Release ZIP into /content first'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == 'dbd44345b383645b12f6a03dae7778165ec0f82df6018dfec4cd27e7706b0882', 'Archive checksum mismatch'
assert not addon.exists(), 'Preserve any existing B2 attempt. Do not reuse or rerun this folder.'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    names = z.namelist()
    assert len(names) == len(set(names))
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
    assert {Path(n).parts[0] for n in names} == {addon.name}
    assert all(not stat.S_ISLNK(i.external_attr >> 16) for i in z.infolist())
    z.extractall('/content')
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
    PYTHONPATH=str(addon/'src') + os.pathsep + str(addon),
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg', GIT_OPTIONAL_LOCKS='0')
print('B2: exact 1200 cache identities, one full query-validation run, then 3 x 4-mode benchmarks.')
print('Keep the existing /content/afml_drive mount connected. No installs, mounts or test access.')
print('Inference progress is saved in <printed B2_RUN_DIR>/validation/query_validation_run.log.')
lines = []
with subprocess.Popen([sys.executable, '-B', str(addon/'scripts/dayC_member1/colab_day8.py')],
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
            print('HALTED; preserve evidence and do not rerun:', json.loads(z.read('failure.json')))
        else:
            completion = json.loads(z.read('validation/validation_completion.json'))
            assert completion['count'] == 1200 and completion['status'] == 'COMPLETE'
            print('Full query validation and 12 benchmark records produced; B2 evaluation remains pending.')
    print('RETURN_ZIP_SHA256:', hashlib.sha256(review.read_bytes()).hexdigest())
    files.download(str(review))
else:
    print('No return ZIP. Return the complete output above; preserve this folder and do not rerun.')
print('Return DAYC_B2_review.zip even on failure. Continue B2 after audit; no C8B review yet.')
```

## Return exactly one artifact

Return **DAYC_B2_review.zip**, even on failure. Successful output contains the actual 1200-row query file, full cache-identity proof, validation context/provenance/log/status, original pilot reference, three matched four-mode benchmark files with all raw samples and per-workload commands/logs, measured descriptive costs and output hashes. The cell verifies every bundle hash and prints the actual ZIP hash before download. Future result hashes are not invented here. If no ZIP is produced, return complete stdout/stderr without rerunning a partial model attempt.

Validation progress is redirected to `<printed B2_RUN_DIR>/validation/query_validation_run.log`; the cell prints full-cache verification and benchmark stage completion. Keep the runtime running throughout all stages. A failure preserves the query attempt lock and existing data.

## Local evidence and block status

291 Member 1 tests passed, 0 failed/skipped. Actual release entry-point smoke test on the local Mac rejects the unapproved environment before model construction, feature reads or claiming a full validation attempt; failure evidence ZIP is produced. Packaged unchanged Member 2/3 modules import successfully. Original source, checkpoint, manifests, three predictions and calibration were rehashed against approved values. Evidence: `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayC/member1/day8/B2_SOURCE_20261112T150540771598Z`.

[Owner handoff requirements](B2_MEMBER2_MEMBER3_HANDOFF_REQUIREMENTS.md) describes the exact unresolved Member 2/3 evidence. After this ZIP returns, continue remaining B2 work; ask for **C8B only once the whole B2 research packet is complete**. The later review gates remain B3/C9, B4/C10 and B5/C10R. No new routine approval is requested here.


## Recovery after returned missing-original preflight

The first returned B2 bundle reported the canonical manifest absent before model launch. For this specific preflight-only case, use `COLAB_B2_PREFLIGHT_RECOVERY.md` instead of the fresh-extraction cell above. It verifies and preserves the existing package/failure evidence, restores only exact original inputs, refuses any existing model-attempt marker/output and continues the unchanged B2 runner. C8A remains accepted; this is execution recovery, not an extra block-review gate.
