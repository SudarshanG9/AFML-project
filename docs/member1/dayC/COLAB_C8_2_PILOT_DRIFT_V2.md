# B1 / C8.2 amended CPU pilot — v2 manual handoff

**Status: BLOCKED: AWAITING USER-RUN COLAB EVIDENCE.** The user explicitly supplied `APPROVE C8.2 PILOT WITH RECORDED VERSION DRIFT; KEEP TORCH UNCHANGED` and the batched execution controller. This package resumes B1 only. No C8A/full-validation, Day 9 or held-out authorization has been inferred.

## One manual action

Upload `AFML_DayC_C8_2_CPU_Pilot_Drift_v2.zip` into `/content` in the existing CPU Colab runtime, then paste the cell below once. Leave the original `AFML_DayC_C8_2_CPU_Pilot` folder and all failure evidence untouched. V2 extracts into a separate new directory and automatically downloads its review ZIP on success **or** an ordinary runtime/preflight/model failure. It does not reinstall anything, mount Drive, download assets or access an account. The agent does not execute these remote actions.

```python
from pathlib import Path
from google.colab import files
import hashlib, json, os, re, stat, subprocess, sys, zipfile

archive = Path('/content/AFML_DayC_C8_2_CPU_Pilot_Drift_v2.zip')
addon = Path('/content/AFML_DayC_C8_2_CPU_Pilot_Drift_v2')
assert archive.is_file(), 'Manually upload the v2 ZIP first'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '0ca59df33a61a4d9ca80b034865184b13b5a70a6420664c5ea8530746d9ff3f0', 'Wrong or changed archive'
assert not addon.exists(), 'Preserve existing v2 evidence; do not rerun or reuse this directory'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    assert {Path(n).parts[0] for n in z.namelist()} == {addon.name}
    assert all(not stat.S_ISLNK(info.external_attr >> 16) for info in z.infolist())
    z.extractall('/content')
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
    PYTHONPATH=str(addon/'src') + os.pathsep + str(addon),
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg', GIT_OPTIONAL_LOCKS='0')
print('Running the approved v2 validation pilot; Torch will not be installed or changed.')
result = subprocess.run([sys.executable, '-B',
    str(addon/'scripts/dayC_member1/colab_query_pilot.py')],
    cwd=addon, env=env, capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print('--- STDERR ---')
    print(result.stderr)
print('Exit code:', result.returncode)
matches = re.findall(r'^RETURN REVIEW ZIP: (.+)$', result.stdout, flags=re.MULTILINE)
if matches:
    review = Path(matches[-1].strip()).resolve()
    assert review.is_file() and review.is_relative_to(addon.resolve())
    with zipfile.ZipFile(review) as z:
        hashes = json.loads(z.read('output_hashes.json'))
        assert set(z.namelist()) == set(hashes) | {'output_hashes.json'}
        for path, expected in hashes.items():
            assert hashlib.sha256(z.read(path)).hexdigest() == expected
        if 'failure.json' in z.namelist():
            print('PILOT HALTED:', json.loads(z.read('failure.json'))['error'])
        else:
            print('Pilot complete; C8A review is still required.')
    print('Downloading review evidence; return this ZIP even if the pilot halted.')
    files.download(str(review))
else:
    print('No review ZIP was produced. Return the stdout/stderr above; do not rerun.')
print('STOP. No full-validation, Day 9 or Day 10 execution is authorized by this cell.')
```

## Exact upload identity

- Local archive: `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayC/member1/colab/AFML_DayC_C8_2_CPU_Pilot_Drift_v2.zip`
- SHA-256: `0ca59df33a61a4d9ca80b034865184b13b5a70a6420664c5ea8530746d9ff3f0`
- Size: 128272 bytes; 53 files.
- Manifest SHA-256: `9f42b505744c369a3703e8361f579e7fc0114406f2c98c6f897520f80813aa6d`.
- All 41 original G5 source/schema file bytes match the frozen approved manifest.
- Excludes weights, features, media, dataset rows and test labels. It includes the actual scoped approvals, release and original G5 context metadata.

## Approved amendment, not a silent environment replacement

| Dependency | Recorded Day B | Explicitly approved pilot |
|---|---|---|
| NumPy | 2.1.3 | 2.4.6 |
| jsonschema | 4.23.0 | 4.26.0 |
| pytest | 8.3.4 | 8.4.2 |
| PyTorch | 2.14.1+cpu | **unchanged: 2.14.1+cpu** |

Python 3.13.18, the exact recorded Linux platform, CPU device, CUDA-unavailable state, one Torch/interop thread, scikit-learn 1.6.1 and matplotlib 3.10.0 still require exact matches. Any other drift stops. The original Day B environment remains recorded in the release; `runtime_observed.json` and the comparison explicitly identify the amendment and are preserved in failure bundles. There are no install, force, retune, alternate-data or version-override flags.

## Recorded inputs: exact historical paths; live availability UNVERIFIED locally

- G5 root: `/content/AFML_G5_CPU_Colab_20261111T180139Z`
- G5 context: `/content/g5_results/G5_CPU_20261111T181935Z_422cf72f/execution_context.json`
- Canonical manifest: `/content/AFML_G5_CPU_Colab_20261111T180139Z/data/manifests/manifest_whole.jsonl` — hash only; test rows are never opened.
- Validation manifest: `/content/AFML_G5_CPU_Colab_20261111T180139Z/data/manifests/manifest_whole_val.jsonl`
- Ready validation manifest: `/content/AFML_G5_CPU_Colab_20261111T180139Z/data/g4_evidence/dayB_val_ready.jsonl`
- Validation cache root: `/content/drive/.shortcut-targets-by-id/1yKf-S0Yyi8ZU7uLx6JbYKZpL95Kn3Mb7/dimodif_emb/val`
- Checkpoint: `/content/g5_results/G5_CPU_20261111T181935Z_422cf72f/runtime/dimodif/ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.pth`
- Model config: same directory, `dfd_lavdf_reduceonplateau_64_4_1_5_True.json`.
- Approved fixed pilot: `7cea6ad75b4ebf41`; no new clip is selected from results.

Important SHA-256 values:

- Canonical manifest: `8339b2476a71e62c6c6fca7d937ba4cc7a704b09d26f08506c20d1f10de8f3dd`
- Validation manifest: `22d1d9e02403cfdbb05f2116da16bc4015f540547a4ac00bf6b7127c119e359c`
- Ready manifest: `c79ad908517878ebf0f3aa0e0dac053a01d9d5c4eb264583db26096985890481`
- Checkpoint: `e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e`
- Pilot feature: `5bf79b6b61f94989e07183c80a4527986edb06f7198920f87307fee75497865f`

All other source, model-config, context, G4 report, pilot-metadata and package hashes are pinned in the release/manifest and checked by the runner. The ready manifest is never rewritten. The complete 1,200 validation metadata cohort, identical IDs and 300 per category must reconcile, and all 1,200 feature/metadata files must exist. This is full-cohort readiness checking, not a sampled replacement for the later full experiment. Only the approved pilot feature is loaded for inference; the entire cache is not claimed to be rehashed this session.

## Additional reference check before diagnostic arms

The locally verified Day B reference prediction file has SHA-256 `e146432023638635b53dc517f3b4b06ecd87a6599efe3709ed14b145c7458ff3`. For the approved typical clip it records native fake probability **7.566657586721703e-05**. The builder verified that value against the actual six-field validation prediction row.

After native/instrumented reference parity, v2 compares the current native score to this fixed value using the existing reference-score tolerance **atol=1e-6, rtol=1e-5**. A disagreement stops before the three diagnostic arms and is returned as failure evidence. Passing this check establishes only single-clip score consistency, not population, metric, timing or runtime equivalence. No model weights, preprocessing, threshold or quantization math are changed.

The pilot uses one adapter and loads the one feature once: one native reference call, one instrumented reference call, then one each for low_precision, restore_value_output and restore_query. It retains all-five-layer/20-boundary traces, finite original probabilities, model/cache integrity and method cleanup. Fake INT8 remains floating-point execution. No scientific accuracy/H3 claim is made from a pilot.

## Return exactly one artifact

The cell prints a fresh run directory beneath `/content/AFML_DayC_C8_2_CPU_Pilot_Drift_v2/artifacts/dayC/member1/day8/C8_2_PILOT_<timestamp>_<id>/` and downloads **C8_2_pilot_review.zip**. It includes `output_hashes.json`, runtime diagnostics, and either failure evidence or completed preflight/reference-consistency/pilot evidence. The cell checks every recorded output hash before download. A future run's actual hashes are not invented here; they are generated from its actual files.

Return the ZIP, whether the run passes or stops. If no ZIP is produced, return the printed stdout/stderr. Do not rerun a partial/model attempt or reuse the v2 extraction. No automatic retry occurs. The prior v1 run was reported to stop at version checking; its ZIP was not received and has not been independently audited. This v2 attempt follows the user's explicit remediation approval.

## Local checks and next gate

**204 Member 1 tests passed; 0 failed, 0 skipped.** They include exact amended versions, rejection of Torch/other dependency/thread drift, reference-score disagreement, original source preservation, package integrity, private-asset exclusion and a packaged-entry smoke test that stops on the local Mac before model/feature access. No live Colab/Drive operation or real pilot was executed by the agent.

After returned evidence is audited, request **C8A**: `ACK DAYC C8A: AUTHORIZE FULL VALIDATION QUERY CONTROL`. Do not start B2 before that ACK. Pilot-only runtime approval does not automatically authorize mixing the amended query run with old full-cohort metrics or timing; that comparability issue remains part of C8A review.
