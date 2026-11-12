# B2 evaluation-only continuation — finish calculations before one block review

The returned B2 execution passed independent audit. All 1200 query rows are schema-valid, finite, unique and match the original validation IDs, checkpoint and 300-per-category cohort. All 2400 feature/metadata identity records match accepted G4 evidence. Package/source, actual C8A, runtime, restoration and input hashes reconcile. Original three predictions and calibration remain unchanged.

The exact new canonical output is now in the local clone: `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/predictions/restore_query_val.jsonl`, SHA-256 `855a90bf1a72c59043ac3e80f04ec15df16d10ea0b98056ff10c56d751d5dbf1`. The user-run model attempt was `B2_20261112T153337Z_f641e297_validation`; it is COMPLETE and must not be repeated. Returned ZIP SHA-256: `ff71f1549b1c63850757378290b03f38de0f1241767047eb1f20d42b65308185`.

## Real matched cost findings

Twelve schema-valid records retain all 600 raw measured samples: original short/typical/long, four modes each, 10 warmups and 50 repeats, CPU/SDPA/FP32 with fake INT8 boundaries, same checkpoint and workload identity. CPU memory is null/not_measured. These are cached-model timings, not real hardware INT8 claims.

| Workload | ref p50 ms | lowp p50 ms | V/output p50 ms | Q p50 ms | E_value ms | E_query ms |
|---|---:|---:|---:|---:|---:|---:|
| short | 115.841651 | 120.460469 | 117.882037 | 171.228341 | -2.578432 | 50.767872 |
| typical_anchor | 118.454390 | 161.434481 | 119.918150 | 167.474019 | -41.516331 | 6.039538 |
| long | 121.195739 | 158.257662 | 121.530770 | 157.677357 | -36.726892 | -0.580305 |

Typical remains primary; short/long are sensitivity checks. All three value/output extra costs are nonpositive, so the planned comparison of BCE recovery per **positive added millisecond** is undefined. Do not force a ratio, change its denominator/sign, assume equal cost, repeat the benchmark to seek a desired sign, or call these hardware INT8 gains. Full timing variability and p95 remain in the original returned records for review.

## Earlier interface concern resolved through accepted Day B precedent

The prior handoff treated missing per-stratum CLI output as requiring a Member 2 patch and explicit fixed-count stratification. The accepted `scripts/dayB_finish/run_g8_g9.py` already demonstrates the intended route: use Member 2 `aligned`, `paired_delta`, `cluster_ci` and `group_clusters` unchanged inside overall and each fake-versus-real analysis. This continuation reuses that route for all four modes; no Member 2/3 source, numerical precision definition or bootstrap algorithm is edited.

Method disclosure for the lead's **C8B full-block review**: analysis subsets are stratified (overall; audio_fake-vs-real; video_fake-vs-real; both_fake-vs-real). Within each subset, the original engine resamples whole connected source/identity groups with replacement. It does **not** hold category counts fixed in every replicate. This preserves the accepted Day B engine and seed. A different fixed-category-count resampling design would be a separate protocol change and is not implemented or claimed. Exact method/source provenance is included with the results; no human owner endorsement is fabricated.

## One upload, one cell

Keep the **same approved Colab CPU runtime and completed B2 folder/results**. Upload only `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayC/member1/colab/AFML_DayC_B2_Evaluation.zip` into `/content`. SHA-256 `1f30245c2393d586cda4f60d051422357a93c5d724dc62e29a1280505c22671b`; 25723 bytes, 8 files. No predictions, labels, checkpoint, feature arrays or media are included in this small source/metadata add-on. It reads the already saved validation outputs and exact original metadata in the existing runtime. It does not mount Drive, install, download, infer, benchmark again, recalibrate or access test features.

The runtime/version checks stay exact to the approved B2 arrangement. The local Mac runtime is used for tests only; real validation statistics are not silently moved into its different numerical environment.

The one job runs unchanged Member 2 prediction-check, evaluate and paired scripts with the original frozen threshold and 1000 source-group bootstrap replicates, seed 0; retains their original JSON/Markdown/plots; completes all four paired contrasts within each fake-versus-real stratum via the same public owner functions; supplies missing per-stratum BCE/FPR intervals via owner `cluster_ci`; and prepares the Member 1 sign/undefined-cost/research-limitation packet. Its added result bindings are new files; no Member 2 output is overwritten. No policy is selected or frozen.

```python
from pathlib import Path
from google.colab import files
import hashlib, json, os, re, stat, subprocess, sys, zipfile

archive = Path('/content/AFML_DayC_B2_Evaluation.zip')
addon = Path('/content/AFML_DayC_B2_Evaluation')
assert archive.is_file(), 'Upload AFML_DayC_B2_Evaluation.zip first'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '1f30245c2393d586cda4f60d051422357a93c5d724dc62e29a1280505c22671b', 'Evaluation ZIP checksum mismatch'
assert Path('/content/AFML_DayC_B2_Full_Validation').is_dir(), 'Keep the completed B2 runtime and saved outputs; do not rerun inference'
assert not addon.exists(), 'Preserve any existing evaluation attempt; do not reuse the folder automatically'
with zipfile.ZipFile(archive) as z:
    names = z.namelist()
    assert z.testzip() is None and len(names) == len(set(names))
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
    assert {Path(n).parts[0] for n in names} == {addon.name}
    assert all(not stat.S_ISLNK(i.external_attr >> 16) for i in z.infolist())
    z.extractall('/content')
manifest = json.loads((addon/'EVALUATION_MANIFEST.json').read_text())
for name, expected in manifest['files'].items():
    assert hashlib.sha256((addon/name).read_bytes()).hexdigest() == expected
print('B2 evaluation only: saved 1200 predictions, 1000 bootstrap replicates, seed 0.')
print('No model inference, feature-cache access, benchmark rerun, recalibration or test scoring.')
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg', GIT_OPTIONAL_LOCKS='0')
lines = []
with subprocess.Popen([sys.executable, '-B', str(addon/'finish_day8_evaluation.py')],
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
    assert review.is_file() and review.is_relative_to(Path('/content/AFML_DayC_B2_Full_Validation').resolve())
    with zipfile.ZipFile(review) as z:
        hashes = json.loads(z.read('output_hashes.json'))
        assert set(z.namelist()) == set(hashes) | {'output_hashes.json'}
        for name, expected in hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
        if 'failure.json' in z.namelist():
            print('HALTED:', json.loads(z.read('failure.json')))
        else:
            completion = json.loads(z.read('evaluation_completion.json'))
            assert completion['status'] == 'B2_EVIDENCE_COMPLETE_PENDING_C8B_REVIEW'
            print('B2 calculations complete. Return evidence for the full-block C8B review.')
    print('RETURN_ZIP_SHA256:', hashlib.sha256(review.read_bytes()).hexdigest())
    files.download(str(review))
else:
    print('No return ZIP. Return complete output; preserve every directory and do not rerun inference.')
print('STOP: no B3/Day9 policy freeze or held-out work before the full-block C8B acknowledgement.')
```

Return **DAYC_B2_evaluation_review.zip**, even on failure. It includes original Member 2 metrics/plots/paired results, all supplemental per-stratum CIs, exact input/source/runtime provenance, commands/logs, explicit resampling disclosure, and the Member 1 H2/H3 packet. The cell verifies output hashes and prints the actual returned ZIP SHA. No future metrics, CIs or result hashes are invented here. Preserve all saved inference and benchmark artifacts; this job never regenerates them.

## Current block status

C8.3 full query output is verified and imported; C8.4 real matched cost evidence passes technical checks. C8.5/C8.6 full statistical computation is the remaining external dependency; C8.7 full-block review follows its audit. **315 Member 1 tests passed, 0 failed/skipped**, including all prior guards and synthetic delegation tests proving every required owner API call uses full 1000 replicates/seed 0, all four analysis strata and all four contrasts. A packaged entry smoke test rejects the local Mac runtime before real statistical work. No real model or real CI computation was performed by the agent during this continuation.

After returned evidence passes audit, present the complete B2 packet for **one C8B review**. No B3 work starts until `ACK DAYC C8B: ACCEPT DAY8 VALIDATION EVIDENCE`. This manual execution handoff is not an additional research-review gate. Five implementation-block reviews remain B1/C8A (accepted), B2/C8B, B3/C9, B4/C10, B5/C10R.


## Latest user steering: joint B2/B3 audit

The user subsequently authorized preparing B3 source before reviewing B2/B3 together. Use `COLAB_B2_B3_COMBINED_AUDIT.md` as the current self-contained handoff. Earlier ZIPs and instructions are preserved; do not run multiple evaluation packages automatically. Source preparation does not sign owner releases, select/freeze a policy or authorize a held-out attempt.
