# Joint B2/B3 audit — source complete, no final freeze or held-out execution

Your latest instruction authorizes preparing B3 now and reviewing B2/B3 together. It supersedes the earlier stop before B3 source development. It is recorded as source-preparation authorization, not a fabricated C8B acceptance, C9 freeze approval or C10 test authorization.

## What is implemented

B3 C9.1–C9.4 source: validation-only Day9 preflight; strict actual-evidence H2/H3 validator; Member2 final threshold/seal/CI release and Member3 runtime/benchmark release checks; exact source/input hash registry; exclusive propose/finalize commands. Proposals stay `PROPOSED_NOT_APPROVED`; no policy can be finalized from a CLI flag or this model's recommendation.

The required C10.1 source preparation is also complete: frozen release verification; persistent one-attempt claim before held-out metadata/features; one cached feature load per clip and exactly one original adapter call for each of all four modes; original scoring/precision math; complete schema/cohort checks; atomic publication of the four prediction filenames; failure ledger with no retry; saved-output audit without re-inference; and a whitelisted local owner handoff excluding labels, weights, media, tensor caches and private cache paths. These programs are review-only at this stage and are never invoked by the joint job below.

**376 Member1 tests passed; 0 failures, 0 errors, 0 skipped.** Isolated `TEST_FIXTURE` tests exercise missing/stale approvals, source/hash/mode/threshold drift, owner acknowledgements, nonfinite evidence, single-attempt locks, cache errors/mutation, no partial publication and safe handoff contents. The synthetic freeze/held-out fixtures are not real project approvals or test runs.

## One upload and one execution

Upload only `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayC/member1/colab/AFML_DayC_B2_B3_Combined_Audit.zip` to `/content` in the **approved CPU runtime**. SHA-256 `9d4fe84cb29d1d73cd07f410147a2593a2d0b2897b3eaaa63d47ff2a4c1abd25`; 617193 bytes; 151 files. Manifest SHA-256 `7dbcd6502d6dfdfa1b4b075fd370034d7f569c4e37e27358c526c158b4501e2e`.

The capsule is self-contained for **saved-output evaluation**. It contains the immutable executed B2 source package, exact four validation prediction files and three benchmark files, frozen calibration, original validation-ready metadata, original canonical metadata, and current B3 source/test snapshots. The canonical manifest is copied byte-for-byte, including its existing train/val/test metadata, so Member2's unchanged hash/protocol/split audit works. It is not rebuilt or filtered and no `feature_path` is rewritten. Canonical metadata access remains part of Member2's B2 validation/split checks; B3 consumes only validation results. No ready-test manifest, test feature arrays, model weights or raw media are included or opened. Do not characterize the historical test-label exposure as completely untouched test information.

All physical metadata/score relocations are explicit read-only path bindings to the **same SHA-256 bytes**. The original executed package, manifests, context, prediction files, calibration and recorded provenance remain unchanged. No old Colab folders, mounted Drive, weights or tensor caches are needed. Keep the accepted runtime versions; the cell makes no installations, downloads, mounts or environment replacements. It rejects unapproved runtime drift.

The job runs unchanged Member2 checks/evaluation/paired interfaces on all 1200 clips, with the original fixed threshold and 1000 source-group bootstrap replicates, seed 0. It completes overall plus audio_fake-vs-real/video_fake-vs-real/both_fake-vs-real contrasts through the accepted DayB public bootstrap APIs. Category counts fluctuate within group resamples: these are analysis-stratified subsets, **not** a newly introduced fixed-category-count resampling algorithm. The exact method and known limitations remain visible for joint review.

It then prepares a data-backed B3 draft, without choosing a policy, signing owner releases or writing final configs. The draft is created under `execution_project/artifacts/dayC/member1/day9/<evaluation_run>_PROPOSAL/` and copied read-only into the returned ZIP. The source snapshots under `review_sources/` are for review; do not execute the held-out commands.

```python
from pathlib import Path
from google.colab import files
import hashlib, json, os, re, stat, subprocess, sys, zipfile

archive = Path('/content/AFML_DayC_B2_B3_Combined_Audit.zip')
addon = Path('/content/AFML_DayC_B2_B3_Combined_Audit')
assert archive.is_file(), 'Upload the combined audit ZIP into /content first'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '9d4fe84cb29d1d73cd07f410147a2593a2d0b2897b3eaaa63d47ff2a4c1abd25', 'Combined ZIP checksum mismatch'
assert not addon.exists(), 'Preserve any existing joint audit attempt; do not overwrite its directory'
with zipfile.ZipFile(archive) as z:
    names = z.namelist()
    assert z.testzip() is None and len(names) == len(set(names))
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
    assert {Path(n).parts[0] for n in names} == {addon.name}
    assert all(not stat.S_ISLNK(i.external_attr >> 16) for i in z.infolist())
    z.extractall('/content')
manifest = json.loads((addon/'JOINT_MANIFEST.json').read_text())
for name, expected in manifest['files'].items():
    assert not Path(name).is_absolute() and '..' not in Path(name).parts
    assert hashlib.sha256((addon/name).read_bytes()).hexdigest() == expected
project = addon/'execution_project'
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
    PYTHONPATH=str(project/'src') + os.pathsep + str(project),
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg', GIT_OPTIONAL_LOCKS='0')
print('JOINT B2/B3 AUDIT: saved-score statistics, 1000 source-bootstrap replicates, seed 0; then unapproved B3 draft.')
print('No Drive mount, checkpoint, cached tensors, model inference, benchmark rerun, final freeze or test scoring.')
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
    assert review.is_file() and review.is_relative_to(addon.resolve())
    with zipfile.ZipFile(review) as z:
        hashes = json.loads(z.read('output_hashes.json'))
        assert len(z.namelist()) == len(set(z.namelist()))
        assert set(z.namelist()) == set(hashes) | {'output_hashes.json'}
        for name, expected in hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
        if 'failure.json' in z.namelist():
            print('HALTED:', json.loads(z.read('failure.json')))
        else:
            complete = json.loads(z.read('evaluation_completion.json'))
            proposal = json.loads(z.read('day9_preparation/policy_proposal.json'))
            assert complete['n'] == 1200 and complete['n_boot'] == 1000 and complete['seed'] == 0
            assert proposal['status'] == 'PROPOSED_NOT_APPROVED'
            assert proposal['ready_for_freeze'] is False and proposal['selected_interpretation'] is None
            print('B2 statistics and B3 draft are ready for joint review. Missing real owner releases remain explicit.')
    print('RETURN_ZIP_SHA256:', hashlib.sha256(review.read_bytes()).hexdigest())
    files.download(str(review))
else:
    print('No return ZIP. Return this complete output; preserve the directory. Do not rerun inference or benchmarks.')
print('STOP: return DAYC_B2_B3_review.zip even on failure. No final freeze or held-out test authorization is inferred.')
```

Return **DAYC_B2_B3_review.zip**, including failure evidence if the job stops. It contains actual Member2 metrics/CIs/plots, per-stratum results, input/source/runtime provenance, the Member1 research packet, and B3 proposal/dependency/source-verification records. Future CI values and returned ZIP hashes are not invented here.

## Remaining genuine dependencies

The current local B3 draft has no chosen policy and no fabricated H2/H3 values while real B2 statistics are pending. The joint job fills only measured validation findings. Final freeze readiness still needs genuine Member2 final protocol/threshold/test-seal acknowledgement, Member3 benchmark/runtime acknowledgement, signed opaque ready-test manifest identity and full-count declaration, complete frozen input/source identity, and the lead's exact proposal-bound C9 approval. The known DayB threshold is not automatically final approval. No test rows need to be inspected to prepare those release identities.

Current held-out source enforces project-local approved cached-feature paths. No actual test-cache runtime/path arrangement is configured or silently inferred from validation access; that must be supplied and reviewed in the genuine freeze proposal before C10.

The original B2 inference and all 12 benchmark records are already verified. They are not rerun. Value/output extra p50 cost is nonpositive on all workloads; the planned recovery-per-positive-extra-ms comparison stays undefined. No positive damage, rescue, superiority, equivalence, production acceleration or final policy is asserted without the actual evidence and human review.

Local audit/test/draft evidence: `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayC/member1/day9/B3_SOURCE_20261112T161540549058Z`. All local changes remain in `/Users/sudarshan/Documents/GitHub/AFML-project`; no Git commits, pushes, merges, branch changes or remote writes were made. The prior automatic approval-review usage failure was resolved through the normal approval path; its rejected command was not executed or bypassed.
