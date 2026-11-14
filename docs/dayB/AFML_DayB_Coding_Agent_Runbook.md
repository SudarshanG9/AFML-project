# AVFuse-CausalQ — Day B Coding-Agent Execution Runbook

**Day B = original Day 5 + Day 6 + Day 7, completed within one coordinated execution day, with no reduction of the original scientific requirements.**

**Target repository:** `SudarshanG9/AFML-project`  
**Working branch for Member 1:** `member1`  
**Repository snapshot inspected:** `member1` at `4e0235184136ff89a8d9282fdaffdbe423b2b64c`; `main` at `401836f4f437383c6de5fa3b105614bfb97e7236`; `feature/member2-data` at `1f72bfac86ebf233fd0e4f535360a929c1484114`; `feature/member3-runtime` at `9d3b499dce7cb390cf1d95966d5f718abaee6051`. Reconfirm the actual **local** worktree before changing anything.  
**Source plan:** *AFML Final.pdf*, “AVFuse-CausalQ — 12-Day Execution Plan,” especially the Day 5–7 timeline, role-owned task lists, dependency table, and scientific constraints.  
**Authoritative implementation:** files actually present in the checked-out repository; prefer inspection and reuse to blind replacement.

> **TO THE CODING AGENT — READ FIRST.** This is an execution contract, not permission to perform every phase in one uninterrupted autonomous run. Execute **only the phase explicitly approved by the user**. Stop at each **HARD GATE**, display results and request the user to upload/inspect them. Continue **only after the user explicitly acknowledges and authorizes the next phase**. Never infer approval from a test passing, a timeout, an absent reply, a short “working” status, or another teammate's note. If blocked, report the exact blocker and stop. The user, not the agent, interprets experimental findings.

## 0. Non-negotiable operational rules

1. **LOCAL-ONLY EDITS.** Do not run `git add`, `git commit`, `git push`, `git merge`, `git rebase`, `git reset --hard`, `git clean`, `gh pr create`, or GitHub write APIs. Do not alter any branch on GitHub. Never open or update a PR. Do not automatically commit after passing tests. A local change may remain uncommitted at the end.
2. **Never destroy work.** Record pre-existing modified/untracked files. Never overwrite or delete existing code, cache, checkpoint, reports, or predictions unless the user explicitly approves that exact change. Write new outputs under uniquely named local paths. Preserve Member 2 and Member 3 ownership. Do not auto-install from an untrusted source.
3. **No scaling down.** A 10-clip or one-clip pilot is a development gate, **not** a substitute for the plan's full validation pass. Day B is *scientifically complete* only when all **1,200 validation videos** have reference, low-precision, and value/output-restored predictions, Member 2's corresponding statistics exist, and Member 3's actual-device benchmark evidence exists. Full remote validation on the already-extracted shared Drive cache is allowed and preferred over downloading ~4 GB to the Mac. No synthetic metrics may masquerade as research results.
4. **Test remains sealed.** Never run inference over the 2,000 held-out test clips, view their scores, calibrate a threshold on test, or tune from test outcomes. Day 9 freezes protocol; Day 10 performs final test inference. Read only full-manifest **metadata** when auditing val/test group separation; do not load test feature tensors.
5. **One fixed detector.** Use the approved pretrained DiMoDif LAV-DF checkpoint, SHA-256 `e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e`; keep its weights and detector head frozen, eval mode, and full FP32 outside instrumented attention. No model retraining or new classifier.
6. **Four named configurations exist**, but **Day 8 query-control evaluation is outside the original Days 5–7 scope**. The code may smoke-test the existing `restore_query` mode now; do not claim the Day 8 statistical comparison is completed by Day B. Do not substitute a query-control result for the Day 7 restoration test.
7. **Precision is simulated INT8** (float tensor, quantize-dequantize). Report no INT8 storage speedup, real integer kernels, quantized weights, or physically equal-cost restoration unless actually measured.
8. **Interpretation is a mandatory human gate.** On model parity, data integrity, score orientation, low-precision degradation, matched-upstream attribution, restoration recovery, and cost differences, present the evidence and **STOP**. Never interpret a sign or statistical finding as a reason to silently change the research protocol.
9. **Use tracked code as the baseline.** Read each named file before editing. Many functions already exist; add only missing tests, missing integration glue, or a demonstrably necessary fix. If a code block below duplicates a valid implementation, **do not duplicate it**. Document it as “already implemented; verified in …”.
10. **No unrequested scope creep.** No new pretrained architecture, custom FlashAttention kernel, new quantization scheme, revised manifest schema, alternative threshold-selection method, branch merges, or production UI.
11. **Every terminal action is auditable.** Capture commands, exit status, traceback (if any), file paths, SHA-256 hashes, device, Torch version, and what was proved versus merely smoke-tested. Prefer `python -m pytest` and the repo's current environment. Do not claim tests passed unless they actually passed.
12. **No network changes to the repository.** The agent may read the local repository. If a team Drive/Colab environment is essential, request the user to provide or mount access and upload the current *local working tree* by a user-approved transfer method—**not** by pushing uncommitted work to GitHub.

### Required wording at EVERY hard gate

```
HARD GATE <ID> — STOPPED; AWAITING USER ACKNOWLEDGEMENT
What I inspected/implemented:
Exact files changed (or NONE):
Commands run and exit status:
Pass/fail evidence, including hashes and counts:
Scientific interpretation required:
Known limitations or unresolved blockers:
Artifacts/logs for the user to upload:
My next proposed action (NOT yet executed):
Please upload/review the results and explicitly say "ACK GATE <ID> — PROCEED".
```

Do not run the next gate's first command until that exact acknowledgement (or another unambiguous explicit approval) is received. If the user changes instructions, follow their newest request while preserving the local-only, no-test-leakage safety rules.

---

## 1. What Day B combines — **exact duties from the original plan**

| Original day | Member 1 / Lead | Member 2 / data & evaluation | Member 3 / runtime | Required outcome |
|---|---|---|---|---|
| **Day 5 — FP32 baseline + precision wiring** | Implement symmetric per-tensor fake-INT8 function; instrument attention Q/K/V and its output only; confirm reference path matches original pretrained detector | Run FP32 on complete cached validation data; create reference prediction file; compute baseline AUROC/AUPRC, and validation-calibrated recall | Build real cached-feature **model-level** FP32 latency and memory benchmark | Full reference predictions, baseline forensic metrics and honest model-level timings |
| **Day 6 — Low-precision consequences (H2)** | Execute the *same* frozen detector and dataset with low-precision attention; produce predictions in exactly the same schema | Compute low-precision AUROC, AUPRC, and recall by manipulation type against the reference; source-group CIs and paired contrasts | Benchmark low precision versus reference on the **same device/workload** | First real validation-based answer to whether attention fake quantization damages forensic correctness |
| **Day 7 — Targeted restoration + integration** | Implement/verify value-and-output precision restoration while Q/K remain fake-quantized; inspect what “same low-precision upstream” actually means for five sequential joint-attention layers | Provide a **validation-only**, paired, label-grounded recovery harness; do not inspect test | Extend benchmark handling to mixed-precision configurations and **include conversion overhead** in measured cost | First real joined restoration run and paired recovery evidence; any scientific ambiguity presented to the lead |

### Boundary of Day B

- **Day 8 (later):** query-path control on full validation, cost comparison to restoration. Its current code exists and can be checked, but full Day 8 claims are not part of this runbook unless the user separately authorizes them.
- **Day 9 (later):** policy/threshold/protocol freeze based on validation evidence.
- **Day 10 (later):** one final untouched held-out test run.
- The original plan assumes a cross-modal fusion sub-operation, but the selected pretrained DiMoDif checkpoint uses **joint audio/video self-attention**: audio↔audio, video↔video, and audio↔video attention occur in the joint sequence. The implemented patch affects **all five joint-attention layers**, not only audio→video edges. Do not call it “pure cross-attention” or a uniquely isolated cross-modal edge ablation. This is a critical accuracy-of-claims requirement.

### Frozen precision semantics

| Mode | Q | K | V | Attention output (before original `out_proj`) |
|---|---|---|---|---|
| `reference` | FP32 | FP32 | FP32 | FP32 |
| `low_precision` | fake INT8 | fake INT8 | fake INT8 | fake INT8 |
| `restore_value_output` | fake INT8 | fake INT8 | FP32 | FP32 |
| `restore_query` (Day 8 control) | FP32 | fake INT8 | fake INT8 | fake INT8 |

`fake_quant_int8(x) = clamp(round(x / (max(abs(x))/127)), -127, 127) * (max(abs(x))/127)`, with numerical safeguards for zero tensors. **Weights, dense projections, residual paths, layer norms, encoders and detector head are not quantized by this protocol.**

### Labels/metrics

- Full metadata: `data/manifests/manifest_whole.jsonl` (3,200 whole videos, validation 1,200 and sealed test 2,000).
- Validation-only metadata: `data/manifests/manifest_whole_val.jsonl` (300 each: real, audio_fake, video_fake, both_fake).
- Target dataset: LAV-DF whole videos, not the earlier 4-second clip manifest.
- For every fake manipulation, calculate against **the real class** (real vs audio_fake; real vs video_fake; real vs both_fake). Source-group/identity-group cluster bootstrap for uncertainty.
- Fix the threshold from **reference validation scores** at 5% real FPR, and apply that identical threshold when comparing low precision and restored configurations. Do **not** recompute a threshold per mode.
- Use the repository's existing prediction schema: `run_id`, `clip_id`, `split`, `config`, `score_fake`, `checkpoint_sha256`, with no label in prediction rows.

---

## 2. Repository audit — source of truth before editing

### Branch-specific facts verified by read-only GitHub inspection

| File/feature | `main` | `member1` | Member 2 feature branch | Member 3 feature branch |
|---|---|---|---|---|
| `scripts/day6_preflight.py` | absent | **present** | absent | absent |
| `scripts/dayB_inference.py` | absent | **present** | absent | absent |
| Quantization/attention adapter | earlier baseline | **Day B implementation** | earlier state | earlier state |
| Whole-video pilot cache | present | **present** | not committed | not committed |
| Full 1,200-video validation cache | not in GitHub | not in GitHub | not in GitHub | not in GitHub |
| `dayB_val_ready.jsonl` | not in GitHub | not in GitHub | not in GitHub | not in GitHub |

`member1` is the implementation branch to work on. The other branches are NOT alternative locations of the full cache. The team's Drive holds the feature tensors; `data/` and `artifacts/` are git-ignored. Check whether local untracked files already exist before writing.

### Reuse these existing files; do not replace them wholesale

**Member 1:**

```
src/avfuse/member1_fusion/precision.py              # fake_quant_int8(), prepare_qkv(), quantize_output()
src/avfuse/member1_fusion/attention.py              # dense_attention(), sdpa_attention(), fusion_attention()
src/avfuse/member1_fusion/dimodif_intervention.py   # temporary five-layer MHA patches
src/avfuse/member1_fusion/dimodif_adapter.py        # original frozen DiMoDif, outputs score_fake
scripts/dayB_inference.py                          # --features pilot vs --manifest validation
scripts/day6_preflight.py                          # full 1,200 validation cache audit
scripts/dayA_model_parity.py                       # original-vs-instrumented real model parity
scripts/dayA_verify_handoff.py                     # real Q/K/V oracle/SDPA checks
tests/member1/test_precision.py                    # already has manual low-precision test
tests/member1/test_attention.py                    # attention correctness tests
tests/member1/test_dimodif_intervention.py         # toy-model integration and cleanup tests
tests/member1/test_day6_preflight.py               # synthetic preflight error cases
tests/member1/day6_integration_test.py             # executable real-checkpoint integration; not pytest auto-discovered
```

**Member 2:**

```
src/avfuse/member2_data/manifest.py                # read_manifest, verify_manifest, write_manifest
src/avfuse/member2_data/splits.py                  # split_audit, group_clusters
src/avfuse/member2_data/predictions.py             # validate_predictions, join, read_predictions
src/avfuse/member2_data/evaluate.py                # AUROC/AUPRC/BCE/threshold/cluster bootstrap
src/avfuse/member2_data/paired.py                  # paired damage/recovery, measured R per ms
scripts/validate_manifest.py
scripts/m2_check_predictions.py
scripts/m2_evaluate.py
scripts/m2_paired.py
scripts/m2_shortcut_control.py
```

**Member 3:**

```
src/avfuse/member3_runtime/benchmark.py            # benchmark_callable() sync/warmup/p50/p95/peak memory
src/avfuse/member3_runtime/operator_benchmark.py   # benchmark_attention_operator()
src/avfuse/member3_runtime/artifacts.py            # build_benchmark_record(), schema validation/writer
src/avfuse/member3_runtime/environment.py          # collect_environment_report()
src/avfuse/member3_runtime/provenance.py           # checkpoint/cache hashes
scripts/m3_day3_verify_stage_profile.py
scripts/m3_day4_verify_operator_sweep.py
```

**Avoid trusting older comments blindly:** root `README.md`, `data/README.md`, and some Member 2 docs still describe an earlier `.pt` cache. The **actual selected DiMoDif path** uses `video_features` and `audio_features` in `features.npz` with 768 channels, and `scripts/dayB_inference.py` maps them to `video_tokens`/`audio_tokens` internally. Do not convert all 1,200 caches to `.pt`.

### The critical missing artifact and the prior error

`data/manifests/dayB_val_ready.jsonl` is **generated by** a successful full-cache invocation of `scripts/day6_preflight.py`; it is **not** a committed file. If a command fails with `FileNotFoundError` for it, **do not create an empty placeholder**, misname the pilot manifest, or rerun full inference against nonexistent feature files.

- Pilot route: `--features data/pilot/day2_features/features1.npz` — **no manifest needed**.
- Engineering mini-batch route: use a separately named `data/manifests/pilot_available_val.jsonl` containing only available committed caches; **not a scientific final result**.
- Full scientific validation route: after authentic preflight on the complete Drive cache, use `--manifest data/manifests/dayB_val_ready.jsonl` in the environment where it exists.

### Scope of available pilot

`data/pilot/test_day2/manifest_rows.jsonl` contains **20** metadata rows (five per category); **only 10 of them have committed `features.npz` files** (two real, three audio_fake, two video_fake, three both_fake). Build a 10-row pilot from those 10 only and label the output `PILOT_ONLY`. Never silently drop clips in a full-validation run.

---

## 3. Mandatory gate sequence (NO automatic transitions)

| Gate | Work | Can run on local Mac without full Drive cache? | Stops for user interpretation? |
|---|---|---|---|
| **G0** | Worktree/branch audit, 12-day-plan and current code crosswalk, existing outputs inventory | yes | **YES** |
| **G1** | Received manifest integrity, training-overlap/extraction-provenance review, cache/checkpoint availability | yes (metadata only) | **YES** |
| **G2** | Day 5 fake-INT8 mathematics, deterministic operator tests, real-model parity integration | yes, if checkpoint exists | **YES** |
| **G3** | FP32/lowp/restoration **pilot** on one clip + ten available caches and schema checks | yes, if checkpoint exists | **YES** |
| **G4** | Remote **full** 1,200 validation cache preflight, immutable ready manifest | **no**, access the team Drive/Colab cache | **YES** |
| **G5** | Day 5 full 1,200 reference predictions, Member 2 baseline metrics, Member 3 FP32 benchmark | remote | **YES** |
| **G6** | Day 6 full 1,200 low-precision predictions, H2 metrics and measured runtime | remote | **YES** |
| **G7** | Day 7 targeted V/output math, upstream-control audit, restoration tests | locally for code/tests | **YES** |
| **G8** | Day 7 full restored 1,200 validation predictions, paired R, mixed-mode cost | remote | **YES** |
| **G9** | Verify all Day B deliverables, compare hashes, identify remaining Day 8/9 tasks, final read-only status | local/remote reports | **YES** |

The agent must **begin at G0** and STOP after G0, even though the entire plan is supplied below. The code snippets are specifications for **future, separately approved phases**.

---

## G0 — Worktree snapshot, local safety, file-by-file gap check

### Commands — from repository root

```bash
git branch --show-current
git rev-parse HEAD
git status --short
git branch -a
python --version
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('CPU available: yes')
print('CUDA:', torch.cuda.is_available())
print('MPS:', torch.backends.mps.is_available())
PY
python -m pip show avfuse-causalq
```

`git status` is **read only**, allowed. If branch is not `member1`, **stop and ask** before switching, especially if the worktree contains changes. The agent cannot discard, stash, or merge user work implicitly. Note that local files may differ from the inspected remote snapshot.

```bash
ls -l scripts/dayB_inference.py scripts/day6_preflight.py \
  src/avfuse/member1_fusion/precision.py \
  src/avfuse/member1_fusion/dimodif_intervention.py
ls -l data/pilot/day2_features/features1.npz
ls -l data/pilot/test_day2/manifest_rows.jsonl
ls -l third_party/dimodif/src/models.py
ls -l third_party/dimodif/ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.pth
ls -l third_party/dimodif/ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.json
```

If a checkpoint/model/config is missing, report it explicitly. The file is NOT available on GitHub and cannot be fabricated. Do not try to run the model with random weights.

Read all source paths listed in Section 2, plus `contracts/schemas/{manifest,prediction,benchmark}.schema.json` and `docs/EXPERIMENT_PROTOCOL.md`. For each function/test, write an audit entry: `implemented`, `implemented but needs test`, `missing`, or `broken with evidence`. Do not regenerate an already-correct file.

**HARD GATE G0.** Present: local branch/commit, dirty files, checkpoint presence, exact remaining code gaps, paths/hashes, proposed G1 actions. **STOP AND AWAIT ACK G0.**

---

## G1 — Validate Member 2 handoff without downloading the ~4 GB feature archive

### Received handoff layout (local-only recommendation)

```
data/manifests/manifest_whole.jsonl
data/manifests/manifest_whole_val.jsonl
data/manifests/manifest_whole_report.json
data/manifests/videos_whole.csv
data/member2_handoff/README.md
data/member2_handoff/extraction_provenance.json
data/member2_handoff/feature_coverage_report.json
data/member2_handoff/feature_report.json
data/member2_handoff/split_integrity.json
data/member2_handoff/train_overlap.json
data/member2_handoff/video_features.csv
```

These eleven files were **reported as received**, but their bytes were not present in the repository audit. Inspect the paths where the user actually placed them. Do not assert their contents pass until inspected. Never replace the repo-root `README.md` with the handoff `README.md`.

### Existing validators — no new code required

```bash
python scripts/validate_manifest.py data/manifests/manifest_whole.jsonl
python scripts/validate_manifest.py data/manifests/manifest_whole_val.jsonl
```

### One-off cross-manifest check (run without creating an extra source file)

```bash
python - <<'PY'
from collections import Counter
from avfuse.member2_data.manifest import read_manifest
from avfuse.member2_data.splits import split_audit

whole = read_manifest('data/manifests/manifest_whole.jsonl')
val = read_manifest('data/manifests/manifest_whole_val.jsonl')
assert len(whole) == 3200, len(whole)
assert Counter(r['split'] for r in whole) == {'val': 1200, 'test': 2000}
assert len(val) == 1200 and all(r['split'] == 'val' for r in val)
assert Counter(r['manipulation_type'] for r in val) == {
    'real': 300, 'audio_fake': 300, 'video_fake': 300, 'both_fake': 300
}
original_val = {r['clip_id']: r for r in whole if r['split'] == 'val'}
received_val = {r['clip_id']: r for r in val}
assert original_val.keys() == received_val.keys()
for clip_id, r in received_val.items():
    for key in ('dataset', 'source_id', 'identity_id', 'split', 'label', 'manipulation_type'):
        assert r[key] == original_val[clip_id][key], (clip_id, key)
audit = split_audit(whole)
assert audit['passed'] and audit['groups_crossing_splits'] == 0, audit
print('FULL MANIFEST AND VAL SUBSET: PASS')
print('counts:', audit['counts'])
PY
```

Inspect, **do not auto-repair**, the received `split_integrity.json`, `train_overlap.json`, `extraction_provenance.json`, `feature_coverage_report.json`, `feature_report.json`. Check that names, hashes, encoder identity, reported 3,200/3,200 coverage and a no-unacceptable-overlap conclusion are actually supported. The checkpoint was trained on LAV-DF train and selected on dev, so **explicitly disclose validation-selection bias** even if the train-overlap report says clean. Read-only metadata audit may inspect the whole manifest, **not test embeddings**.

### The cache-root contract (for later remote run)

`day6_preflight.py` requires:

```
<CACHE_ROOT>/val/<clip_id>/features.npz
<CACHE_ROOT>/val/<clip_id>/meta.json
```

Per `.npz`: `video_features` and `audio_features`, arrays `[T_v,768]` and `[T_a,768]`, float32, finite. It verifies per-clip metadata, consistency, cache hashes and approved checkpoint SHA. It uses only `val/`. The source documentation says the full cache is on the team Drive at a placeholder path `<drive>/dimodif_emb`; **the exact Drive filesystem path cannot be known from the GitHub repository**. Do not guess it.

### File preservation check

```bash
python - <<'PY'
from pathlib import Path
for p in (
    'data/manifests/manifest_whole.jsonl',
    'data/manifests/manifest_whole_val.jsonl',
    'data/manifests/dayB_val_ready.jsonl',
):
    print(p, 'EXISTS' if Path(p).is_file() else 'NOT PRESENT')
PY
```

`dayB_val_ready.jsonl` is expected **not** to exist locally if full preflight was deferred. Do not create it by copying `manifest_whole_val.jsonl`; its paths still point at Member 2's old `.pt` placeholders. Do not pass that placeholder manifest to `dayB_inference.py` for real inference.

**HARD GATE G1.** Give a verified manifest/hash/audit table, any mismatch, checkpoint availability, and full-cache-access status. Ask for the user's review of the audit artifacts. **STOP AND AWAIT ACK G1.**

---

## G2 — Day 5 implementation: verify fake INT8, operator isolation, unchanged pretrained model

### Files to inspect before modification

- `src/avfuse/member1_fusion/precision.py`: `fake_quant_int8`, `prepare_qkv`, `quantize_output` already implemented.
- `src/avfuse/member1_fusion/attention.py`: SDPA and dense oracle already implemented.
- `src/avfuse/member1_fusion/dimodif_intervention.py`: projects Q/K/V from the original packed MHA weights and uses the original output projection. Five joint attention blocks are intercepted.
- `src/avfuse/member1_fusion/dimodif_adapter.py`: frozen DiMoDif and original score aggregation already implemented.
- `scripts/dayB_inference.py`: current `create_adapter()` rejects unapproved checkpoint SHA and non-frozen/training model; verify locally, do not add duplicate assertions if present.
- `tests/member1/test_precision.py`: **`test_manual_low_precision_attention` already exists on the inspected `member1` branch**; run it instead of duplicating it.

### Existing core implementation reference — **inspect, do not duplicate**

The checked-out `src/avfuse/member1_fusion/precision.py` should already contain the following function-level behavior. Use this as a comparison target; only make a minimal correction if actual code differs **and** a failing test demonstrates a defect. Keep the existing public function names, signature and call sites.

```python
# src/avfuse/member1_fusion/precision.py — EXISTING behavior, not a replacement file
import torch


def fake_quant_int8(x):
    if not x.is_floating_point():
        raise TypeError('fake quant expects floating point input')
    max_abs = x.detach().abs().amax()
    scale = max_abs.clamp_min(torch.finfo(x.dtype).tiny) / 127.0
    return (x / scale).round().clamp(-127.0, 127.0) * scale


def prepare_qkv(q, k, v, mode):
    if mode == 'reference':
        return q, k, v
    if mode == 'low_precision':
        return fake_quant_int8(q), fake_quant_int8(k), fake_quant_int8(v)
    if mode == 'restore_value_output':
        return fake_quant_int8(q), fake_quant_int8(k), v
    if mode == 'restore_query':
        return q, fake_quant_int8(k), fake_quant_int8(v)
    raise ValueError(f'Unsupported mode: {mode}')


def quantize_output(out, mode):
    return fake_quant_int8(out) if mode in ('low_precision', 'restore_query') else out
```

Note: the real file already enforces `PRECISION_MODES` in `prepare_qkv()`, and contains diagnostic helpers. **Do not replace it with this abbreviated reference block.** `attention.py` calls `prepare_qkv()`, then the tested dense or SDPA attention, then `quantize_output()` before the original `out_proj`; the surrounding Transformer remains frozen and FP32.

### Mathematical invariants to test

1. `reference`: Q, K, V and output unchanged at fake-quant boundaries.
2. `low_precision`: all Q/K/V + output equal explicit fake-quantization applied to SDPA.
3. No mutation of input tensors; no changes to frozen model state dictionary.
4. Mask semantics: original MHA boolean `True=blocked`, converted to additive SDPA `-inf` for blocked positions; do not accidentally use the SDPA boolean `True=permitted` convention when entering the MHA wrapper.
5. Source tensors retain FP32 dtype; simulated INT8 does not create an actual 8-bit kernel.
6. `reference` instrumented forward equals original MHA logits and `score_fake` within pre-agreed tolerances.
7. Temporary monkeypatch restores original attention methods on success **and exception**.

### New file (only if not already covered): `tests/member1/test_dayB_mode_math.py`

Create the following test module. These are **independent expected-value constructions**, not tests that compare a function to itself.

```python
import torch

from avfuse.member1_fusion.attention import fusion_attention, sdpa_attention
from avfuse.member1_fusion.precision import fake_quant_int8, prepare_qkv


def _inputs():
    g = torch.Generator().manual_seed(20261111)
    return (
        torch.randn(1, 2, 7, 8, generator=g),
        torch.randn(1, 2, 7, 8, generator=g),
        torch.randn(1, 2, 7, 8, generator=g),
    )


def test_low_precision_matches_manual_four_boundaries():
    q, k, v = _inputs()
    expected = fake_quant_int8(sdpa_attention(
        fake_quant_int8(q), fake_quant_int8(k), fake_quant_int8(v)
    ))
    got = fusion_attention(q, k, v, mode='low_precision', backend='sdpa')
    torch.testing.assert_close(got, expected, atol=0, rtol=0)
    assert got.dtype == torch.float32


def test_restored_value_output_equals_manual_targeted_math():
    q, k, v = _inputs()
    expected = sdpa_attention(fake_quant_int8(q), fake_quant_int8(k), v)
    got = fusion_attention(q, k, v, mode='restore_value_output', backend='sdpa')
    torch.testing.assert_close(got, expected, atol=0, rtol=0)
    qq, kk, vv = prepare_qkv(q, k, v, 'restore_value_output')
    torch.testing.assert_close(qq, fake_quant_int8(q), atol=0, rtol=0)
    torch.testing.assert_close(kk, fake_quant_int8(k), atol=0, rtol=0)
    assert torch.equal(vv, v)


def test_query_control_remains_independent():
    q, k, v = _inputs()
    expected = fake_quant_int8(sdpa_attention(
        q, fake_quant_int8(k), fake_quant_int8(v)
    ))
    got = fusion_attention(q, k, v, mode='restore_query', backend='sdpa')
    torch.testing.assert_close(got, expected, atol=0, rtol=0)


def test_reference_retains_original_operands_and_output():
    q, k, v = _inputs()
    snapshots = [x.clone() for x in (q, k, v)]
    actual = fusion_attention(q, k, v, mode='reference', backend='sdpa')
    expected = sdpa_attention(q, k, v)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert all(torch.equal(a, b) for a, b in zip((q, k, v), snapshots))


def test_quantizer_is_numerical_not_integer_kernel():
    x = torch.tensor([[-1.75, -0.34, 0.0, 0.49, 2.1]], dtype=torch.float32)
    q = fake_quant_int8(x)
    assert q.dtype == x.dtype
    assert q.shape == x.shape
    assert torch.isfinite(q).all()
    assert torch.equal(fake_quant_int8(torch.zeros_like(x)), torch.zeros_like(x))
```

Tests (verify existing ones too):

```bash
python -m pytest -q tests/member1/test_precision.py tests/member1/test_dayB_mode_math.py
python -m pytest -q tests/member1/test_attention.py tests/member1/test_dimodif_intervention.py
python -m pytest -q tests/member1/test_day6_preflight.py
```

If new tests fail while existing tests pass, do **not** weaken expected values or silently change tolerances. Diagnose and present a minimal implementation patch with a concrete reason; request user authorization before changing original model semantics.

### Real model parity on one committed pilot

```bash
python scripts/dayB_inference.py \
  --features data/pilot/day2_features/features1.npz \
  --mode reference --device cpu \
  --trace-out artifacts/dayB/reference_trace.json
```

Expected **messages, not guessed numerical scores**:

```
FULL-MODEL FP32 PARITY: PASS
Instrumented layers: 5
Diagnostic trace: artifacts/dayB/reference_trace.json
```

Then run the existing executable integration test **directly** (it is named `day6_integration_test.py`, not `test_*.py`, so default pytest discovery does not collect it):

```bash
python tests/member1/day6_integration_test.py \
  --feature data/pilot/day2_features/features1.npz \
  --device cpu \
  --output artifacts/day6/integration_report.json
```

Inspect trace record count (5 layers × Q,K,V,output = 20), per-layer shapes, `fraction_changed`, `max_abs_error`, original/instrumented score and model logit parity. For reference, the quantized fraction should be zero at the defined intervention boundaries. If backend-specific precision tolerances fail on CPU/MPS/CUDA, collect errors; do not assume switching devices proves identical numeric parity.

**HARD GATE G2.** Present all pytest results, the real-model parity and integration reports, checkpoint SHA, attention trace table, and any anomalies. **STOP AND AWAIT ACK G2.**

---
## G3 — Create a distinct 10-clip pilot and exercise the entire prediction contract

**Purpose:** establish a reliable local engineering integration checkpoint without misrepresenting the incomplete pilot as the full 1,200-clip Day 5/6/7 dataset. This is an *additional test*, not a replacement for G4–G8.

### New file: `scripts/dayB_prepare_pilot_manifest.py`

Implement exactly once; do not edit `manifest_whole*.jsonl`. The output is a distinct test artifact. This script intentionally permits missing **pilot** clips, but no scientific full-validation runner may do so.

```python
"""Create a pilot-only manifest for committed, existing NPZ caches.

Never use this output in place of the 1,200-clip scientific validation manifest.
"""
import argparse
from collections import Counter
from pathlib import Path

from avfuse.member2_data.manifest import read_manifest, write_manifest


def build_pilot_manifest(source: Path, cache_root: Path, output: Path) -> dict:
    source = Path(source)
    cache_root = Path(cache_root).resolve()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    rows = read_manifest(source)
    if any(r['split'] != 'val' for r in rows):
        raise ValueError('Pilot source must be validation-only')
    selected = []
    missing = []
    for r in rows:
        path = (cache_root / 'val' / r['clip_id'] / 'features.npz').resolve()
        if not path.is_relative_to(cache_root):
            raise ValueError('Path escaped the pilot cache root')
        if path.is_file():
            item = dict(r)
            item['feature_path'] = str(path)
            selected.append(item)
        else:
            missing.append(r['clip_id'])
    if not selected:
        raise ValueError('No actual NPZ pilot caches exist')
    output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(selected, output)
    return {
        'status': 'PILOT_ONLY_NOT_FULL_VALIDATION',
        'source_records': len(rows),
        'available_records': len(selected),
        'missing_records': len(missing),
        'per_type': dict(Counter(r['manipulation_type'] for r in selected)),
        'output': str(output.resolve()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path,
        default=Path('data/pilot/test_day2/manifest_rows.jsonl'))
    p.add_argument('--cache-root', type=Path,
        default=Path('data/pilot/test_day2'))
    p.add_argument('--out', type=Path,
        default=Path('data/manifests/pilot_available_val.jsonl'))
    a = p.parse_args()
    print(build_pilot_manifest(a.source, a.cache_root, a.out))


if __name__ == '__main__':
    main()
```

### New test: `tests/member1/test_dayB_pilot_manifest.py`

```python
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.dayB_prepare_pilot_manifest import build_pilot_manifest
from avfuse.member2_data.manifest import read_manifest, write_manifest


def _fixture(tmp_path: Path):
    root = tmp_path / 'pilot'
    source = tmp_path / 'source.jsonl'
    rows = []
    categories = ('real', 'audio_fake', 'video_fake', 'both_fake')
    for i, category in enumerate(categories):
        clip_id = f'{i + 1:016x}'
        rows.append({
            'clip_id': clip_id, 'dataset': 'lavdf',
            'source_id': f'src{i}', 'identity_id': f'src{i}',
            'split': 'val', 'label': 0 if i == 0 else 1,
            'manipulation_type': category,
            'feature_path': f'data/cache/{clip_id}.pt',
        })
        # Leave one pilot clip deliberately absent.
        if i != 2:
            directory = root / 'val' / clip_id
            directory.mkdir(parents=True)
            np.savez(directory / 'features.npz',
                video_features=np.zeros((5, 768), dtype=np.float32),
                audio_features=np.zeros((6, 768), dtype=np.float32))
    write_manifest(rows, source)
    return source, root


def test_pilot_lists_existing_only_without_changing_labels(tmp_path):
    source, root = _fixture(tmp_path)
    output = tmp_path / 'pilot_ready.jsonl'
    original = source.read_bytes()
    report = build_pilot_manifest(source, root, output)
    assert report['available_records'] == 3
    assert report['missing_records'] == 1
    assert report['status'] == 'PILOT_ONLY_NOT_FULL_VALIDATION'
    rows = read_manifest(output)
    assert len(rows) == 3
    assert all(Path(r['feature_path']).is_file() for r in rows)
    assert all(r['split'] == 'val' for r in rows)
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        build_pilot_manifest(source, root, output)


def test_pilot_rejects_nonvalidation_manifest(tmp_path):
    source, root = _fixture(tmp_path)
    rows = read_manifest(source)
    rows[0]['split'] = 'test'
    write_manifest(rows, source)
    with pytest.raises(ValueError, match='validation-only'):
        build_pilot_manifest(source, root, tmp_path / 'out.jsonl')
```

Run:

```bash
python -m pytest -q tests/member1/test_dayB_pilot_manifest.py
python scripts/dayB_prepare_pilot_manifest.py \
  --source data/pilot/test_day2/manifest_rows.jsonl \
  --cache-root data/pilot/test_day2 \
  --out data/manifests/pilot_available_val.jsonl
python scripts/validate_manifest.py data/manifests/pilot_available_val.jsonl
```

Before running, **check whether the output already exists**. If so, verify its source, SHA and correctness; don't overwrite it. On the inspected remote snapshot the expected count is 10 (2 real/3 audio_fake/2 video_fake/3 both_fake), but treat any local difference as an inspection gate, not as permission to ignore missing data.

### New test: `tests/member1/test_dayB_prediction_contract.py`

Unit-test Member 1's existing `run_validation()` using a fake adapter and a manifest with exact cache paths. No model download required. The fake adapter is permitted only inside a **unit test**, never to generate research predictions.

```python
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from avfuse.member2_data.manifest import write_manifest
from avfuse.member2_data.predictions import validate_predictions
from avfuse.shared.jsonl import read_jsonl
from scripts.dayB_inference import run_validation


class StubAdapter:
    checkpoint_sha256 = 'a' * 64

    def predict_from_cached_features(self, features, mode):
        assert features['video_tokens'].shape == (5, 768)
        assert features['audio_tokens'].shape == (6, 768)
        assert mode == 'reference'
        return torch.tensor([0.625], dtype=torch.float32)


def test_manifest_inference_writes_exact_schema_and_one_row(tmp_path):
    path = tmp_path / 'features.npz'
    np.savez(path,
        video_features=np.zeros((5, 768), dtype=np.float32),
        audio_features=np.zeros((6, 768), dtype=np.float32))
    manifest = tmp_path / 'manifest.jsonl'
    write_manifest([{
        'clip_id': '0000000000000001', 'dataset': 'lavdf',
        'source_id': 'source1', 'identity_id': 'source1',
        'split': 'val', 'label': 1, 'manipulation_type': 'audio_fake',
        'feature_path': str(path),
    }], manifest)
    output = tmp_path / 'predictions.jsonl'
    args = SimpleNamespace(manifest=str(manifest), mode='reference',
                           repo_root=str(tmp_path), run_id='unit-test',
                           out=str(output))
    run_validation(StubAdapter(), args)
    rows = read_jsonl(output)
    validate_predictions(rows)
    assert len(rows) == 1
    assert rows[0]['config'] == 'reference'
    assert rows[0]['clip_id'] == '0000000000000001'
    assert rows[0]['score_fake'] == pytest.approx(0.625)
    assert 'label' not in rows[0]
    assert rows[0]['checkpoint_sha256'] == 'a' * 64


def test_missing_manifest_is_a_real_error(tmp_path):
    args = SimpleNamespace(manifest=str(tmp_path / 'dayB_val_ready.jsonl'),
                           mode='reference', repo_root=str(tmp_path),
                           run_id='missing', out=str(tmp_path / 'out.jsonl'))
    with pytest.raises(FileNotFoundError):
        run_validation(StubAdapter(), args)
```

Run:

```bash
python -m pytest -q tests/member1/test_dayB_prediction_contract.py
```

### Small verified CLI guard issue: `scripts/dayB_inference.py`

In the inspected `member1` source, `main()` checks `Path(args.out).exists()` **before** checking whether `--out` was supplied. With `--manifest` but no `--out`, this can cause an unexpected `TypeError` instead of a clean argument error. This is not the cause of the user's missing `dayB_val_ready.jsonl`; it is a separate small reliability fix. **Only if the local copy still has this ordering**, surgically reorder these lines in `scripts/dayB_inference.py` (preserve the rest of the code):

```python
# Inside main(), immediately after args = parser.parse_args()
if args.manifest and not args.out:
    parser.error('--out is required for manifest inference')

if args.manifest and Path(args.out).exists():
    raise FileExistsError(
        f'Prediction output already exists: {args.out}. '
        'Use a new output name for a new run.'
    )
```

**New test file:** `tests/member1/test_dayB_cli_safety.py`

```python
import sys
import pytest
from scripts import dayB_inference


def test_manifest_requires_out_before_touching_model(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', [
        'dayB_inference.py', '--manifest', 'missing.jsonl',
        '--mode', 'reference',
    ])
    with pytest.raises(SystemExit) as error:
        dayB_inference.main()
    assert error.value.code == 2
    assert '--out is required for manifest inference' in capsys.readouterr().err
```

Run `python -m pytest -q tests/member1/test_dayB_cli_safety.py`. If the local code is already correctly ordered, mark this step **already implemented** and only add the test if coverage is missing. No wider refactor is authorized.

### Pilot inference commands: use actual existing `--manifest` vs `--features` CLI correctly

Single-clip FP32 parity was already checked at G2. Now run a 10-clip pipeline check:

```bash
python scripts/dayB_inference.py \
  --manifest data/manifests/pilot_available_val.jsonl \
  --mode reference --device cpu --run-id dayB-pilot-ref \
  --out artifacts/predictions/dayB_pilot_reference_val.jsonl

python scripts/dayB_inference.py \
  --manifest data/manifests/pilot_available_val.jsonl \
  --mode low_precision --device cpu --run-id dayB-pilot-lowp \
  --out artifacts/predictions/dayB_pilot_low_precision_val.jsonl

python scripts/dayB_inference.py \
  --manifest data/manifests/pilot_available_val.jsonl \
  --mode restore_value_output --device cpu --run-id dayB-pilot-restored \
  --out artifacts/predictions/dayB_pilot_restore_value_output_val.jsonl
```

**Optional code-smoke only:** `--mode restore_query` can also run on the same pilot to test the already-implemented future Day 8 control. If run, label it `pilot_control_only`; it does not count as completing Day 8 statistics.

Inspect all three pilot JSONLs using this existing validator plus a cross-mode ID check:

```bash
python - <<'PY'
from pathlib import Path
from avfuse.member2_data.predictions import validate_predictions
from avfuse.shared.jsonl import read_jsonl
root = Path('artifacts/predictions')
modes = ('reference', 'low_precision', 'restore_value_output')
sets = []
for mode in modes:
    p = root / f'dayB_pilot_{mode}_val.jsonl'
    rows = read_jsonl(p)
    validate_predictions(rows)
    assert len(rows) == 10, (mode, len(rows))
    assert all(r['split'] == 'val' and r['config'] == mode for r in rows)
    clip_ids = {r['clip_id'] for r in rows}
    assert len(clip_ids) == 10
    sets.append(clip_ids)
    print(mode, 'PASS', len(rows))
assert all(s == sets[0] for s in sets)
print('PILOT ID ALIGNMENT: PASS')
PY
```

**Do not run `scripts/m2_evaluate.py` on the 10-clip data and report its output as an H2 finding**; its tiny nonrepresentative sample has only two real clips and cannot support stable 5% FPR calibration or source-bootstrap conclusions. You may validate JSON schemas and detect gross score bugs only.

**HARD GATE G3.** Give pilot IDs/counts, parity, each config's score range, output file paths, tests, unchanged checkpoint and file hashes. State explicitly: **“PILOT PASSED; FULL DAYS 5–7 NOT YET COMPLETE.”** **STOP AND AWAIT ACK G3.**

---

## G4 — Mandatory full-cache preflight, run WHERE THE CACHE IS (no 4 GB local download)

### What is required before running

- User-approved access to Member 2's **existing team Drive** cache. It reportedly has `dimodif_emb/val/<clip_id>/{features.npz,meta.json}` for 1,200 validation clips and a separate sealed `test/` subtree.
- The **same code snapshot** that passed G2/G3, including any uncommitted local changes. Because nothing may be pushed, transport the current working code to Colab by a **user-approved archive/upload/mounted folder**, not a Git commit. Do not claim a fresh GitHub clone necessarily includes local modifications.
- The approved `.pth` AND corresponding `.json` checkpoint config under `third_party/dimodif/ckpt/dfd/` or the chosen code directory's equivalent; original upstream `third_party/dimodif/src/models.py` is required for inference.
- Exact `manifest_whole_val.jsonl` and canonical `manifest_whole.jsonl` from G1, unchanged.
- GPU environment available and pinned/reported, if running on Colab. Use **one device** consistently for all configurations within the forensic experiment and for paired runtime comparisons. CPU or MPS results cannot be mixed with CUDA numbers as if device were identical.

### Local packaging rules (no GitHub writes)

Agent: **do not auto-upload**. Prepare a read-only inventory of necessary local code and ask the user how they want it transferred. If the user approves local archive creation and upload, include `src/`, `scripts/`, `tests/`, `contracts/`, `pyproject.toml`, required docs/configs, and already-verified manifest files; exclude `.git`, `.venv`, model cache, video media and test features. Verify archive contents. The local modifications must be present in the remote environment. No network upload is authorized without user approval.

### Run preflight on Colab/Drive only after G4 permission

Example, to adapt **after the user provides the exact cache root**:

```bash
# From a working copy of the exact local code snapshot:
python -m pip install -e '.[dev]'

python scripts/day6_preflight.py \
  --manifest data/manifests/manifest_whole_val.jsonl \
  --cache-root /content/drive/MyDrive/REPLACE_WITH_ACTUAL_TEAM_PATH/dimodif_emb \
  --checkpoint third_party/dimodif/ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.pth \
  --out-manifest data/manifests/dayB_val_ready.jsonl \
  --out-report artifacts/dayB/preflight_report.json
```

The path above is an **illustrative placeholder**, not a discovered Drive location. The agent must not substitute it without verifying a real directory. The preflight must see 1,200/1,200 verified, 300 per category, **zero errors**, approved SHA, and no accidental test input. The ready manifest's absolute cache paths are valid **in the environment where the preflight ran**. Do not carry those absolute paths to a different machine unless they exist there too.

### Inspect the generated report without rerunning inference

```bash
python - <<'PY'
import json
from pathlib import Path
from avfuse.member2_data.manifest import read_manifest
p = Path('artifacts/dayB/preflight_report.json')
r = json.loads(p.read_text())
assert r['status'] == 'PASS', r.get('errors', [])[:5]
assert r['n_manifest'] == r['n_verified'] == 1200
assert r['counts_by_type'] == {'real':300,'audio_fake':300,'video_fake':300,'both_fake':300}
assert r['errors_count'] == 0
assert r['checkpoint_sha256'] == 'e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e'
rows = read_manifest('data/manifests/dayB_val_ready.jsonl')
assert len(rows) == 1200
assert all(r['split'] == 'val' and Path(r['feature_path']).is_file() for r in rows)
print('DAY B FULL VALIDATION PREFLIGHT: PASS')
PY
```

**Do not accept** an empty ready manifest, a 10-clip manifest renamed to `dayB_val_ready.jsonl`, skipped/filtered missing validation clips, synthetic cache substitutes, test set leakage, or a different checkpoint. Do not reduce `--expected-count` to make the pass green: the original Day B requirements are unchanged.

**HARD GATE G4.** Show preflight JSON, original/derived manifest hash, checkpoint SHA, verified per-type counts, environment and exact no-test-access confirmation. Ask user to upload `preflight_report.json` or its structured summary. **STOP AND AWAIT ACK G4.**

---

## G5 — Day 5 scientific deliverables: 1,200 FP32 predictions, baseline forensic metrics and cached-model benchmark

### Member 1 — full FP32 predictions (same remote cache environment)

```bash
python scripts/dayB_inference.py \
  --manifest data/manifests/dayB_val_ready.jsonl \
  --mode reference --device cuda:0 \
  --run-id dayB-full-reference-val \
  --out artifacts/predictions/dayB_reference_val.jsonl
```

Use `cuda:0` only if it was verified at G4. If the approved device is CPU/MPS, use that **same** device for all later modes and runtime comparisons. `scripts/dayB_inference.py` currently **refuses to overwrite** an existing prediction path. On reruns use a distinct `--out` and `--run-id`, preserving old evidence. If it fails mid-run, preserve the traceback; do not silently remove/skip failed clips.

**Success:** exactly 1,200 rows, every `clip_id` from the validation manifest, `split=val`, `config=reference`, finite score in `[0,1]`, fixed checkpoint SHA and no labels in the prediction JSONL. Distinguish the unmodified pretrained reference from the instrumented reference used for parity testing.

### Member 2 — run the existing full-manifest prediction checks and baseline statistics

**IMPORTANT:** Member 2's evaluation should join **against the canonical `manifest_whole.jsonl`** (3,200 metadata rows), not a fabricated or relabeled pilot manifest. The code selects `split=val` and the sealed test entries are not scored.

```bash
python scripts/m2_check_predictions.py \
  --manifest data/manifests/manifest_whole.jsonl \
  --predictions artifacts/predictions/dayB_reference_val.jsonl \
  --out artifacts/reports/dayB_reference_prediction_check.json

python scripts/m2_evaluate.py calibrate \
  --manifest data/manifests/manifest_whole.jsonl \
  --predictions artifacts/predictions/dayB_reference_val.jsonl \
  --out artifacts/reports/dayB_validation_calibration.json

python scripts/m2_evaluate.py evaluate \
  --manifest data/manifests/manifest_whole.jsonl \
  --predictions artifacts/predictions/dayB_reference_val.jsonl \
  --protocol artifacts/reports/dayB_validation_calibration.json \
  --split val --n-boot 1000 --seed 0 \
  --out artifacts/reports/dayB_reference_val
```

The calibration JSON is **provisional Day B validation evidence**, not the Day 9 locked freeze decision. Never silently replace it in G6/G8; all modes use this **same exact threshold file and manifest hash**. Inspect orientation (`ok` / `INVERTED` / `unclear`), overall AUROC/AUPRC/BCE, real FPR, recall at threshold, type-versus-real metrics and 95% cluster bootstrap CIs. If orientation is inverted, **stop** and investigate the original scoring convention; never simply flip scores to improve AUROC. If threshold falls outside `[0,1]` because of saturated real scores, or CIs are undefined/NaN, **stop** and request a scientifically justified protocol fix rather than silently changing threshold calibration.

### Member 3 — cached-feature model reference benchmark

Use **the existing benchmark engine** `avfuse.member3_runtime.benchmark.benchmark_callable`: `warmup=10`, `repeats=50`, device synchronization, p50/p95 and supported peak memory. Inputs and model must be on device **before** starting the timed function; no disk I/O, raw video, face detection, audio encoder or per-call CPU→CUDA transfer inside timing. On a Mac CPU/MPS benchmark, peak memory may be unavailable and must be **null**, not zero. Actual GPU memory claims need supported CUDA measurements.

The complete, reusable new benchmark script is specified in **G8 Appendix A** below; **implement and test it during G5** if missing, and reuse it unchanged in G6/G8. At this point run only `reference`. Time representative validation cached-feature shapes **on the same device** later used for low-precision/restored modes; record selected validation clip IDs and token lengths **separately in private run metadata**. Do not mix the expensive offline feature-extraction timings (already in Member 2's `FEATURE_CACHE_RESULTS.md`) into cached-model timing. The latter is “model on cached encoder features”, not raw-video latency.

**Exact reference benchmark CLI after the user has verified a real feature cache path and approved a representative validation clip:**

```bash
python scripts/dayB_model_benchmark.py \
  --features "<ACTUAL_CACHE_ROOT>/val/<APPROVED_VALIDATION_CLIP_ID>/features.npz" \
  --device cuda:0 --modes reference --warmup 10 --repeats 50 \
  --run-id dayB-full-reference-model-bench \
  --experiment-id dayB-matched-validation-model \
  --out artifacts/benchmarks/dayB_model_reference.jsonl
```

The angle-bracket path is a **placeholder to be filled from the verified G4 cache**, not a usable literal. Preserve the **same chosen file** and device for later lowp/restored timing. For full performance characterization, also benchmark preselected short/typical/long validation token lengths as separate workloads and report their variation; a single sample is only the matched restoration-cost normalization anchor.

**HARD GATE G5.** Present 1,200 reference row count, manifest and checkpoint hashes, orientation + baseline AUROC/AUPRC/recall-per-type with CIs, threshold (val only), model-level benchmark p50/p95, peak-memory availability and scope, and log of any failures. **STOP AND AWAIT ACK G5.**

---

## G6 — Day 6: full low-precision results and H2 interpretation

### Member 1 — full lowp inference

```bash
python scripts/dayB_inference.py \
  --manifest data/manifests/dayB_val_ready.jsonl \
  --mode low_precision --device cuda:0 \
  --run-id dayB-full-lowp-val \
  --out artifacts/predictions/dayB_low_precision_val.jsonl
```

Use the **same approved device**, full ready manifest, checkpoint, exact code version and score aggregation as G5. If any element differs, stop and investigate instead of treating the results as paired.

### Member 2 — verify and evaluate both modes together

```bash
python scripts/m2_check_predictions.py \
  --manifest data/manifests/manifest_whole.jsonl \
  --predictions \
    artifacts/predictions/dayB_reference_val.jsonl \
    artifacts/predictions/dayB_low_precision_val.jsonl \
  --out artifacts/reports/dayB_reference_lowp_check.json

python scripts/m2_evaluate.py evaluate \
  --manifest data/manifests/manifest_whole.jsonl \
  --predictions \
    artifacts/predictions/dayB_reference_val.jsonl \
    artifacts/predictions/dayB_low_precision_val.jsonl \
  --protocol artifacts/reports/dayB_validation_calibration.json \
  --split val --n-boot 1000 --seed 0 \
  --out artifacts/reports/dayB_reference_lowp_val

python scripts/m2_paired.py \
  --manifest data/manifests/manifest_whole.jsonl \
  --predictions \
    artifacts/predictions/dayB_reference_val.jsonl \
    artifacts/predictions/dayB_low_precision_val.jsonl \
  --protocol artifacts/reports/dayB_validation_calibration.json \
  --n-boot 1000 --seed 0 \
  --out artifacts/reports/dayB_lowp_paired_val
```

`m2_paired.py`'s current restoration-table code supports the reference+lowp pair with zero restoration arms; if local code differs, audit before running. Examine paired **BCE damage** `L_low_precision − L_reference` and its source-group 95% CI as well as **AUROC/AUPRC/recall** degradation per type. The two families of metrics answer different questions: BCE measures calibrated prediction loss, whereas AUROC describes ranking and recall uses the fixed reference threshold. A larger BCE does not mathematically guarantee a lower AUROC.

### Member 3 — lowp benchmarking

Run the same cached-model benchmark scope/workloads/device/repeats as G5, changing **only** `mode` to `low_precision`; include fake quant/dequant work **inside** the timed callable. For a fresh paired benchmark with the same input and environment, run:

```bash
python scripts/dayB_model_benchmark.py \
  --features "<ACTUAL_CACHE_ROOT>/val/<APPROVED_VALIDATION_CLIP_ID>/features.npz" \
  --device cuda:0 --modes reference low_precision --warmup 10 --repeats 50 \
  --run-id dayB-full-reference-lowp-model-bench \
  --experiment-id dayB-matched-validation-model \
  --out artifacts/benchmarks/dayB_model_reference_lowp.jsonl
```

Use the **identical actual cache file** chosen at G5 and a distinct output path. Measure isolated attention too with existing `benchmark_attention_operator()` on the **real Day A captured** Q/K/V tensors (see G8) and report “simulated INT8, no integer kernels.” A speedup is not assumed; fake quant may be slower.

### Mandatory H2 interpretation questions

1. Is `L_low_precision − L_reference` positive or negative? Does the 95% CI exclude zero?
2. Did AUROC change materially overall? For real-versus-audio-fake, real-versus-video-fake, and real-versus-both-fake?
3. Did threshold-fixed recall or real FPR change?
4. Were the same 1,200 clips present with **identical** IDs and checkpoint hashes?
5. Were score differences nonzero while scores and logits remained finite?
6. Could any apparent result be caused by score inversion, saturated scores, identity/source leakage, extraction shortcut (e.g. trailing silence), or a device/precision mismatch?
7. If degradation is absent, **do not invent** a positive restoration narrative. The original plan explicitly permits a negative finding; the **Day 7 restoration experiment must still be run** if the user approves it.

**HARD GATE G6.** Present the full per-type H2 table, CIs, paired damage, device/latency, and evidence limitations. **STOP AND AWAIT ACK G6.**

---
## G7 — Day 7 mathematics and the **same-low-precision-upstream** scientific audit

Day 7's original plan requires restoration of the V/output information-carrying path **while Q/K stay low-precision**, with the same low-precision upstream activations. The repository's `restore_value_output` math is already implemented, but **this causal wording needs deliberate verification**:

- **Within one attention call**, `prepare_qkv()` constructs the quantized Q/K and keeps the original V. This is a well-defined controlled intervention when the input Q/K/V is fixed.
- **At layer 0**, the first layer's input is identical between a low-precision whole-model run and a restored whole-model run on the same clip, so the paired computation has a matching upstream input.
- **At layers 1–4**, restoring a preceding layer can change its output and therefore change downstream hidden states and Q/K/V. The two *whole-model* runs no longer necessarily receive the same upstream activations at every layer. Consequently, the model-level comparison is a **cumulative 5-layer intervention**, not an isolated causal effect with every layer's upstream held constant.
- This does **not** invalidate the model-level comparison as a controlled configuration study. It **does** limit any claim that each restored layer was tested against perfectly matched low-precision upstream. The agent may not hide this distinction or rewrite the experiment to imply otherwise.

**Mandatory action:** present these two interpretations to the user; if causal isolation at *every layer* is necessary for the research claim, request explicit approval for a **separate layer-specific matched-upstream diagnostic**. Do not silently rearchitect the main four-config protocol. A layer-specific diagnostic can run the prefix layers identically low-precision up to target layer `i`, switch only layer `i` to `restore_value_output`, and compare against a version with that same prefix and low precision at `i`. Analyze its downstream effect separately; label this **diagnostic**, not a replacement for whole-model Day 7 results.

### Tests already supplied by repo

- `tests/member1/test_dimodif_intervention.py::test_restoration_boundaries`
- `tests/member1/test_dimodif_intervention.py::test_model_weights_are_unchanged`
- `tests/member1/test_dimodif_intervention.py::test_original_forward_is_restored`
- `tests/member1/test_precision.py::test_restore_paths_preserve_original_unquantized_operand`
- New `tests/member1/test_dayB_mode_math.py::test_restored_value_output_equals_manual_targeted_math` from G2.

Run:

```bash
python -m pytest -q tests/member1/test_dimodif_intervention.py \
  tests/member1/test_precision.py tests/member1/test_dayB_mode_math.py
```

### New file: `scripts/dayB_upstream_audit.py`

This is a **diagnostic-only** tool: it reads one real, already-cached validation pilot, compares the inputs *reaching* each attention block across lowp and restored runs, and **does not use labels** or measure forensic accuracy. It writes one JSON artifact and cannot make a scientific interpretation automatically.

```python
"""Audit matched/upstream attention inputs for the five-layer DiMoDif intervention."""
import argparse
import json
from pathlib import Path
import sys

import torch

# Direct `python scripts/<name>.py` execution must also find the sibling scripts package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from avfuse.member1_fusion.dimodif_adapter import DiMoDifAdapter
from scripts.dayB_inference import load_features


def capture_inputs(adapter, features, mode):
    layers = adapter.model.transformer_encoder.layers
    inputs = []
    hooks = []

    def make_hook(layer_index):
        def hook(_module, args):
            inputs.append((layer_index, args[0].detach().cpu().clone()))
        return hook

    try:
        for i, layer in enumerate(layers):
            hooks.append(layer.self_attn.register_forward_pre_hook(make_hook(i)))
        result = adapter.forward_with_mode(features, mode=mode)
    finally:
        for handle in hooks:
            handle.remove()

    inputs.sort(key=lambda item: item[0])
    if len(inputs) != len(layers):
        raise RuntimeError(f'Expected {len(layers)} attention inputs, got {len(inputs)}')
    return [tensor for _, tensor in inputs], float(result['score_fake'][0].item())


def audit(adapter, features):
    low, low_score = capture_inputs(adapter, features, 'low_precision')
    restored, restored_score = capture_inputs(adapter, features, 'restore_value_output')
    assert len(low) == len(restored)
    rows = []
    for i, (a, b) in enumerate(zip(low, restored)):
        assert a.shape == b.shape and a.dtype == b.dtype
        diff = (a - b).abs()
        rows.append({
            'layer': i,
            'input_shape': list(a.shape),
            'upstream_bit_exact': bool(torch.equal(a, b)),
            'upstream_max_abs_difference': float(diff.max().item()),
            'upstream_mean_abs_difference': float(diff.mean().item()),
        })
    if not rows[0]['upstream_bit_exact']:
        raise RuntimeError('Layer-0 upstream is unexpectedly different')
    return {
        'description': 'Descriptive whole-model upstream audit, not per-layer causal isolation',
        'low_precision_score': low_score,
        'restore_value_output_score': restored_score,
        'layers': rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--upstream', default='third_party/dimodif')
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    adapter = DiMoDifAdapter(a.upstream, checkpoint_name='lavdf', device=a.device)
    expected_sha = 'e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e'
    if adapter.checkpoint_sha256 != expected_sha:
        raise RuntimeError('Unapproved detector checkpoint')
    features = load_features(a.features)
    with torch.inference_mode():
        result = audit(adapter, features)
    result['checkpoint_sha256'] = adapter.checkpoint_sha256
    result['feature_file'] = str(a.features.resolve())
    result['device'] = str(adapter.device)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    for row in result['layers']:
        print(row)
    print('Wrote', a.out)


if __name__ == '__main__':
    main()
```

### New unit test: `tests/member1/test_dayB_upstream_audit.py`

The toy model deliberately exercises multiple attention layers and runs with no external checkpoint. Unit tests should prove **layer 0 is matched**, not make a false assertion that every later layer is identical.

```python
import torch
from torch import nn

from avfuse.member1_fusion.dimodif_intervention import intervene_dimodif_attention


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,
            dropout=0.0, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=3)

    def forward(self, x):
        return self.transformer_encoder(x)


def _capture(model, x, mode):
    seen, hooks = [], []
    try:
        for layer in model.transformer_encoder.layers:
            hooks.append(layer.self_attn.register_forward_pre_hook(
                lambda _module, args: seen.append(args[0].detach().clone())
            ))
        with intervene_dimodif_attention(model, mode=mode):
            with torch.inference_mode():
                model(x)
    finally:
        for h in hooks:
            h.remove()
    return seen


def test_first_layer_has_matched_upstream_even_when_whole_model_modes_differ():
    torch.manual_seed(27)
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        model = ToyModel().eval()
        x = torch.randn(1, 13, 64)
        low = _capture(model, x, 'low_precision')
        restored = _capture(model, x, 'restore_value_output')
        assert len(low) == len(restored) == 3
        assert torch.equal(low[0], restored[0])
        assert all(a.shape == b.shape for a, b in zip(low, restored))
        # Do not demand equality for deeper layers; earlier restoration may alter their inputs.
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
```

Run:

```bash
python -m pytest -q tests/member1/test_dayB_upstream_audit.py
python scripts/dayB_upstream_audit.py \
  --features data/pilot/day2_features/features1.npz \
  --device cpu --out artifacts/dayB/upstream_audit.json
```

Inspect each `upstream_bit_exact` value and max difference. The code deliberately does **not** decide whether the project should call the result “causal precision restoration” or a “cumulative intervention.” That interpretation is reserved for the user.

### Full-model pilot restoration trace

```bash
python scripts/dayB_inference.py \
  --features data/pilot/day2_features/features1.npz \
  --mode restore_value_output --device cpu \
  --trace-out artifacts/dayB/restore_value_output_trace.json
```

Inspect `Q`/`K` `fraction_changed > 0`, `V` and `attention_output` `fraction_changed == 0` at the defined boundary; compare model-score behavior with low precision and FP32 but **do not infer better forensic accuracy from one clip**.

**HARD GATE G7.** Present test results, upstream-per-layer table, trace output, the exact scientific wording the agent recommends, and any need for a separate layer-specific matched-input analysis. **STOP AND AWAIT ACK G7**. If the user asks for a stronger causal diagnostic, design/test that diagnostic **in a new, separately approved subphase**; do not replace the frozen four configurations without explicit approval.

---

## G8 — Day 7 full restored predictions, validation-only loss recovery, mixed-precision benchmark

### Member 1 — full restored inference

```bash
python scripts/dayB_inference.py \
  --manifest data/manifests/dayB_val_ready.jsonl \
  --mode restore_value_output --device cuda:0 \
  --run-id dayB-full-restored-val \
  --out artifacts/predictions/dayB_restore_value_output_val.jsonl
```

Again, use the same approved full-validation environment, model/checkpoint, code snapshot, device and 1,200 clip IDs. The intervention must still leave Q/K quantized and restore V + attention output. Do not substitute `restore_query` for Day 7's targeted restoration.

### Member 2 — score validation and *paired restoration*, no held-out test

```bash
python scripts/m2_check_predictions.py \
  --manifest data/manifests/manifest_whole.jsonl \
  --predictions \
    artifacts/predictions/dayB_reference_val.jsonl \
    artifacts/predictions/dayB_low_precision_val.jsonl \
    artifacts/predictions/dayB_restore_value_output_val.jsonl \
  --out artifacts/reports/dayB_three_mode_prediction_check.json

python scripts/m2_evaluate.py evaluate \
  --manifest data/manifests/manifest_whole.jsonl \
  --predictions \
    artifacts/predictions/dayB_reference_val.jsonl \
    artifacts/predictions/dayB_low_precision_val.jsonl \
    artifacts/predictions/dayB_restore_value_output_val.jsonl \
  --protocol artifacts/reports/dayB_validation_calibration.json \
  --split val --n-boot 1000 --seed 0 \
  --out artifacts/reports/dayB_three_mode_val
```

If benchmark records from Appendix A are already generated for **the same chosen cached-feature workload and hardware** for all three modes, use them in paired recovery calculation:

```bash
python scripts/m2_paired.py \
  --manifest data/manifests/manifest_whole.jsonl \
  --predictions \
    artifacts/predictions/dayB_reference_val.jsonl \
    artifacts/predictions/dayB_low_precision_val.jsonl \
    artifacts/predictions/dayB_restore_value_output_val.jsonl \
  --protocol artifacts/reports/dayB_validation_calibration.json \
  --benchmarks artifacts/benchmarks/dayB_model_matched.jsonl \
  --scope cached_feature_model --n-boot 1000 --seed 0 \
  --out artifacts/reports/dayB_restoration_paired_val
```

**If real benchmark records do not yet exist**, omit `--benchmarks` rather than fabricate timings; mark cost normalization as **BLOCKED**, and finish Member 3's benchmark before claiming Day B completion.

The canonical recovery statistic implemented in `src/avfuse/member2_data/paired.py` is:

```
L_cfg           = mean binary cross-entropy on the same labeled validation clips
Damage          = L_low_precision - L_reference
R_value_output  = L_low_precision - L_restore_value_output
Share recovered = R_value_output / Damage   only when Damage > 0
Added cost      = p50_ms(restore_value_output) - p50_ms(low_precision)
R per ms        = R_value_output / Added cost only when Added cost > 0
```

These are **paired, source-cluster-bootstrap** comparisons; present `R` and its 95% CI, AUROC/AUPRC/recall changes by manipulation type, and the actual extra milliseconds separately. Do not assume `R>0` or `Added cost>0`. If measured added cost is zero/negative, `R_per_ms` is **not** an ordinary positive ratio; inspect and explain, do not divide by zero or silently take an absolute value. A confidence interval containing zero is not strong positive evidence.

**Important benchmark alignment bug to avoid:** `paired.measured_extra_ms()` maps one `p50_ms` per config and does not itself enforce identical token lengths or input hashes between multiple records. Feed it **one matched-workload three-mode benchmark set** (`dayB_model_matched.jsonl`) with one record per mode on the **same source sample/device/input shapes**. Keep additional p10/p50/p90-shape benchmark runs in *separate* result files for broader runtime analysis. Do not concatenate heterogeneous benchmark series and let the last row per config define the added cost.

### Member 3 — mixed precision cost, including conversions

**Exact three-mode matched benchmark command** after G4's actual cache and same anchor sample/device have been verified:

```bash
python scripts/dayB_model_benchmark.py \
  --features "<ACTUAL_CACHE_ROOT>/val/<APPROVED_VALIDATION_CLIP_ID>/features.npz" \
  --device cuda:0 --modes reference low_precision restore_value_output \
  --warmup 10 --repeats 50 \
  --run-id dayB-full-three-mode-model-bench \
  --experiment-id dayB-matched-validation-model \
  --out artifacts/benchmarks/dayB_model_matched.jsonl
```

Only after this produces three **schema-validated** benchmark records from the *same* input should it be used by `m2_paired.py --benchmarks ...` above. Collect additional representative-shape measurements in independent files.

Requirements:

- Use the **existing** `benchmark_callable()` and `benchmark_attention_operator()`; do not roll a new ad hoc wall-clock function.
- Same device (named), identical cached tensors, batch size, dtype, warm-up, repeats, PyTorch/backend, sampling method across modes.
- Distinguish **operator scope** and **cached-feature model scope**; report neither as raw-media latency.
- `fake_quant_int8` runs **inside** `fusion_attention`, so the timed callable **already includes quantize/dequantize conversion work**, per the actual benchmark code. Do **not** subtract a conversion estimate from the observed latency, and do not assume isolated conversion overhead has been measured unless an additional conversion-only callable is benchmarked.
- Record p50/p95, CUDA peak memory if supported, and a clear **unavailable** marker for CPU/MPS peak measurements. Never write fabricated zero memory or false “INT8 kernel” speedups.
- The original plan says compare restoration against an equal-cost control, but the code and method do **not** prove equality. Query-control cost comparison belongs to Day 8, and equal cost must be **measured**, not asserted.
- Member 2's already-measured offline extraction times (face track, crop, audio, video encoding) stay separate; if used, cite the actual provenance and clarify whether the “total” is a sum of individual stage medians, not a measured end-to-end median.

### Appendix A — New complete script `scripts/dayB_model_benchmark.py`

This script uses the repository's existing **validated v0.2 benchmark-record builder**, and never accesses labels. It runs cached-model inference with inputs already on the benchmark device. The default four modes allow an engineering smoke test; `--modes reference low_precision restore_value_output` meets Day B scope. For a Day 5 reference-only benchmark use `--modes reference` and for G6 use matched `--modes reference low_precision` (or separate results with unambiguous source evidence).

```python
"""Day B real cached-feature model benchmark; no media decoding or test access.

The benchmark_callable includes precision conversions performed in the model.
One source NPZ is benchmarked per invocation; keep workloads matched across modes.
"""
import argparse
from pathlib import Path
import sys

import torch

# Make the scripts package importable when run directly by file path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from avfuse.member1_fusion.dimodif_adapter import DiMoDifAdapter
from avfuse.member3_runtime.artifacts import (
    build_benchmark_record,
    environment_block,
    validate_benchmark_records,
    write_benchmark_records,
)
from avfuse.member3_runtime.benchmark import benchmark_callable
from avfuse.member3_runtime.environment import collect_environment_report
from avfuse.member3_runtime.provenance import hash_checkpoint, hash_feature_cache
from avfuse.shared.contracts import PRECISION_MODES
from scripts.dayB_inference import load_features

EXPECTED_SHA = 'e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e'


def build_model_benchmark_records(
    adapter, features, source_path: Path, *, device: str, modes,
    warmup: int, repeats: int, run_id: str, experiment_id: str,
):
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if adapter.checkpoint_sha256 != EXPECTED_SHA:
        raise RuntimeError('Benchmark checkpoint SHA mismatch')
    if adapter.model.training or any(p.requires_grad for p in adapter.model.parameters()):
        raise RuntimeError('Model must be eval and fully frozen')

    # Do not benchmark host-to-device copies. Inputs are resident on the device.
    dev = torch.device(device)
    inputs = {key: tensor.detach().to(dev) for key, tensor in features.items()}
    shapes = {key: list(tensor.shape) for key, tensor in inputs.items()}
    report = collect_environment_report()
    environment = environment_block(report, str(dev), git=report.get('git'))
    checkpoint = hash_checkpoint(adapter.checkpoint_path)
    # Hash this ONE actual NPZ cache directory, not the full 4-GB dataset.
    identity = hash_feature_cache(source_path.parent)
    records = []
    for mode in modes:
        if mode not in PRECISION_MODES:
            raise ValueError(f'Unknown precision configuration: {mode}')
        measurement = benchmark_callable(
            lambda mode=mode: adapter.predict_from_cached_features(inputs, mode),
            device=dev, warmup=warmup, repeats=repeats,
            synchronize=True, measure_memory=True,
        )
        record = build_benchmark_record(
            measurement,
            run_id=run_id,
            experiment_id=experiment_id,
            config=mode,
            scope='cached_feature_model',
            backend='sdpa',
            batch_size=1,
            execution_dtype=torch.float32,
            input_boundary='cached_encoder_tokens',
            input_shapes=shapes,
            environment=environment,
            checkpoint=checkpoint,
            input_identity=identity,
            conversion_overhead_ms=None,
            # Total mode latency includes conversions, but isolated conversion-only
            # milliseconds have not been measured: leave that separate field null.
            conversion_overhead_included_in_latency=(mode != 'reference'),
            notes=(
                'Cached encoder tokens already placed on the benchmark device; '
                'includes model preparation/padding, attention and detection head. '
                'Any fake INT8 quantize/dequantize executes inside the timed model. '
                'No raw-media decoding or true INT8 kernel is timed.'
            ),
        )
        records.append(record)
    validate_benchmark_records(records, run_id=run_id)
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--upstream', default='third_party/dimodif')
    p.add_argument('--device', default='cpu')
    p.add_argument('--modes', nargs='+', choices=PRECISION_MODES,
                   default=list(PRECISION_MODES))
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--run-id', required=True)
    p.add_argument('--experiment-id', default='dayB-cached-feature-model')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    adapter = DiMoDifAdapter(args.upstream, checkpoint_name='lavdf', device=args.device)
    features = load_features(args.features)
    records = build_model_benchmark_records(
        adapter, features, args.features,
        device=args.device, modes=args.modes,
        warmup=args.warmup, repeats=args.repeats,
        run_id=args.run_id, experiment_id=args.experiment_id,
    )
    result = write_benchmark_records(
        records, args.out.parent, run_id=args.run_id,
        filename=args.out.name, overwrite=False,
    )
    for r in records:
        print(r['config'], r['p50_ms'], r['p95_ms'], r['peak_memory_bytes'])
    print('Wrote', result['record_count'], 'schema-validated records:', args.out)


if __name__ == '__main__':
    main()
```

### Appendix A test: `tests/member3/test_dayB_model_benchmark.py`

This does **not** pretend a synthetic model benchmark is scientific evidence; it confirms that the new adapter-to-existing-runtime integration creates valid schema records.

```python
import torch
from torch import nn

from avfuse.member3_runtime.artifacts import validate_benchmark_records
from scripts.dayB_model_benchmark import build_model_benchmark_records, EXPECTED_SHA


class FakeFrozenAdapter:
    checkpoint_sha256 = EXPECTED_SHA

    def __init__(self, checkpoint):
        self.checkpoint_path = checkpoint
        self.model = nn.Linear(1, 1, bias=False).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def predict_from_cached_features(self, features, mode):
        assert mode in ('reference', 'low_precision', 'restore_value_output')
        s = features['video_tokens'].mean() + features['audio_tokens'].mean()
        return torch.sigmoid(s.reshape(1))


def test_benchmark_uses_existing_schema_and_explicit_scope(tmp_path):
    import numpy as np
    checkpoint = tmp_path / 'unit-checkpoint.pth'
    checkpoint.write_bytes(b'unit-test-checkpoint-not-real-weights')
    feature_dir = tmp_path / 'val' / '0000000000000001'
    feature_dir.mkdir(parents=True)
    features_file = feature_dir / 'features.npz'
    np.savez(features_file,
        video_features=np.zeros((5, 768), dtype=np.float32),
        audio_features=np.zeros((6, 768), dtype=np.float32))
    features = {
        'video_tokens': torch.zeros(5, 768),
        'audio_tokens': torch.zeros(6, 768),
    }
    adapter = FakeFrozenAdapter(checkpoint)
    records = build_model_benchmark_records(
        adapter, features, features_file, device='cpu',
        modes=('reference', 'low_precision', 'restore_value_output'),
        warmup=0, repeats=3, run_id='unit-benchmark', experiment_id='dayB-unit',
    )
    validate_benchmark_records(records, run_id='unit-benchmark')
    assert len(records) == 3
    assert {r['config'] for r in records} == {
        'reference', 'low_precision', 'restore_value_output'
    }
    assert all(r['scope'] == 'cached_feature_model' for r in records)
    assert all(r['peak_memory_bytes'] is None for r in records)
    assert all(r['precision']['real_int8_kernel'] is False for r in records)
    assert all(r['workload']['input_boundary'] == 'cached_encoder_tokens'
               for r in records)
```

Run:

```bash
python -m pytest -q tests/member3/test_dayB_model_benchmark.py
```

### Real benchmark — identical cached input for all Day B modes

Run the benchmark on the same **representative validation-only cached video**, without touching test. Example single pilot for local wiring verification (not publication-ready performance):

```bash
python scripts/dayB_model_benchmark.py \
  --features data/pilot/test_day2/val/2961dde91d90fee1/features.npz \
  --device cpu --modes reference low_precision restore_value_output \
  --warmup 10 --repeats 50 \
  --run-id dayB-pilot-model-benchmark \
  --experiment-id dayB-pilot-three-modes \
  --out artifacts/benchmarks/dayB_model_pilot_matched.jsonl
```

For **full Day B research cost evidence**, rerun on the same user-approved GPU as full validation, using one or more **real validation samples** with typical and stress-case token lengths. For the **single matched** three-mode cost-normalization benchmark, choose one fixed val sample and give the file `artifacts/benchmarks/dayB_model_matched.jsonl`. Separately collect realistic variation across short/median/long validation inputs and across categories, without feeding heterogeneous cases into `paired.measured_extra_ms()`. Do not select a benchmark input based on its observed restoration effect.

**Required extra operator-level benchmark:** reuse the real projected Day A tensors saved by `scripts/dayA_model_parity.py` at `artifacts/dayA/layer0_qkv.pt`. Those tensors contain `q`, `k`, `v`, and SDPA-convention `attn_mask`. Run the existing API:

```python
# One-off script/REPL or a Member 3-owned runner, not a replacement benchmark engine.
from pathlib import Path
import json
import torch
from avfuse.member3_runtime.operator_benchmark import benchmark_attention_operator

saved = torch.load('artifacts/dayA/layer0_qkv.pt', map_location='cpu', weights_only=True)
device = 'cpu'  # replace with the approved hardware and move all inputs first
q, k, v = (saved[name].to(device) for name in ('q', 'k', 'v'))
mask = saved['attn_mask']
if mask is not None:
    mask = mask.to(device)
results = []
for mode in ('reference', 'low_precision', 'restore_value_output'):
    results.append(benchmark_attention_operator(
        q, k, v, mode=mode, backend='sdpa', device=device,
        input_kind='real', shape_source='artifacts/dayA/layer0_qkv.pt',
        attn_mask=mask, warmup=10, repeats=50,
    ))
out = Path('artifacts/benchmarks/dayB_operator_raw_diagnostics.json')
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    raise FileExistsError(out)
out.write_text(json.dumps(results, indent=2) + '\n')
print('Saved three raw operator diagnostics (not v0.2 records):', out)
```

Do **not** pass `dayB_operator_raw_diagnostics.json` to `m2_paired.py`: it is a JSON report from the operator runner, **not** a schema-v0.2 benchmark JSONL and **not** cached-model latency. If Member 3 needs official operator records, wrap the **measurement dict** using their `build_benchmark_record(..., scope='operator', input_boundary='cached_encoder_tokens', ...)`, with consistent environment/checkpoint/identity, and run the existing validator before writing. Do not label synthetic Q/K/V as genuine real-model data.

### Mandatory Day 7 interpretation questions

1. Was the 1,200-clip restored prediction file complete, compatible, fixed-checkpoint and same-device?
2. Were Q/K quantized and V/output unquantized at the intended boundary across five joint attention layers?
3. What did the upstream-audit report establish: matched first-layer inputs versus possible different deeper-layer inputs?
4. What is the paired `R_value_output = L_lowp − L_restored` with 95% CI? Is that CI entirely positive, entirely negative, or overlapping zero?
5. How did type-versus-real AUROC/AUPRC and threshold-fixed recall change?
6. How many real CUDA memory measurements exist, or is memory properly `not_measured`?
7. What is measured incremental latency? Are fake-quant conversions included in the timed scope?
8. If low precision did **not** measurably hurt performance, is the restoration result still being framed honestly as a negative/no-damage finding?
9. Does evidence justify a targeted restoration conclusion, or merely a model-wide configuration comparison pending the later Day 8 query control?

**HARD GATE G8.** Present three validated prediction files, three-mode metric tables and source bootstrap CIs, matched benchmark schema records, upstream limitation, recovery statistic, runtime/memory evidence. Explicitly say **“Day 8 equal-cost control comparison not yet evaluated.”** **STOP AND AWAIT ACK G8.**

---
## G9 — Day B end-of-day audit, acceptance conditions, and handoffs

### Local test suite (only after previous gates were acknowledged)

```bash
python -m pytest -q tests/member1 tests/member2 tests/member3 tests/integration
python -m pytest -q tests/member1/test_dayB_mode_math.py \
  tests/member1/test_dayB_pilot_manifest.py \
  tests/member1/test_dayB_prediction_contract.py \
  tests/member1/test_dayB_cli_safety.py \
  tests/member1/test_dayB_upstream_audit.py \
  tests/member3/test_dayB_model_benchmark.py
```

The second command is deliberately redundant: it names every new Day B test explicitly so none disappears from discovery. A failing test must be reported; do not silence it, skip it, weaken an assertion, or mark it `xfail` without the user's acknowledgement. On the actual local machine, use `python -m pytest`; the repository is a Python package installed using `python -m pip install -e '.[dev]'`.

### Mandatory end-to-end validation of the three full prediction files

Run this **in the environment holding the real full-validation results**:

```bash
python - <<'PY'
import hashlib
from pathlib import Path
from avfuse.member2_data.manifest import read_manifest
from avfuse.member2_data.predictions import validate_predictions
from avfuse.shared.jsonl import read_jsonl

manifest_path = Path('data/manifests/manifest_whole.jsonl')
full_manifest = read_manifest(manifest_path)
expected = {m['clip_id'] for m in full_manifest if m['split'] == 'val'}
assert len(expected) == 1200
modes = ('reference', 'low_precision', 'restore_value_output')
checkpoint = 'e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e'
for mode in modes:
    path = Path('artifacts/predictions') / f'dayB_{mode}_val.jsonl'
    rows = read_jsonl(path)
    validate_predictions(rows)
    assert len(rows) == 1200, (mode, len(rows))
    assert {p['clip_id'] for p in rows} == expected, mode
    assert all(p['split'] == 'val' and p['config'] == mode for p in rows)
    assert all(p['checkpoint_sha256'] == checkpoint for p in rows)
    assert all(0.0 <= float(p['score_fake']) <= 1.0 for p in rows)
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(mode, '1200/1200 PASS', 'sha256=', file_sha)
print('THREE-MODE FULL VALIDATION COMPLETENESS: PASS')
PY
```

Review `artifacts/reports/dayB_reference_val/metrics.json`, `.../dayB_reference_lowp_val/metrics.json`, `.../dayB_three_mode_val/metrics.json`, `.../dayB_restoration_paired_val/paired.json` and the benchmark `dayB_model_matched.jsonl`. Confirm the same canonical manifest hash, checkpoint hash, validation threshold and training-overlap/shortcut limitations accompany the results. If the remote environment contains required outputs, the user may transfer **only the prediction JSONLs, validated benchmark JSONLs, metrics tables and reports** for local analysis; the large feature cache stays remote.

### Final scientific sign-off table

| Workstream | Required evidence | Status must be either PASS / BLOCKED / FAIL, never guessed |
|---|---|---|
| Day 5 — Member 1 | Exact reference/native parity; checkpoint SHA; frozen original model; fake-INT8 QKV/output operator unit tests | status + artifact |
| Day 5 — Member 2 | 1,200 reference predictions; baseline AUROC/AUPRC/BCE; per-type metrics and source CIs; validation-only fixed threshold | status + artifact |
| Day 5 — Member 3 | Real cached-feature model FP32 p50/p95; device; memory measured or unavailable | status + artifact |
| Day 6 — Member 1 | 1,200 low-precision predictions; same IDs/checkpoint/split; correct QKV/output fake quant | status + artifact |
| Day 6 — Member 2 | Lowp-vs-reference AUROC/AUPRC/recall and paired loss damage, by type, with 95% CIs | status + artifact |
| Day 6 — Member 3 | Same-device, same-shape lowp-vs-FP32 operator/model runtime and honest precision disclaimer | status + artifact |
| Day 7 — Member 1 | Correct V/output restoration, Q/K still quantized; five-layer trace; upstream-limit audit; 1,200 restored predictions | status + artifact |
| Day 7 — Member 2 | Validation-only `R_value_output`, BCE/stratum/CI recovery harness and output, no test scoring | status + artifact |
| Day 7 — Member 3 | Mixed-mode model/operator benchmark including conversions; compatible latency cost for R-per-ms; peak-memory disclosure | status + artifact |
| Research hygiene | Real vs fake type labeling, group-disjoint audit, train-overlap/shortcut report, reference score orientation | status + artifact |
| Reproducibility | Exact code snapshot or local diff, package versions, checkpoint and manifest hashes, used device, output hashes, commands | status + artifact |
| Ethics/data handling | No raw licensed media uploaded to GitHub; no test embeddings or scores accessed | status + artifact |
| Git local-only requirement | Local `git status --short` inspected; **zero Git writes, commits, pushes, merges or PRs** performed by agent | status + statement |

### Acceptance conditions — “Day B complete” is allowed only if ALL are true

1. The 1,200-clip validation manifest has been verified against the complete whole-video metadata manifest, and full cache preflight passes without omissions.
2. FP32 unmodified DiMoDif and reference-instrumented DiMoDif have passed real-checkpoint parity tests.
3. Fake INT8 numerical semantics are tested on Q/K/V/output, and restoration preserves V/output while leaving Q/K quantized.
4. **Three full and distinct** validation prediction JSONLs exist: reference, low precision, restore value/output. All have exactly the same 1,200 IDs and pretrained checkpoint SHA. Pilot files do not count.
5. Member 2 has finished baseline metrics, low-precision H2 metrics, and validation-only paired restoration metrics (including source-cluster confidence intervals and per-type real-versus-fake comparisons), using one validation-fixed threshold.
6. Member 3 has recorded same-device and same-input operator/model latency evidence for the three modes, including conversions inside the timed scope, plus honest CUDA/CPU/MPS peak-memory reporting.
7. Every human **hard gate** has been acknowledged; interpretation of numerical results is recorded as a human decision, not invented by the agent.
8. No Day 9 final policy has been prematurely locked, and no Day 10 held-out test inference was run.
9. No GitHub write, commit or push occurred, and no preexisting local work was overwritten without explicit approval.

If any item is missing, report **`DAY B PARTIALLY COMPLETE — BLOCKED BY [SPECIFIC REQUIREMENT]`** rather than declaring a pass. Missing Drive access is a logistics blocker, not permission to reduce the scientific sample size.

### Required final summary from the coding agent

```
DAY B (ORIGINAL DAYS 5 + 6 + 7) — FINAL LOCAL AUDIT
Repository branch / local HEAD:
Dirty/untracked files seen before work:
Files implemented or changed locally:
Files already implemented and reused:
New/updated unit and integration tests:
FP32 parity status:
Full cache preflight: verified / blocked / failed
Full val row counts: reference=..., lowp=..., restored=...
Approved checkpoint SHA:
Canonical manifest SHA:
Selected fixed validation threshold:
Per-type AUROC/AUPRC/recall/BCE and uncertainty:
Paired quantization damage and restoration recovery, CI:
Operator and cached-model benchmark p50/p95 / device / memory:
Scientific limitations (joint self-attention; upstream; fake INT8; dataset shortcut):
List of artifacts the user should upload for review:
NO commits / NO pushes / NO PRs / NO test-set inference: confirmed or explain deviation
DAY B STATUS: COMPLETE / PARTIALLY COMPLETE / FAILED
NEXT: Day 8 query-control evaluation (ONLY after separate authorization)

HARD GATE G9 — STOPPED; AWAITING USER ACKNOWLEDGEMENT
```

**HARD GATE G9.** Present this final evidence table and stop. Do not start Day 8, freeze Day 9, or touch Day 10 until a new explicit user request.

---

## 4. Fast error-decision reference for the agent

| Observed error | Root cause to inspect | Correct reaction |
|---|---|---|
| `FileNotFoundError: data/manifests/dayB_val_ready.jsonl` | Full preflight was not run, or ready manifest is on Colab only | **Stop** the 1,200 validation run; use `--features` for a pilot. Never invent a ready manifest. |
| `Missing cached features: data/cache/<clip_id>.pt` | Old source-manifest `feature_path` placeholder used directly | Run preflight on real `dimodif_emb/val/<id>/features.npz` to write correct absolute paths; never bulk-convert `.npz` just to satisfy stale placeholder. |
| `Cache root does not exist` | `--cache-root` is not the directory containing `val/` | Find/ask for the real team Drive root; do not create empty directories and lower expected counts. |
| Missing `.pth` or corresponding `.json` | Frozen pretrained detector is not installed locally/Colab | Stop; obtain only approved checkpoint/config. No random model or substitute checkpoint. |
| `Checkpoint mismatch` | SHA does not match approved pretrained detector | Stop; preserve current files and obtain human review. |
| `Prediction output already exists` | Runner refuses overwriting prior evidence | Inspect existing artifact and choose a new, uniquely named run/path with user's approval. |
| FP32 parity fails | MHA mask, packed projection, scoring, feature alignment, dtype or kernel differences | Save trace; inspect code and tolerances; don't loosen assertions or claim parity. |
| NaNs/infs | Incompatible mask, invalid feature tensors or FP32/quantization instability | Stop; preserve specific input ID + trace and debug. |
| Reference score orientation `INVERTED` | Score aggregation or dataset-label direction may be wrong | Stop; audit against original DiMoDif eval; never flip score labels silently. |
| Lowp AUROC unexpectedly improves | Fake-quant can sometimes regularize; or another problem | Present paired CI and controls; don't call it degraded without evidence. |
| V/output recovery negative or inconclusive | Hypothesis unsupported or measurement noisy | Report honest result and CI; no hand-tuned data exclusions. |
| Restoration and lowp upstream differ after layer 0 | Expected propagation through five sequential attention layers | Report cumulative-intervention limitation; request approval for separate targeted prefix-matched test if needed. |
| MPS/CPU peak memory `None` | No supported resettable PyTorch memory tracker | Report “not measured”; don't fabricate peak bytes. |
| Conversion overhead field `None` | Isolated conversion-only cost has not been measured | Note that full model latency **includes** conversion; do not claim a separate measured conversion number. |
| Matched benchmark results span different inputs/devices | Cost comparison would be invalid | Re-run same-input, same-device matched benchmarks; do not normalize R/ms across mismatched runs. |

## 5. How the agent should start this runbook

The agent's **first response** should be only:

> “I will begin at G0. I will inspect the local `member1` worktree, current code and dataset/checkpoint presence without editing or committing anything. I will then present the G0 report and stop for your acknowledgement before G1.”

It should then **perform G0**, show evidence, and wait for the user to upload results and explicitly approve the next gate. A single acknowledgement advances at most one gate; it does not grant permission to run the remainder of this document unattended.

**END OF DAY B AGENT RUNBOOK — NO GITHUB COMMITS OR PUSHES AUTHORIZED.**
