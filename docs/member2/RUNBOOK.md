# Member 2 runbook

All commands run from the repo root after `pip install -e '.[dev]'`. Paths under `data/` and
`artifacts/` are git-ignored, so nothing here can be committed by accident.

## Days 3–4: real clips and the manifest

```bash
# 1. which videos to keep (needs data/raw/LAV-DF/metadata.json)
python scripts/m2_lavdf.py select --root data/raw/LAV-DF --cap train=1500 val=300 test=500 \
    --out data/raw/lavdf_wanted.txt
# 2. stream LAV-DF.tar and keep only those (HF_TOKEN = your Hugging Face read token)
curl -L --fail -H "Authorization: Bearer $HF_TOKEN" \
    https://huggingface.co/datasets/ControlNet/LAV-DF/resolve/main/LAV-DF.tar \
    | tar -x -C data/raw -T data/raw/lavdf_wanted.txt
# 3. decode 4 s clips (60-80 min on Colab with 2 workers)
python scripts/m2_lavdf.py clips --root data/raw/LAV-DF --cap train=1500 val=300 test=500 --out data/clips
# 4. manifest + split audit (add --resplit if it reports groups crossing splits)
python scripts/m2_build_manifest.py --clips data/clips/clips.csv --out data/manifests/manifest.jsonl
python scripts/validate_manifest.py data/manifests/manifest.jsonl
```

A `manifest.csv` from the earlier Colab pipeline can be passed to `--clips` as is (same columns).

## Day 11 (run as soon as the clips exist, rerun before sign-off): shortcut audit

```bash
python scripts/m2_audit.py --clips data/clips/clips.csv --manifest data/manifests/manifest.jsonl \
    --out artifacts/reports/audit
```

## Days 5–10: once the Lead's prediction files exist

```bash
P=artifacts/predictions/*.jsonl
M=data/manifests/manifest.jsonl
python scripts/m2_evaluate.py calibrate --manifest $M --predictions $P --out artifacts/reports/protocol.json
python scripts/m2_evaluate.py evaluate --manifest $M --predictions $P --protocol artifacts/reports/protocol.json \
    --split val --out artifacts/reports/val
python scripts/m2_paired.py --manifest $M --predictions $P --protocol artifacts/reports/protocol.json \
    --benchmarks artifacts/benchmarks/*.jsonl --out artifacts/reports/paired_val
# Day 10, once, after the Day 9 freeze:
python scripts/m2_evaluate.py evaluate --manifest $M --predictions $P --protocol artifacts/reports/protocol.json \
    --split test --unseal-test --purpose "final held-out run" --out artifacts/reports/test
```

## What to commit

Code, tests, docs and small synthetic fixtures only. Results go in the report; if the team wants a
results file in git, commit only aggregated tables (no clip IDs, no media).
