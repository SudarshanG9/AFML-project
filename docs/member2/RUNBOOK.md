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

## Whole-video manifest for DiMoDif (val + test)

DiMoDif reads whole videos, so its manifest has one row per video (same sample as above):

```bash
python scripts/m2_whole_video_manifest.py --root data/raw/LAV-DF --cap val=300 test=500 \
    --out data/manifests/manifest_whole.jsonl --videos-out data/manifests/videos_whole.csv --check-files
```

## DiMoDif features (GPU, val + test)

DiMoDif needs PyTorch 2.5.1 and MediaPipe 0.10.15, which have no wheels for Colab's newer
Python, so the extraction runs in a separate Python 3.10 environment:

```bash
git clone --depth 1 https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages /content/vsr
pip install uv gdown
uv venv /content/py310 --python 3.10
uv pip install --python /content/py310/bin/python torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
uv pip install --python /content/py310/bin/python "numpy<2" mediapipe==0.10.15 hydra-core av scikit-image scipy six opencv-python jsonschema
uv pip install --python /content/py310/bin/python -e . --no-deps
# encoder models into /content/vsr/data: LRS3_V_WER19.1 (http://bit.ly/40EAtyX), LRS3_A_WER1.0 (http://bit.ly/3ZSdh0l)
/content/py310/bin/python scripts/m2_dimodif_features.py --videos data/manifests/videos_whole.csv \
    --vsr-dir /content/vsr --out <drive>/dimodif_emb --split all
```
Resumable: rerunning skips finished videos. Results: `docs/member2/results/FEATURE_CACHE_RESULTS.md`.

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

## When Member 1's predictions arrive (whole videos)

```bash
M=data/manifests/manifest_whole.jsonl
P="reference_val.jsonl"            # later: all four configs' files
# 1. contract + checkpoint hash + score direction (stops if INVERTED)
python scripts/m2_check_predictions.py --manifest $M --predictions $P --out artifacts/reports/check.json
# 2-3. freeze the threshold on reference val, then evaluate val
python scripts/m2_evaluate.py calibrate --manifest $M --predictions $P --out artifacts/reports/protocol.json
python scripts/m2_evaluate.py evaluate --manifest $M --predictions $P --protocol artifacts/reports/protocol.json \
    --split val --out artifacts/reports/val
# overlap with the checkpoint's training data (needs LAV-DF metadata.json)
python scripts/m2_train_overlap.py --metadata data/raw/LAV-DF/metadata.json --manifest $M \
    --out artifacts/reports/train_overlap.json
```

## What to commit

Code, tests, docs and small synthetic fixtures only. Results go in the report; if the team wants a
results file in git, commit only aggregated tables (no clip IDs, no media).
