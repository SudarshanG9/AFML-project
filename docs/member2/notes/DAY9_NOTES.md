# Day 9 notes: freezing the rules and sealing the test set

Files: `src/avfuse/member2_data/seal.py`, `scripts/m2_evaluate.py` (test option),
`tests/member2/test_seal.py`.

## The problem in one line

If we look at the test results, then go back and change something (the threshold, the data, the
chosen restore), the test set stops being a fair exam. It has become practice material.

**Analogy:** an exam is only fair if the paper stays sealed until the exam. If you peek at it and
then "study", your score no longer shows what you really know.

## What gets frozen

On Day 5-6, `calibrate` writes `protocol.json` with:
- the **threshold** (chosen on validation, 5% of real clips flagged),
- a **fingerprint (SHA-256 hash) of the manifest**,
- the **checkpoint hash** of the model that made the predictions.

From then on, evaluation **refuses to run** if:
- the manifest has changed (its fingerprint no longer matches), or
- the predictions come from a different checkpoint.

So nobody can quietly swap the data or the model after the rules were fixed.

## How the test set is sealed (`seal.py`)

Evaluating on **test** needs all three:
1. a frozen `protocol.json`,
2. the flag `--unseal-test`, and
3. a written reason: `--purpose "final held-out run"`.

Without them, the script stops with `SEALED`.

Every time test **is** opened, one line is added to `test_access.jsonl`: the time, the reason and
which configs were scored. At the end we can state exactly how many times test was opened, and why.
The goal is **once**.

## What stays on validation only

- choosing the threshold (Day 5-6),
- deciding whether a restore helps (Day 7-8).

Both are decisions. Making them on test would bias the final result.

## Running it (once, at the very end)

```
python scripts/m2_evaluate.py evaluate --manifest <manifest.jsonl> --predictions <preds.jsonl> \
    --protocol protocol.json --split test --unseal-test --purpose "final held-out run" --out <folder>
```

## Tests

| Test | Checks |
| --- | --- |
| `test_test_split_is_sealed_and_logged` | test is refused without a protocol, refused without the flag (nothing logged), and allowed with flag + reason (logged once) |
| `test_changed_manifest_refused` | a manifest changed after freezing is rejected |

## Connection

- Freezes the decisions from Days 5-8.
- Day 10 is the one allowed opening of the test set.
