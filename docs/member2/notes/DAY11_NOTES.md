# Day 11 notes: checking the data for shortcuts

Files: `src/avfuse/member2_data/shortcuts.py`, `scripts/m2_audit.py`, `tests/member2/test_shortcuts.py`.

## The problem in one line

A model can get a high score by spotting something that has **nothing to do with the fake**,
for example "fake clips start with less silence". That's a **shortcut**, and it makes the score
look better than the model really is.

**Real example from research:** a CVPR 2025 paper found that in a popular audio-visual deepfake
dataset, the amount of silence at the start of a clip alone could predict real vs fake.

**Analogy:** if every fake banknote in a test pile had a coffee stain, a "detector" could just look
for stains. It would score well on that pile and fail in real life.

## What the audit measures, per clip

| Feature | Why it could leak |
| --- | --- |
| silence at the start / end | real and fake clips may be cut differently |
| loudness | edited audio may be louder or quieter |
| duration, number of frames | fakes may be shorter or longer |
| where the clip was cut (`cut_start_s`) | real clips start at 0 s, fakes are centred on the edit |
| codec, sample rate, fps, resolution (optional, via ffprobe) | fakes may be re-encoded differently |

Silence and loudness are measured on the whole stored clip **and** on the 2-second window the
model actually hears (`win_...` features).

## The test for each feature

For each feature: "if we used **only this feature** to guess real vs fake, how well would it work?"
That is measured as a **shortcut AUROC**:
- 0.5 = useless, like a coin toss (good: no shortcut)
- 1.0 = perfect separation (bad: a full shortcut)

**Rule, fixed in advance:** a feature with shortcut AUROC **>= 0.65** is **flagged**.
It is checked per split, overall and per fake type.

A flag doesn't automatically mean the model cheats. It means we must report it and, if possible,
run a control (for example, remove the silence and check whether the score drops).

## What we expect

With our clip cutting, `cut_start_s` will almost certainly be flagged: by design, real clips start
at 0 s and fakes are cut around the edit. It is not a model input, but it can change what the clip
contains (the first second of a video is often silent). So if silence is flagged too, that is the
likely reason.

With the DiMoDif model, which reads **whole videos**, this cut-position difference goes away,
because real and fake videos are both used from start to end.

## Running it

```
python scripts/m2_audit.py --clips <clips.csv> --manifest <manifest.jsonl> --out <folder>
```
Output: `shortcut_audit.json` and `audit.md` (a table per split plus the list of flags).

## Tests

| Test | Checks |
| --- | --- |
| `test_silence_profile` | leading/trailing silence measured correctly on a hand-made signal |
| `test_flags_injected_silence_shortcut` | when real clips are given extra silence on purpose, the audit flags it |
| `test_quiet_without_shortcut` | when silence is unrelated to the label, nothing is flagged |

## Status

Code done and tested. The real audit runs on Colab with the saved audio.

## Connection

- Uses the clips (Days 3-4) and the manifest labels.
- Its result goes into the final sign-off: the "no-leakage statement" in the ownership guide.
