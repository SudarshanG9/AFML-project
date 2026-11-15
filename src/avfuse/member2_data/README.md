# Member 2: data preparation and statistical evaluation

The final experiment evaluates whole LAV-DF videos using cached DiMoDif encoder outputs. Earlier windowed preparation files remain as historical methods; they are not the final 3,200-video manifest. The canonical cohort has 1,200 validation and 2,000 test videos, with equal counts in the four manipulation categories. Source groups are separated; `identity_id = source_id` does not verify speaker separation.

## Current implementation

| Module | Responsibility |
| --- | --- |
| `labels.py`, `lavdf.py` | map modality manipulation flags; select and prepare dataset records |
| `wholevideo.py` | whole-video manifest and media mapping for DiMoDif |
| `dimodif_features.py` | resumable feature bookkeeping and aggregate metadata reports |
| `manifest.py`, `splits.py` | schema and connected source-group integrity |
| `cache.py` | legacy `.pt` feature-cache interface, retained for compatibility |
| `predictions.py`, `readiness.py` | prediction schema, checkpoint, coverage and readiness checks |
| `evaluate.py` | validation threshold; source-cluster bootstrap; AUROC, average precision, BCE, recall and real FPR |
| `paired.py` | aligned paired differences using the same resampled sources in both modes |
| `shortcut_control.py`, `shortcuts.py` | matching/auditing dataset shortcut measurements |
| `seal.py` | explicit test-access checks and access ledger |
| `report.py` | render stored evaluation outputs into tables and optional figures |

The executable encoder pipeline is `scripts/m2_dimodif_features.py`: auto-AVSR visual/audio encoders and MediaPipe preprocessing, writing `<root>/<split>/<video_id>/features.npz` (`video_features`, `audio_features`) and `meta.json`. `preprocess.py` remains an abstract starter interface; it is not the actual encoder implementation. Do not use it as evidence that raw-video runtime is implemented.

## Supported commands

Run these from the repository root after installation. `--help` is safe and requires no research assets:

```bash
python scripts/m2_lavdf.py --help
python scripts/m2_whole_video_manifest.py --help
python scripts/m2_dimodif_features.py --help
python scripts/m2_check_predictions.py --help
python scripts/m2_evaluate.py --help
python scripts/m2_paired.py --help
python scripts/m2_shortcut_control.py --help
```

`m2_evaluate.py calibrate` selects the reference operating threshold using validation only; `evaluate` consumes a frozen protocol. The current final protocol uses 0.6627789735794068. Never recalibrate it for reproducing the accepted study. The evaluator reads prediction files and labels; the inference runtime does not compute accuracy. Its final report extends paired outputs to real FPR and manipulation subsets; preserve that current reporting behavior alongside the older frozen scoring-source inventory.

Historical `m2_` script names are retained by frozen package manifests and existing callers. Their module names already explain their functions. The `m2` prefix identifies the evaluation owner, not an obsolete version. No duplicate CLI aliases are introduced.

Test with `PYTHONPATH=src python -m pytest tests/member2 -q`. These tests use synthetic inputs. The accepted held-out outputs are already saved; running tests does not authorize reopening the research test cohort. Licensed media, canonical test labels, full feature caches and private handoffs are external inputs, not files to add to a public checkout.
