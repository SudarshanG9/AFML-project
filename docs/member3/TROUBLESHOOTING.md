# Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| exit 3, `UNSUPPORTED low_precision ...` | Member 1's adapter has not integrated that mode | Run `--modes reference` for development; request the mode from Member 1. Do not use `--allow-partial` output as a four-mode result. |
| exit 4 after a run | `--allow-partial`, `--max-clips`, or an unfinalized run | Re-run on the whole split with all requested modes supported. |
| exit 5 | Test split flags missing | Provide `--protocol <existing file> --unseal-test --purpose "..."` after the validation freeze. Never to "check quickly". |
| exit 6 `manifest not found` / `feature root not found` | Wrong path; features live on the team Drive, not Git | Use the whole-video manifest (`manifest_whole.jsonl`), and `--feature-root` containing `val/` and `test/`. |
| exit 6 `checkpoint or model config missing` | `<upstream>/ckpt/dfd/<stem>.pth/.json` absent | Check `--upstream-root`/`--checkpoint-name`. |
| exit 7 | Checkpoint SHA-256 differs from the expected value | Stop. Do not edit the expected hash or swap files; tell Member 1/2. |
| `FeatureLoadError` (exit 1) | NPZ with wrong keys, dim != 768, NaN, pickled arrays | The loader never repairs data; regenerate that video with Member 2's extractor. |
| `ERROR[USAGE] ... already exists` | Run id / output exists | Use a new `--run-id`/`--output`, or `--overwrite` deliberately. |
| `DeviceUnavailableError` | Requested cuda/mps not usable | Fix the device; there is intentionally no CPU fallback. |
| Run start is slow | Feature-cache hashing (1,200 val videos) | `--no-hash-cache` for development only; the run then has no cache hash. |
| `ModeFailedError ... outside [0, 1]` | Model returned invalid scores | Not clipped on purpose; report to Member 1. |
| `inspect` exit 1 with `integrity_ok=false` | An artifact changed after finalization | Treat the run as invalid; re-run. |

Failure artifacts: `execution_summary.json` (blocked/partial/complete) or `failure_summary.json` (exception) inside the run
directory. A failed run never contains a finalized `run_manifest.json`.
