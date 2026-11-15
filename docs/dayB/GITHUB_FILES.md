# GitHub file layout

All implementation files are in the repository itself. No runtime source should exist solely under ignored artifact snapshots.

## Versionable project files

- `src/`: project modules.
- `scripts/`: inference, preflight, diagnostics, runtime/evaluation CLIs and Colab packaging sources.
- `tests/`: all unit/integration tests; no tests need to be uploaded from an artifact folder.
- `contracts/`: schemas.
- `configs/`: example/research definitions, environment reference and frozen package file manifests; no model weights or private dataset manifests.
- `requirements/` and `pyproject.toml`: dependency declarations.
- `docs/` and `README.md`: protocol, current status, summarized results, limitations and reproducibility instructions.
- `.gitignore` and `.env.example`: repository policy and placeholder environment example.

The original local experiment archives remain unchanged. Canonical copies of their G4/G5 helper code are now under `scripts/dayB_colab/`, and their exact package manifests are in `configs/dayB/`. Use the builder described in [README.md](README.md) to recover the original package layout with authorized private inputs.

## Files that stay local under the existing ignore policy

- `artifacts/`: full predictions, metrics, raw benchmark records, received evidence, audit inventories, ZIPs, historical snapshots and Member 2 handoffs.
- `data/`: private canonical/ready manifests, licensed media and feature caches.
- `checkpoints/`, weight/tensor files, external `third_party/` source checkout and runtime assets.
- Actual `.env` files, caches and operating-system metadata.

These files remain inside the project where already placed, or in the approved remote cache environment. Exclusion from Git does not mean that they were deleted. A GitHub clone requires private artifacts to be restored for research replay; source/tests and a results summary are available without those data.

The existing `.gitignore` already implements these exclusions. Do not force-add ignored datasets, archives or weights. Manual web uploads do not apply local Git ignore rules automatically; select the versionable source/documentation folders rather than uploading the whole local directory.

## Existing staging state

The preparation did not run git add/reset/rm/commit/push. Previously staged code and tracked `.DS_Store` changes were preserved. Read-only inspection also found already tracked pilot `.mp4`/`.npz` files and private pilot metadata. Existing ignore rules do not untrack them. These are not source-upload requirements; an index-only cleanup/review is still needed before uploading with Git, while keeping the local experiment files intact. No staging change or history rewrite was performed. The exact observed state and eligibility audit are stored locally in `artifacts/dayB/github_preparation/`.

## Member 2 deliverables

The local verified handoff ZIP contains three full 1,200-row validation prediction files, exact config definitions, fixed provisional calibration, three-mode metrics/CIs, paired recovery/per-type CIs, one matched benchmark set, README, provenance and hashes. See the local artifact index `PROJECT_FILES.md` after private outputs are restored. The handoff does not contain `restore_query_val.jsonl` or held-out predictions; those belong to later separately authorized stages.
