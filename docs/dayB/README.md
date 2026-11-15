# Day B implementation and reproducibility

The Day B workflow (original Days 5–7) has audited full-validation outputs for reference, low_precision and restore_value_output. Each file contains the same 1,200 approved clips, with 300 per category. G8/G9 technical checks pass; final human scientific acknowledgement remains pending. Day 8 restore_query, Day 9 policy freeze and held-out test are not completed by Day B.

- [Results and limitations](RESULTS.md)
- [Files to include in GitHub](GITHUB_FILES.md)
- [Precision definitions](../../configs/dayB/precision_modes.json)
- [Reference environment and input hashes](../../configs/dayB/environment.reference.json)
- [Research protocol](../EXPERIMENT_PROTOCOL.md)
- [Day B runbook](AFML_DayB_Coding_Agent_Runbook.md)

## Source layout

- `src/avfuse/member1_fusion/`: unchanged detector adapter, attention and fake-INT8 primitives.
- `src/avfuse/member2_data/`: schemas, manifest joins, evaluation, source bootstrap, paired recovery and sealed-test controls.
- `src/avfuse/member3_runtime/`: runtime/operator engines, validated benchmark artifacts and provenance.
- `scripts/day6_preflight.py`, `dayB_inference.py`, `dayB_model_benchmark.py`: core execution CLIs.
- `scripts/dayB_upstream_audit.py`, `dayB_matched_upstream_diagnostic.py`: separate whole-model and target-matched diagnostics.
- `scripts/dayB_g6/`, `dayB_g7/`, `dayB_finish/`: historical guarded execution runners. Their accepted-context payloads stay in ignored local artifact folders; restore those private inputs to replay the audited runs.
- `scripts/dayB_colab/g4/` and `g5/`: original packaging helper sources, copied byte-for-byte from the approved archives. These are package templates: the builder places them at `g4/` or `g5/` inside their original package layout. Do not run the template copies directly from `scripts/dayB_colab/`.
- `scripts/dayB_build_colab_package.py`: packaging-only builder for original G4/G5 file contents.
- `tests/member1/`, `tests/member2/`, `tests/member3/`, `tests/integration/`: tests, including package-integrity guards.
- `requirements/dayB-g4.txt` and `dayB-colab-cpu.txt`: historical dependency pins. The latter deliberately leaves the accepted CPU Torch installation unchanged.

## Rebuild a frozen G4/G5 package

The approved archive manifests are versionable small files in `configs/dayB/`. Licensed/private manifests, G4 evidence, approved workload metadata and upstream model assets stay outside Git. Obtain them through the project's authorized handoff and restore the original payload layout locally.

```bash
python scripts/dayB_build_colab_package.py \
  --gate g5 \
  --private-root artifacts/dayB/local_deliverables/g5_colab/AFML_G5_CPU_Colab_20261111T180139Z \
  --out artifacts/dayB/rebuilt_packages/AFML_G5_Colab_rebuilt.zip
```

For G4, use `--gate g4` and its original G4 private payload root. Every file must match its frozen hash before the builder creates an archive. Required project Python code must be present in the uploadable repository tree; it cannot be substituted from a hidden private snapshot. Approved third-party model source/config remain external assets. The output file must be new. ZIP container bytes/timestamps may differ, while the packaged file contents and original package-manifest bytes remain exact.

The builder performs no inference, data download, account operation or Git write. It does not fabricate a ready manifest, supply missing data, bypass gates or update the approved scientific protocol. A missing/mismatched input stops packaging with a path list.

## Private execution inputs

A fresh GitHub clone does not contain the validation cache, original checkpoint, private canonical/ready manifests, accepted execution contexts or prediction outputs. Restore authorized copies separately; use the documented checkpoint/config hashes and original cache layout `<cache-root>/val/<clip_id>/features.npz` with `meta.json`. The large cache remains in the user-controlled execution environment. Existing historical Colab instructions are preserved for provenance; obtain a new explicit authorization before any later research stage.

The public source tree and tests are self-contained for unit tests after project installation; reproducing research runs additionally requires the private inputs. No synthetic test output should be represented as research evidence.
