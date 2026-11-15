# Final cleanup and review report

## Scope and common foundation

Five new branches share `18d4af3fa05c9453eafb4666f78f0d5867bc3437`. The foundation explicitly composes Member 1's accepted implementation/recovery safeguards, main's final Member 2 evaluation reporting and later integrated Member 3 runtime. [The source map](docs/cleanup/SOURCE_BRANCHES.json) pins intake tips; [the foundation record](docs/cleanup/INTEGRATED_BASE.md) explains the single-parent tree composition. No Git merge was executed, no existing remote branch was updated, and no history was rewritten.

## Reviewable branch deltas after the foundation

| New branch | Work completed | Scope |
| --- | --- | --- |
| `cleanup/member1-backend` | replace stale component status with working adapter/intervention navigation; add historical-script index; explicitly skip only missing private fixture-dependent cases | Member 1 docs/tests; production frozen source unchanged |
| `cleanup/member2-data-evaluation` | document actual whole-video extractor, canonical cohort, final evaluator and paired reporting, retained historical entry points | Member 2 package guide; final reporting supplied by foundation |
| `cleanup/member3-runtime` | replace superseded reference-only notes with current four-mode CLI, timing boundaries, provenance, backend and stage APIs | Member 3 package guide; later runtime supplied by foundation |
| `cleanup/repository-hygiene` | untrack 63 generated metadata/bytecode/installation files or pilot media/cache/symlink entries; inventory original hashes | future public tree only; originals/history retained |
| `cleanup/research-documentation` | original README and ten research chapters; complete saved-result tables, claim index, naming audit and validation reports | authored research account; no private inputs copied |

Member 2/3 cleanup did not alter scientific algorithms merely to manufacture code churn. Their important code integration is explicit in the shared foundation. No source renames were accepted because provenance or stable callers outweighed stylistic benefit. [Rename audit](RENAMING_AND_COMPATIBILITY_REPORT.md) records the candidates and reasons.

## Removed, preserved and reorganized files

The hygiene branch's `docs/cleanup/REMOVED_FILES.csv` is the exact 63-path removal inventory with original SHA-256 values. Categories are operating-system metadata, tracked Python bytecode, local egg-info, dataset-derived pilot MP4/NPZ files and the machine-specific `data/day2_features` symlink. `git rm --cached` preserves local working copies while removing tracked release entries. Existing ignore rules already cover these categories.

All production source, essential tests, schemas, frozen configurations, approvals/acknowledgements, original aggregate result files and figures remain. Historical pilot JSON metadata remains as evidence with external inputs required. Private local evidence packages, licensed media, feature populations, proposal/Elicit references, sealed labels and clip-level prediction files are not newly published. No original evidence file was moved or rewritten. New research chapters create a coherent reading path alongside original records; they do not rename old notes and call them final reports.

## Research and compatibility findings

The accepted final saved evidence supports no statistically established overall degradation or V/output restoration advantage. No equivalence, reliable equal-cost restoration, native INT8 speedup, measured memory saving or raw-video runtime is claimed. The records disclose test-summary exposure, checkpoint-selection bias, source/speaker distinction, interrupted attempts, incomplete historical feature-array audit and Member 3 nested actor inconsistency. AuthentiQ primary full text, a standalone pretrained-model assessment, older Elicit-referenced PDFs and original Undermind output were unavailable; exact final shell invocation/full original cache baseline remain unverified.

CLI/module paths, six-field prediction schema, cached NPZ/PT formats, original masks and sealed-test safeguards remain. Static frontend inspection confirms its CLI/artifact imports remain available. No frontend file was touched. The current runtime and final evaluator reporting extend older frozen source inventories; accepted old hashes are preserved rather than relabeled as today's code.

## Validation

Initial integration: 1,078 passed, 3 skipped, 1 failure/18 errors, all from absent private historical fixtures. The Member 1 change makes those precise asset-dependent cases skip while retaining synthetic safety tests. Final composed-suite result: **1,078 passed, 22 skipped, zero failures/errors**. All 14 CLI smoke/help checks pass. The link, original-hash and numerical table checks are recorded in `docs/cleanup/FINAL_VALIDATION.json` and the [chapter checklist](DOCUMENTATION_VALIDATION_CHECKLIST.md). No held-out inference/evaluation, training, raw-media extraction or new scientific benchmark was run. Synthetic timing inside software tests/smoke is not a research measurement.

## Recommended manual merge order

1. Review the shared foundation once, then merge `cleanup/member1-backend` to bring it and the historical-fixture test handling into the chosen target.
2. Merge `cleanup/member2-data-evaluation`.
3. Merge `cleanup/member3-runtime`.
4. Merge `cleanup/repository-hygiene`.
5. Merge `cleanup/research-documentation` after reviewing all ten chapters and their source limits.

The lead performs every merge manually. If the target is the frontend integration branch, preserve all its frontend files during conflict resolution and take only the backend/documentation decisions after review. These backend-only branches do not remove or replace the frontend. Existing branch histories differ, so the composed local tree check does not guarantee a conflict-free future merge into every possible target.
