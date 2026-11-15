# Naming and compatibility review

## Decision

No existing source/script/config path was renamed. The review found that the explicitly named day/stage files are bound to frozen package/C9 source inventories, historical execution/recovery paths or existing interfaces. Newly authored numbered chapters and purpose tables provide clarity without invalidating those bindings. This is an intentional provenance-preserving outcome, not a claim that cosmetic renaming is inherently unsafe.

[FILE_RENAME_MAP.csv](FILE_RENAME_MAP.csv) records every reviewed proposal with the requested nine columns. `new_path` is a **candidate**, not an implemented path, whenever status is rejected/retained. No alias modules or duplicate entry points were introduced. No imports or CLI arguments required adjustment. Existing descriptive adapter, precision, evaluator, paired, runtime, benchmark and provenance modules remain the primary functional navigation.

## Frozen and historical paths

The G4/G5 package manifests explicitly enumerate `day2_probe.py`, Day A scripts, the Day B runner/benchmark, preflight and Member 2 commands. C9 explicitly binds the research/guards/validation/freeze/held-out/cache-mapping/precision-audit sources. Their names and bytes are evidence. The later recovery/progress files also retain dynamic candidate/package/test references and original stage identity; they were not falsely declared identical to every C9 source entry. Historical directories retain nearby purpose indexes on the Member 1 branch. Member READMEs explain current responsibilities on each owner's cleanup branch.

## Frontend compatibility

Read-only inspection of `member3/frontend-only-updates` at `6de021adb93142c3f3a2383c7ae8f834101157c0` found the frontend invoking `avfuse.member3_runtime.cli` for smoke/validate-config/run, importing `avfuse.member3_runtime.artifacts.verify_run_integrity`, and displaying old producer script names. Those APIs and paths were retained. Changing such imports or producer references would require frontend edits, so the reviewed alternatives were rejected. No `frontend/**` path was imported, changed or deleted in any cleanup commit. The frontend application itself was not run; this is static interface verification plus backend tests, not a claim of end-to-end UI validation.

## Dependency checks and documentation

Source imports, script callers, dynamic module loading, schema/config paths, tests, frozen inventories, Member 3 loader/runner use and frontend-facing references were inspected. The new README and ten chapters link current descriptive code paths; legacy records are left as original stage narratives. Documentation links are checked on the composed final tree. The malformed existing upstream gitlink remains documented rather than silently rewritten; the user guide uses an explicit external clone/revision.

## Tests and remaining names

The Member 1 missing-private-fixture checks pass/skip explicitly (46 passed, 19 skipped in the targeted group). Member 2's suite has 46 passed and 3 optional-dependency skips. Member 3's 531 passing tests initially exposed two README contract checks after prose revision; their missing backend/stage descriptions were restored and both checks passed. The composed final suite and CLI/link/hash checks are recorded in the final validation files.

Remaining `dayA`, `dayB`, `dayC`, G4/G5, C9/C10/C10R and package revision identifiers distinguish historical phases and acceptances. Their purpose is documented rather than erased. No stable interface was renamed merely for snake-case consistency, and no historical archive was made to absorb every old file. Public media/cache removals are separately inventoried by the hygiene branch and are not presented as renames.
