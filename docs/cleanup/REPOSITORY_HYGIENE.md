# Public-tree hygiene

Removed 63 tracked generated or dataset-derived files/links from the proposed release tree. [REMOVED_FILES.csv](REMOVED_FILES.csv) records every original path, SHA-256 and reason. Original Git history and local files are retained; this does not purge previously published bytes. `git rm --cached` preserves ignored working files while excluding them from commits.

The existing `.gitignore` already excludes bytecode, installation metadata, media, tensors and generated runs. Removing tracked entries makes that policy effective. No source, test, schema, frozen config, approval, acknowledgement, original aggregate result or figure was deleted. Historical pilot JSON metadata remains as evidence; its media/cache inputs must be supplied privately.

`tests/member1/day6_integration_test.py` is an explicit asset-dependent executable whose historical default pilot path remains supported when supplied privately. Public unit tests do not need those media. The pre-existing DiMoDif gitlink lacks `.gitmodules`; it remains preserved as an upstream identity. The user guide supplies an explicit clone URL/revision.

No frontend path was touched. No private archive/cache was published. No held-out evaluator, inference or research benchmark ran.
