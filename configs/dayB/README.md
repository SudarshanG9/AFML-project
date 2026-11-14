# Day B versionable configuration metadata

- `precision_modes.json`: exact attention operand/output precision definitions; the query configuration is defined but not evaluated in Day B.
- `environment.reference.json`: accepted CPU software context, checkpoint/input hashes and 1,200-clip requirement. Contains no feature arrays, dataset labels or account credentials.
- `frozen_g4_package_manifest.json`, `frozen_g5_package_manifest.json`: original exact file/hash inventories used by the packaging-only builder. Do not regenerate them to accommodate mismatched files.

Private canonical/ready dataset manifests and accepted execution-context payloads stay in ignored `data/` and `artifacts/` locations. The original checkpoint/config and third-party implementation must be restored through the project's authorized handoff; weights are not included here.
