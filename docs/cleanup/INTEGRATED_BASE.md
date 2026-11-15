# Backend foundation for the five cleanup branches

Prepared on 15 November 2026. Existing remote branches remain unchanged. This is an explicit tree composition on detached `member1`, with one parent, rather than a Git merge. The common foundation is the parent of each workstream's first cleanup commit. Exact source tips are in [SOURCE_BRANCHES.json](SOURCE_BRANCHES.json).

* Member 1 source, recovery contracts, frozen configs, historical scripts and records: `9eb32dec1ecf9a0c901b520bcd7252b8ecb60a8b`.
* Member 2 source, evaluator/paired CLI, tests and documents: `a6c9bf786efe5e5290baf9ec24d5efb7d12fda80` (the current main, including final reporting extensions).
* Member 3 source and tests: `6de021adb93142c3f3a2383c7ae8f834101157c0` (backend paths only). This brings the later four-mode runtime alongside the accepted Member 1 adapter without importing frontend code.

No frontend path was imported, edited or deleted. No historical inference, evaluator or benchmark was run. The accepted experiment source is identified separately from today's integrated runtime: the frozen policy records old hashes for Member 2 paired reporting and Member 3 CLI. Those historical hashes are retained, and their source revisions remain in Git; the present integration must not be represented as the identical C9 source bundle.

Initial isolated-checkout verification: 1,078 passed, 3 skipped, 1 failed and 18 setup errors. All 19 failures/errors were absent private historical fixtures (reference validation predictions, execution archives, and recovery candidate sources). Public runtime/evaluator tests passed. Workstream changes make unavailable-fixture skips explicit; they do not insert private data or weaken production guards. The external DiMoDif checkout is not populated and its historical gitlink has no `.gitmodules` mapping, so real checkpoint inference is an asset-dependent check.

Each cleanup branch adds only its owned changes after this shared foundation. Review the common foundation once, then the individual workstream delta. The project lead performs all final merges manually.
