# Experimental outputs (ignored by git)

`predictions/`: four prediction JSONL files by split and frozen configuration.
`benchmarks/`: operator-level and cached-feature model latency/memory JSONL.
`reports/`: manipulation-specific metrics and source-cluster bootstrap CIs.
`runs/`: config snapshots, checkpoint/data hashes, software/device provenance.

Do not add test labels to public prediction records. Keep dataset ground truth in
Member 2's evaluator. These directories are created by real experiments later.
