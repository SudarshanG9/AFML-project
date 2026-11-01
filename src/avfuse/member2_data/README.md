# Member 2 — Dataset + forensic evaluation

**Owns:** data license and manipulation labels, source/identity-disjoint train/val/test
manifest, offline A/V preprocessing, temporal alignment, frozen encoder feature
caching, baseline/lowp/restored metrics and source-cluster bootstrap CIs.

**Implemented in starter:** manifest schema validation, source and identity leak
checks, `.pt` tensor-only cache loader, baseline stratified AUROC/AUPRC/recall.

**Requires real implementation:** decode/face track/audio feature extraction,
encoder-specific feature generation, validation threshold policy, paired cluster
bootstrap confidence intervals, and shortcuts audit.
