# Checkpoints (ignored by git)

Record source URL, license, SHA256 hash, model architecture, training dataset(s),
input preprocessing, accessible Q/K/V boundary, and modality-ablation results.

**Mandatory gate:** check whether any evaluation source, identity, or clip was seen
in the checkpoint's original training. A new split of a previously seen dataset
cannot guarantee a genuinely independent test.
