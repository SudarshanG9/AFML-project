# Mandatory Day-1 checkpoint audit

Record **before coding against a model**:

- Model/repository/version, weights URL, license, SHA256 checksum.
- Is this a jointly trained AV deepfake detector? Is it actually sensitive to both audio and visual ablations?
- Which checkpoint training sources overlap with the evaluation dataset, clips, identities, speakers or original videos?
- Exactly which fusion block is modified? One attention direction or multiple? Shapes and masks for Q/K/V; pre/post projection boundaries.
- Are cached feature tensors before that exact fusion boundary, with identical normalization and temporal alignment to raw inference?
- What hardware/backend/dtype is actually supported? Is SDPA a drop-in replacement at that location?
- If no usable checkpoint is available by end of Day 1, formally switch to frozen pretrained encoders + a newly trained small fusion/classifier head and update the scope and training split.

Do not assume an available unofficial checkpoint is faithful to the paper without verification.
