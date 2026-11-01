# Private dataset and feature cache (ignored by git)

Expected local layout after dataset approval:

```
data/raw/              # original licensed clips, not committed
 data/cache/            # one tensor-only .pt per clip, not committed
 data/manifests/        # source-disjoint split manifests, not committed by default
```

`cache/<clip_id>.pt` must contain **frozen encoder output token tensors** named
`audio_tokens` and `video_tokens` with model-specific documented dimensions.
The checkpoint adapter must consume that precise feature format. Do not accidentally
cache already-fused representations and then claim to test the fusion operation.
If model-specific masks/timestamps are needed, extend the shared contract first.

Document sample rates, frames per second, timestamp alignment, face-selection policy,
encoder preprocessing, clip duration, provenance, and source/identity split logic.
Never share restricted dataset content publicly.
