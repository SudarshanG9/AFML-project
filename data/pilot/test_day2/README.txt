20 val videos, 5 per manipulation_type, one per source_id.
val/<clip_id>/features.npz: video_features (T_v,768), audio_features (T_a,768), float32 (auto-AVSR outputs).
labels.csv: clip_id, label (1=fake), manipulation_type, source_id, identity_id, T_v, T_a, sha256.
identity_id = source_id (LAV-DF has no speaker IDs). Test split not included (sealed).
LAV-DF is CC BY-NC 4.0: team use only.
