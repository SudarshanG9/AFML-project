import os

import numpy as np

from avfuse.member2_data.dimodif_features import STAGES, out_dir, pending, summarize, write_meta


def _videos():
    return [{"video_id": f"v{i}", "split": "val" if i < 3 else "test", "n_frames": 100 + i} for i in range(5)]


def _done(root, v, T_v, T_a, status="ok"):
    meta = {"video_id": v["video_id"], "split": v["split"], "n_frames": v["n_frames"], "status": status}
    if status == "ok":
        d = out_dir(root, v["split"], v["video_id"])
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, "features.npz"), video_features=np.zeros((T_v, 768), np.float32),
                 audio_features=np.zeros((T_a, 768), np.float32))
        meta.update({"T_v": T_v, "T_a": T_a, "D_v": 768, "D_a": 768, "dtype": "float32",
                     "frames_no_landmarks": 0, "timings_ms": {s: 10.0 for s in STAGES}})
    else:
        meta["error"] = "RuntimeError: no face"
    write_meta(root, v["split"], v["video_id"], meta)


def test_pending_resumes_and_filters(tmp_path):
    vids = _videos()
    _done(str(tmp_path), vids[0], 100, 100)
    assert [v["video_id"] for v in pending(vids, str(tmp_path))] == ["v1", "v2", "v3", "v4"]
    assert [v["video_id"] for v in pending(vids, str(tmp_path), split="test")] == ["v3", "v4"]
    assert len(pending(vids, str(tmp_path), limit=1)) == 1


def test_summary_reports_shapes_alignment_and_failures(tmp_path):
    vids = _videos()
    _done(str(tmp_path), vids[0], 100, 101)
    _done(str(tmp_path), vids[3], 103, 103)
    _done(str(tmp_path), vids[4], 0, 0, status="failed")
    rep = summarize(str(tmp_path))
    assert rep["n_ok"] == 2 and rep["n_failed"] == 1 and rep["per_split_ok"] == {"test": 1, "val": 1}
    assert rep["video_dim"] == [768] and rep["T_v_minus_T_a"]["min"] == -1
    assert rep["T_v_minus_video_frames"]["max"] == 0 and rep["longer_than_600"] == 0
    assert set(rep["timings_ms"]) == set(STAGES)
