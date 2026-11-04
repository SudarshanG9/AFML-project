"""DiMoDif feature extraction for the whole-video manifest (Member 2). Needs a GPU.

Runs the same two frozen encoders DiMoDif uses (auto-AVSR LRS3_V_WER19.1 visual and
LRS3_A_WER1.0 audio, from github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages)
with the MediaPipe face tracker, and saves DiMoDif's own format:
    <out>/<split>/<video_id>/features.npz  (video_features (T_v, D_v), audio_features (T_a, D_a))
plus meta.json with shapes and per-stage timings.

Differences from DiMoDif's external/features.py, none of which changes the features:
  * no transcription (beam search): DiMoDif's detector never reads the transcripts;
  * no facenet-pytorch import (only needed for the MTCNN option, and it pins an old PyTorch).

    python scripts/m2_dimodif_features.py --videos data/manifests/videos_whole.csv \
        --vsr-dir /content/vsr --out data/dimodif_emb --limit 1        # one-video test
    python scripts/m2_dimodif_features.py ... --split test              # all test videos (resumable)
    python scripts/m2_dimodif_features.py --summarize-only --out data/dimodif_emb
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

from avfuse.member2_data.dimodif_features import out_dir, pending, summarize, write_meta
from avfuse.member2_data.wholevideo import read_videos_csv


def load_models(vsr_dir, device):
    sys.path.insert(0, vsr_dir)
    from pipelines.data.data_module import AVSRDataLoader
    from pipelines.detectors.mediapipe.detector import LandmarksDetector
    from pipelines.model import AVSR

    def avsr(modality, name):
        d = os.path.join(vsr_dir, "data", name)
        return AVSR(modality, os.path.join(d, "model.pth"), os.path.join(d, "model.json"), device=device)

    return {
        "landmarks": LandmarksDetector(),
        "video_loader": AVSRDataLoader("video", detector="mediapipe"),
        "audio_loader": AVSRDataLoader("audio", detector="mediapipe"),  # as in DiMoDif; ignored for audio
        "video_model": avsr("video", "LRS3_V_WER19.1"),
        "audio_model": avsr("audio", "LRS3_A_WER1.0"),
    }


def environment(device):
    import torch
    import torchaudio
    import torchvision
    env = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
           "torchaudio": torchaudio.__version__, "device": device}
    if device.startswith("cuda") and torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
    try:
        import mediapipe
        env["mediapipe"] = mediapipe.__version__
    except Exception:
        pass
    return env


def extract_one(m, path, device):
    import torch
    sync = (lambda: torch.cuda.synchronize()) if device.startswith("cuda") else (lambda: None)
    t = {}
    t0 = time.perf_counter()
    landmarks = m["landmarks"](path)
    t["landmarks_ms"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    video = m["video_loader"].load_data(path, landmarks)
    t["video_load_crop_ms"] = (time.perf_counter() - t0) * 1e3
    if video is None:
        raise RuntimeError("no face landmarks in any frame")
    t0 = time.perf_counter()
    audio = m["audio_loader"].load_data(path)
    t["audio_load_ms"] = (time.perf_counter() - t0) * 1e3
    with torch.no_grad():
        sync(); t0 = time.perf_counter()
        v = m["video_model"].model.encode(video.to(device))
        sync(); t["video_encode_ms"] = (time.perf_counter() - t0) * 1e3
        sync(); t0 = time.perf_counter()
        a = m["audio_model"].model.encode(audio.to(device))
        sync(); t["audio_encode_ms"] = (time.perf_counter() - t0) * 1e3
    v, a = v.detach().cpu().numpy(), a.detach().cpu().numpy()
    no_lm = sum(x is None for x in landmarks)
    return v, a, no_lm, t


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", help="videos_whole.csv from m2_whole_video_manifest.py")
    p.add_argument("--vsr-dir", help="clone of Visual_Speech_Recognition_for_Multiple_Languages with models in data/")
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="all", choices=["val", "test", "all"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--summarize-only", action="store_true")
    a = p.parse_args()

    if not a.summarize_only:
        todo = pending(read_videos_csv(a.videos), a.out, a.split, a.limit)
        print(f"{len(todo)} videos to process")
        m = load_models(os.path.abspath(a.vsr_dir), a.device)
        env = environment(a.device)
        for i, vid in enumerate(todo, 1):
            meta = {"video_id": vid["video_id"], "split": vid["split"], "n_frames": vid["n_frames"],
                    "environment": env}
            try:
                v, au, no_lm, t = extract_one(m, vid["path"], a.device)
                d = out_dir(a.out, vid["split"], vid["video_id"])
                os.makedirs(d, exist_ok=True)
                np.savez(os.path.join(d, "features.npz"), video_features=v, audio_features=au)
                meta.update({"status": "ok", "T_v": int(v.shape[0]), "D_v": int(v.shape[1]),
                             "T_a": int(au.shape[0]), "D_a": int(au.shape[1]), "dtype": str(v.dtype),
                             "frames_no_landmarks": int(no_lm), "timings_ms": {k: round(x, 2) for k, x in t.items()}})
            except Exception as e:  # recorded, not fatal: the summary lists failures
                meta.update({"status": "failed", "error": f"{type(e).__name__}: {e}"})
            write_meta(a.out, vid["split"], vid["video_id"], meta)
            if meta["status"] == "ok":
                print(f"{i}/{len(todo)} ok   {vid['video_id']}  T_v={meta['T_v']} T_a={meta['T_a']} "
                      f"D={meta['D_v']}/{meta['D_a']}  frames={vid['n_frames']}")
            else:
                print(f"{i}/{len(todo)} FAIL {vid['video_id']}  {meta['error'][:150]}")

    rep = summarize(a.out)
    with open(os.path.join(a.out, "feature_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: rep[k] for k in rep if k not in ("timings_ms", "failed_examples")}, indent=2))
    print(f"wrote {os.path.join(a.out, 'feature_report.json')}")


if __name__ == "__main__":
    main()
