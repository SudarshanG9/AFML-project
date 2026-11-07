"""Leakage / shortcut audit (plan Day 11; EXPERIMENT_PROTOCOL.md: "Check input
duration/silence/metadata shortcuts"; report ref [R8]).

A shortcut is any property that differs between real and fake clips but is not
the manipulation itself. Smeu et al. (CVPR 2025) found leading silence alone
predicts the label in a popular AV dataset. For each nuisance feature we compute
a "shortcut AUROC": how well the feature ALONE separates real from fake,
direction ignored (0.5 = useless, 1.0 = perfect shortcut).
Rule fixed in advance: shortcut AUROC >= 0.65 -> FLAG.

Expected on LAV-DF with our cutting rule: real clips start at 0 s of a video,
fakes are centred on the edit. Video openings are often silent, so real clips
may carry more leading silence. The audit measures both the stored 4 s clip and
the 2 s window the model actually hears (win_*).
"""
from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from avfuse.member2_data.lavdf import SAMPLE_RATE, SAMPLES_PER_FRAME, WindowSpec, eval_window, long_enough

SILENCE_RMS = 0.01   # about -40 dBFS for float audio in [-1, 1]
WINDOW_MS = 10
FLAG_AUROC = 0.65


def silence_profile(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    a = np.asarray(audio, dtype=np.float32)
    w = sr * WINDOW_MS // 1000
    n = len(a) // w
    if n == 0:
        return {"leading_silence_ms": 0.0, "trailing_silence_ms": 0.0, "rms_db": -120.0}
    rms = np.sqrt(np.mean(a[:n * w].reshape(n, w) ** 2, axis=1))
    loud = np.flatnonzero(rms >= SILENCE_RMS)
    lead, trail = (n * WINDOW_MS, n * WINDOW_MS) if len(loud) == 0 else \
        (loud[0] * WINDOW_MS, (n - 1 - loud[-1]) * WINDOW_MS)
    return {"leading_silence_ms": float(lead), "trailing_silence_ms": float(trail),
            "rms_db": float(20 * np.log10(max(float(np.sqrt(np.mean(a ** 2))), 1e-6)))}


def probe_media(path: str) -> dict:
    """Codec, sample rate, fps, resolution, bitrate and file size via ffprobe (raw video must exist)."""
    try:
        info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json",
                                          path], capture_output=True, check=True, text=True).stdout)
    except (OSError, subprocess.CalledProcessError):
        return {}
    out = {}
    fmt = info.get("format", {})
    if fmt.get("bit_rate"):
        out["container_bitrate_kbps"] = float(fmt["bit_rate"]) / 1000.0
    if fmt.get("size"):
        out["file_size_mb"] = float(fmt["size"]) / 1e6
    for st in info.get("streams", []):
        if st.get("codec_type") == "audio":
            out.update(audio_codec=st.get("codec_name"), audio_sample_rate=st.get("sample_rate"))
        elif st.get("codec_type") == "video":
            out.update(video_codec=st.get("codec_name"), video_fps=st.get("avg_frame_rate"),
                       video_resolution=f"{st.get('width')}x{st.get('height')}")
    return out


def clip_features(clip: Mapping, probe: bool = False) -> dict:
    """clip: a clips.csv row (has the audio path, lengths and cut position)."""
    f: dict = {"n_frames": float(clip["n_frames"]), "cut_start_s": float(clip.get("start_seconds") or 0.0)}
    if clip.get("audio") and os.path.exists(clip["audio"]):
        audio = np.load(clip["audio"]).astype(np.float32)
        f.update(silence_profile(audio))
        f["audio_duration_s"] = len(audio) / SAMPLE_RATE
        row = dict(clip, n_samples=len(audio))
        if long_enough(row):
            _, a0 = eval_window(row)
            win = audio[a0:a0 + WindowSpec().clip_frames * SAMPLES_PER_FRAME]
            f.update({"win_" + k: v for k, v in silence_profile(win).items()})
    if probe and clip.get("source") and os.path.exists(clip["source"]):
        f.update(probe_media(clip["source"]))
    return f


def shortcut_auroc(y, v) -> float:
    y, v = np.asarray(y), np.asarray(v, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 2 or len(set(y[ok].tolist())) < 2 or len(set(v[ok].tolist())) < 2:
        return 0.5 if ok.sum() >= 2 and len(set(y[ok].tolist())) == 2 else float("nan")
    a = roc_auc_score(y[ok], v[ok])
    return float(max(a, 1 - a))


def _numeric(y, v) -> dict:
    v = np.asarray(v, dtype=float)
    s = shortcut_auroc(y, v)
    out = {"shortcut_auroc": s, "flag": bool(np.isfinite(s) and s >= FLAG_AUROC)}
    for name, m in (("real", y == 0), ("fake", y == 1)):
        x = v[m & np.isfinite(v)]
        out[name + "_median"] = float(np.median(x)) if len(x) else None
    return out


def _categorical(y, v) -> dict:
    table: dict = {}
    for yy, vv in zip(y, v):
        table.setdefault(str(vv), [0, 0])[int(yy)] += 1
    best = max((shortcut_auroc(y, [float(str(x) == k) for x in v]) for k in table), default=float("nan"))
    return {"values_real_fake": table, "shortcut_auroc": best, "flag": bool(np.isfinite(best) and best >= FLAG_AUROC)}


def shortcut_audit(manifest: Sequence[Mapping], features: Mapping[str, dict]) -> dict:
    """features: clip_id -> clip_features(...). Per split, overall and per attack type."""
    names = sorted({k for f in features.values() for k in f})
    report: dict = {"rule": f"flag if shortcut AUROC >= {FLAG_AUROC}", "splits": {}, "flags": []}
    for split in sorted({m["split"] for m in manifest}):
        rows = [m for m in manifest if m["split"] == split and m["clip_id"] in features]
        block = {}
        for group in ("all", "audio_fake", "video_fake", "both_fake"):
            sel = [m for m in rows if group == "all" or m["manipulation_type"] in ("real", group)]
            y = np.array([int(m["label"]) for m in sel])
            if len(set(y.tolist())) < 2:
                continue
            gb = {}
            for n in names:
                vals = [features[m["clip_id"]].get(n) for m in sel]
                if all(v is None for v in vals):
                    continue
                numeric = all(v is None or isinstance(v, (int, float)) for v in vals)
                gb[n] = _numeric(y, [np.nan if v is None else v for v in vals]) if numeric else _categorical(y, vals)
                if gb[n]["flag"]:
                    report["flags"].append(f"{split}/{group}: {n} (shortcut AUROC {gb[n]['shortcut_auroc']:.3f})")
            block[group] = gb
        report["splits"][split] = block
    report["passed"] = not report["flags"]
    return report


# ---------------------------------------------------------------- whole videos (DiMoDif unit)

def video_audio(path: str) -> np.ndarray:
    """Whole audio track as float32 mono 16 kHz (same decode settings as the clip pipeline)."""
    from avfuse.member2_data.lavdf import _ffmpeg
    raw = _ffmpeg(["-i", path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"])
    return np.frombuffer(raw, dtype=np.float32)


def whole_video_features(video: Mapping, probe: bool = True) -> dict:
    """video: a videos_whole.csv row. No cut position exists here: every video is used whole,
    so the cut-position shortcut of the clip pipeline cannot occur."""
    f: dict = {"duration_s": float(video["duration_s"]), "n_frames": float(video["n_frames"])}
    try:
        audio = video_audio(video["path"])
        f.update(silence_profile(audio))
        f["audio_duration_s"] = len(audio) / SAMPLE_RATE
    except (OSError, subprocess.CalledProcessError) as e:
        f["audio_error"] = type(e).__name__
    if probe:
        f.update(probe_media(video["path"]))
    return f


def audit_markdown(rep: dict, title: str = "Shortcut (leakage) audit") -> str:
    lines = [f"# {title}", "", f"Rule fixed in advance: a feature is flagged if it alone separates real from fake "
             f"with AUROC >= {FLAG_AUROC} (0.5 = useless, 1.0 = perfect shortcut).", ""]
    fmt = lambda v: "" if v is None else f"{v:.3g}"
    for split, block in rep["splits"].items():
        for group, feats in block.items():
            lines += [f"## {split} / {group}", "", "| Feature | real median | fake median | shortcut AUROC | flag |",
                      "| --- | --- | --- | --- | --- |"]
            for n, s in feats.items():
                if "values_real_fake" in s:
                    vals = ", ".join(f"{k}: {v[0]} real / {v[1]} fake" for k, v in list(s["values_real_fake"].items())[:4])
                    lines.append(f"| {n} | {vals} | | {s['shortcut_auroc']:.3f} | {'**FLAG**' if s['flag'] else ''} |")
                else:
                    lines.append(f"| {n} | {fmt(s.get('real_median'))} | {fmt(s.get('fake_median'))} | "
                                 f"{s['shortcut_auroc']:.3f} | {'**FLAG**' if s['flag'] else ''} |")
            lines.append("")
    lines += ["## Flags", ""] + ([f"- {f}" for f in rep["flags"]] or ["- none"])
    return "\n".join(lines) + "\n"
