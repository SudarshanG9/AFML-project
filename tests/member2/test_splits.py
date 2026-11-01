from avfuse.member2_data.manifest import verify_manifest
from avfuse.member2_data.splits import group_clusters, resplit_by_group, split_audit


def row(clip, src, ident, split, mtype="real"):
    return {"clip_id": clip, "dataset": "toy", "source_id": src, "identity_id": ident, "split": split,
            "label": int(mtype != "real"), "manipulation_type": mtype, "feature_path": f"{clip}.pt"}


def test_groups_are_transitive():
    # a-b share a source, b-c share an identity -> one group; d alone
    rows = [row("a", "S1", "P1", "train"), row("b", "S1", "P2", "train"),
            row("c", "S2", "P2", "train"), row("d", "S3", "P3", "train")]
    g = group_clusters(rows)
    assert g["a"] == g["b"] == g["c"] != g["d"]


def test_transitive_leak_is_caught_and_repaired_without_redrawing_test():
    rows = [row("a", "S1", "P1", "train"), row("b", "S1", "P2", "val", "audio_fake"),
            row("c", "S2", "P2", "test", "video_fake"), row("d", "S3", "P3", "val")]
    audit = split_audit(rows)
    assert not audit["passed"] and audit["groups_crossing_splits"] == 1 and audit["clips_in_crossing_groups"] == 3
    fixed = resplit_by_group(rows)
    assert split_audit(fixed)["passed"]
    assert {r["clip_id"] for r in fixed if r["split"] == "test"} == {"a", "b", "c"}  # test only grows
    assert next(r for r in fixed if r["clip_id"] == "d")["split"] == "val"         # untouched
    verify_manifest(fixed)  # the shared validator agrees


def test_resplit_is_deterministic():
    rows = [row(f"c{i}", f"S{i // 2}", f"P{i}", "train" if i % 2 else "val") for i in range(40)]
    assert resplit_by_group(rows, seed=3) == resplit_by_group(rows, seed=3)
