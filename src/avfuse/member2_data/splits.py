"""Group-disjoint split audit and repair (plan Day 2).

Two clips must stay in the same split if they share a source video OR an
identity, and this is transitive: A shares a source with B, B shares a person
with C, so A and C belong together. Those transitive groups are the connected
components of a graph whose nodes are clips and whose edges are "same source"
or "same identity". docs/CONTRACTS.md asks for exactly this ("connected-group
partition"). The same groups are used later for cluster bootstrap CIs.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence


class _UnionFind:
    def __init__(self):
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def group_clusters(rows: Sequence[Mapping]) -> dict[str, str]:
    """clip_id -> cluster id (a stable name: the smallest key in the component)."""
    uf = _UnionFind()
    for r in rows:
        clip = ("clip", r["clip_id"])
        uf.union(clip, ("source", r["dataset"], r["source_id"]))
        uf.union(clip, ("identity", r["dataset"], r["identity_id"]))
    names: dict = {}
    out = {}
    for r in rows:
        root = uf.find(("clip", r["clip_id"]))
        names.setdefault(root, "|".join(map(str, root[1:])))
        out[r["clip_id"]] = names[root]
    return out


def split_audit(rows: Sequence[Mapping]) -> dict:
    """Counts and leaks. 'passed' means no group of any kind crosses splits."""
    clusters = group_clusters(rows)
    splits_of = defaultdict(set)
    counts: dict = defaultdict(lambda: defaultdict(int))
    for r in rows:
        splits_of[clusters[r["clip_id"]]].add(r["split"])
        counts[r["split"]][r["manipulation_type"]] += 1
    leaking = {c: sorted(s) for c, s in splits_of.items() if len(s) > 1}
    per_split_groups = defaultdict(set)
    for r in rows:
        per_split_groups[r["split"]].add(clusters[r["clip_id"]])
    return {
        "n_clips": len(rows),
        "n_groups": len(splits_of),
        "counts": {s: dict(c) for s, c in counts.items()},
        "groups_per_split": {s: len(g) for s, g in per_split_groups.items()},
        "groups_crossing_splits": len(leaking),
        "clips_in_crossing_groups": sum(1 for r in rows if clusters[r["clip_id"]] in leaking),
        "examples": dict(list(leaking.items())[:5]),
        "passed": not leaking,
    }


def resplit_by_group(rows: Sequence[Mapping], val_fraction: float = 0.15, seed: int = 0) -> list[dict]:
    """Repair leaks by moving whole groups. Test is never redrawn: any group
    with a test clip goes entirely to test, so test can only grow. Other
    groups keep their split unless they cross train/val, in which case a hash
    of (seed, group) decides, so the result is identical on every machine."""
    clusters = group_clusters(rows)
    splits_of = defaultdict(set)
    for r in rows:
        splits_of[clusters[r["clip_id"]]].add(r["split"])
    out = []
    for r in rows:
        r = dict(r)
        c = clusters[r["clip_id"]]
        s = splits_of[c]
        if "test" in s:
            r["split"] = "test"
        elif len(s) > 1:
            u = int(hashlib.md5(f"{seed}:{c}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            r["split"] = "val" if u < val_fraction else "train"
        out.append(r)
    return out
