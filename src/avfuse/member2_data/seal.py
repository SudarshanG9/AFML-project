"""Test-set seal (plan Day 9: "sealed test remains untouched").

Every evaluation that touches the test split must pass unseal=True with a
purpose, and is appended to a ledger next to the frozen protocol. The final
report can then state exactly how many times test was opened and why. The
restoration table never accepts test at all (it is a selection step).
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path


class SealedTestError(PermissionError):
    pass


def ledger_path(protocol_path: str | Path) -> Path:
    return Path(protocol_path).with_name("test_access.jsonl")


def open_test(protocol_path: str | Path | None, purpose: str | None, configs, unseal: bool) -> None:
    if protocol_path is None:
        raise SealedTestError("test needs a frozen protocol (--protocol); calibrate and freeze on val first")
    if not unseal or not purpose:
        raise SealedTestError("the test split is sealed. For the final held-out run pass "
                              "--unseal-test --purpose \"...\" (the access is logged)")
    with ledger_path(protocol_path).open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": dt.datetime.now().isoformat(timespec="seconds"),
                            "purpose": purpose, "configs": sorted(configs)}) + "\n")


def accesses(protocol_path: str | Path) -> list[dict]:
    p = ledger_path(protocol_path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
