"""Validate schema and identity/source splits: python scripts/validate_manifest.py <file>"""
import argparse
from avfuse.shared.jsonl import read_jsonl
from avfuse.member2_data.manifest import verify_manifest

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    a = p.parse_args()
    records = read_jsonl(a.manifest)
    verify_manifest(records)
    print(f"Validated {len(records)} rows without source/identity split leakage")
