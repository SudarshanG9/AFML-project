"""Hash inventory from the accepted backend, independent of git availability in CI."""
import hashlib
import json
from pathlib import Path


def test_all_protected_backend_and_evidence_bytes_unchanged():
    root = Path(__file__).resolve().parents[2]
    inventory = json.loads((root / 'docs/integration/PHASE_4_PROTECTED_FILES.json').read_text())
    for entry in inventory['files']:
        if 'sha256' in entry:
            assert hashlib.sha256((root / entry['path']).read_bytes()).hexdigest() == entry['sha256'], entry['path']
