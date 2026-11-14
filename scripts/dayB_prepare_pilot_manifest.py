"""Create a pilot-only manifest for committed, existing NPZ caches.

Never use this output in place of the 1,200-clip scientific validation manifest.
"""
import argparse
from collections import Counter
from pathlib import Path

from avfuse.member2_data.manifest import read_manifest, write_manifest


def build_pilot_manifest(source: Path, cache_root: Path, output: Path) -> dict:
    source = Path(source)
    cache_root = Path(cache_root).resolve()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    rows = read_manifest(source)
    if any(r['split'] != 'val' for r in rows):
        raise ValueError('Pilot source must be validation-only')
    selected = []
    missing = []
    for r in rows:
        path = (cache_root / 'val' / r['clip_id'] / 'features.npz').resolve()
        if not path.is_relative_to(cache_root):
            raise ValueError('Path escaped the pilot cache root')
        if path.is_file():
            item = dict(r)
            item['feature_path'] = str(path)
            selected.append(item)
        else:
            missing.append(r['clip_id'])
    if not selected:
        raise ValueError('No actual NPZ pilot caches exist')
    output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(selected, output)
    return {
        'status': 'PILOT_ONLY_NOT_FULL_VALIDATION',
        'source_records': len(rows),
        'available_records': len(selected),
        'missing_records': len(missing),
        'per_type': dict(Counter(r['manipulation_type'] for r in selected)),
        'output': str(output.resolve()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path,
        default=Path('data/pilot/test_day2/manifest_rows.jsonl'))
    p.add_argument('--cache-root', type=Path,
        default=Path('data/pilot/test_day2'))
    p.add_argument('--out', type=Path,
        default=Path('data/manifests/pilot_available_val.jsonl'))
    a = p.parse_args()
    print(build_pilot_manifest(a.source, a.cache_root, a.out))


if __name__ == '__main__':
    main()
