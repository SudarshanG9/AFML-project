"""Read-only validation path mapping and exact G4 file-identity checks."""
import hashlib
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_mapping_release(release, parent):
    mapping = release['validation_cache_mapping']
    original = dict(release)
    original.pop('validation_cache_mapping')
    original['package_name'] = parent['package_name']
    require(original == parent, 'Mapping release changes approved runtime or protocol')
    require(mapping['original_root'] == parent['cache_root'] and mapping['validation_only'] is True,
            'Mapping must preserve original root and validation-only scope')
    for key in ('original_root', 'mapped_root'):
        path = Path(mapping[key])
        require(path.is_absolute() and '..' not in path.parts, 'Unsafe cache root')
    return mapping


class ValidationCacheResolver:
    def __init__(self, original_root, mapped_root):
        self.original = Path(original_root)
        self.mapped = Path(mapped_root)
        for path in (self.original, self.mapped):
            require(path.is_absolute() and '..' not in path.parts, 'Unsafe root')
        self.val = self.mapped / 'val'
        require(not self.val.is_symlink() and self.val.is_dir(), 'Missing or symlink validation directory')
        self.physical_val = self.val.resolve(strict=True)

    def resolve(self, row):
        clip = row['clip_id']
        require(isinstance(clip, str) and re.fullmatch(r'[A-Za-z0-9_-]+', clip) is not None,
                'Unsafe clip ID')
        require(row['split'] == 'val', 'Non-validation row')
        require(row['feature_path'] == str(self.original / 'val' / clip / 'features.npz'),
                'Original feature path differs')
        folder = self.val / clip
        require(not folder.is_symlink(), 'Symlink clip directory')
        for path in (folder / 'features.npz', folder / 'meta.json'):
            require(not path.is_symlink() and path.is_file(), 'Missing/symlink cache file: ' + str(path))
            require(path.resolve(strict=True).is_relative_to(self.physical_val), 'Cache path escapes validation')
        return folder / 'features.npz'


def verify_mapped_validation_cache(ready, inventory, original_root, mapped_root):
    require(len(ready) == len(inventory) == 1200, 'Exactly 1200 validation identities required')
    expected = {r['clip_id']: r for r in inventory}
    require(len(expected) == len({r['clip_id'] for r in ready}) == 1200,
            'Duplicate validation identity')
    require(set(expected) == {r['clip_id'] for r in ready}, 'Approved cohort ID sets differ')
    resolver = ValidationCacheResolver(original_root, mapped_root)
    records = []
    for row in sorted(ready, key=lambda r: r['clip_id']):
        approved = expected[row['clip_id']]
        require(approved['feature_path'] == row['feature_path'], 'G4 feature identity path differs')
        for key in ('source_id', 'manipulation_type'):
            require(approved[key] == row[key], 'G4 cohort metadata differs: ' + key)
        feature = resolver.resolve(row)
        result = {'clip_id': row['clip_id'], 'original_feature_path': row['feature_path'],
                  'resolved_feature_path': str(feature)}
        for key, path in (('feature_sha256', feature), ('metadata_sha256', feature.parent / 'meta.json')):
            require(isinstance(approved[key], str) and re.fullmatch('[0-9a-f]{64}', approved[key]),
                    'Missing approved hash: ' + key)
            actual = digest(path)
            require(actual == approved[key], 'Cache identity mismatch: ' + str(path))
            result[key] = actual
        records.append(result)
        if len(records) % 100 == 0:
            print('VALIDATION_CACHE_IDENTITIES_VERIFIED=' + str(len(records)) + '/1200', flush=True)
    return {'status': 'PASS', 'verified_count': 1200, 'original_root': str(original_root),
            'mapped_root': str(mapped_root), 'test_access': False, 'manifest_modified': False,
            'feature_arrays_loaded': False, 'hash_scope': 'all 1200 feature and metadata files',
            'records': records}
