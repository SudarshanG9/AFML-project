"""Explicit held-out cohort/path contracts; metadata construction never opens data."""
from pathlib import Path
import re
from avfuse.member1_fusion.dayc_guards import read_strict_json,sha256_file

OFFICIAL_M2_SHA='6ec2a65c90b3d91b93522fe8abbf09f914ad71499863e3c6ea32ff126b58d454'
CANONICAL_SHA='8339b2476a71e62c6c6fca7d937ba4cc7a704b09d26f08506c20d1f10de8f3dd'


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical_release(record):
    """Recognize the explicit selector in the supplied owner release, not a path guess."""
    identity=record.get('ready_test_manifest',{})
    return (record.get('release_id')=='M2-REL-20261113-01' and
        identity.get('path')=="Drive avfuse/m2_handoff/manifest_whole.jsonl (rows with split == 'test'; no separate test file, not regenerated)")


def cohort_contract(record):
    identity=record['ready_test_manifest']
    if canonical_release(record):
        require(identity['sha256']==record['canonical_manifest_sha256']==CANONICAL_SHA,'Canonical cohort identity differs')
        require(type(identity['expected_count']) is int and identity['expected_count']==2000,'Official cohort count differs')
        return {'kind':'canonical_split','split':'test','manifest_sha256':identity['sha256'],'expected_count':identity['expected_count']}
    return {'kind':'ready_manifest','split':'test','manifest_sha256':identity['sha256'],'expected_count':identity['expected_count']}


def validate_path_policy(policy):
    require(isinstance(policy,dict),'Missing held-out feature path policy')
    if policy=={'kind':'project_local_only'}:return policy
    require(set(policy)=={'kind','cache_root','split','feature_filename','metadata_filename'} and
        policy['kind']=='split_clip_cache' and policy['split']=='test' and
        policy['feature_filename']=='features.npz' and policy['metadata_filename']=='meta.json','Invalid held-out cache mapping policy')
    root=Path(policy['cache_root'])
    require(root.is_absolute() and '..' not in root.parts and str(root)==policy['cache_root'] and root!=Path('/'),'Unsafe cache root')
    return policy


def resolve_mapped_test_feature(row,policy):
    # Production callers must already hold C10 authorization and an immutable attempt.
    validate_path_policy(policy)
    clip=row['clip_id'];require(row['split']=='test' and isinstance(clip,str) and re.fullmatch('[A-Za-z0-9_-]+',clip),'Unsafe/non-test cache identity')
    base=Path(policy['cache_root']);folder=base/'test'/clip
    for path in (folder/'features.npz',folder/'meta.json'):
        require(path.is_file(),'Missing feature/metadata file: '+str(path))
        for parent in (path,*path.parents):require(not parent.is_symlink(),'Symlink cache path')
        require(path.resolve(strict=True).is_relative_to((base/'test').resolve(strict=True)),'Cache escaped frozen root')
    return folder/'features.npz'


def inspect_cache_identity(row,feature):
    """After C10 only: full tensor checks, metadata and acquisition hashes, no inference."""
    import numpy as np
    feature=Path(feature);meta=feature.parent/'meta.json'
    before=sha256_file(feature);metadata_sha=sha256_file(meta);record=read_strict_json(meta)
    require(record.get('status')=='ok','Cache metadata status is not ok')
    if 'clip_id' in record:require(record['clip_id']==row['clip_id'],'Cache metadata clip ID mismatch')
    with np.load(feature,allow_pickle=False) as data:
        for key in ('video_features','audio_features'):
            array=data[key]
            require(array.dtype==np.float32 and array.ndim==2 and 0<array.shape[0]<=600 and array.shape[1]==768,'Invalid original float32 feature shape/dtype')
            require(np.isfinite(array).all(),'Nonfinite cached features')
        require(abs(len(data['video_features'])-len(data['audio_features']))<=3,'Stream alignment exceeds original cache contract')
    require(sha256_file(feature)==before and sha256_file(meta)==metadata_sha,'Cache changed during inspection')
    return {'clip_id':row['clip_id'],'feature_sha256':before,'metadata_sha256':metadata_sha,
        'identity_basis':'C10-time acquisition hashes and owner cohort/provenance; no original per-file tensor hash inventory supplied'}
