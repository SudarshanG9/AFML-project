"""Fresh one-thread child; only path reads are adapted, Day B loops unchanged."""
import argparse
from pathlib import Path
import sys
from avfuse.member1_fusion.dayc_cache_mapping import ValidationCacheResolver
from avfuse.member1_fusion.dayc_guards import read_strict_json,require_sha256
from scripts.dayC_member1.run_day8 import load_release,observed_runtime,read_rows
from scripts.dayC_member1.colab_query_pilot import check_environment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind',choices=('validation','benchmark'))
    p.add_argument('--approval-file',type=Path,required=True)
    a,rest=p.parse_known_args()
    if rest and rest[0]=='--':rest=rest[1:]
    root=Path(__file__).resolve().parents[2]
    _,release=load_release(root,a.approval_file)
    check_environment(observed_runtime(),release['environment'],release['runtime_amendment']['allowed_version_changes'])
    mapping=release['validation_cache_mapping']
    resolver=ValidationCacheResolver(mapping['original_root'],mapping['mapped_root'])
    ready=read_rows(release['ready_manifest'])
    require_sha256(Path(release['ready_manifest']),release['ready_manifest_sha256'])
    require_sha256(Path(release['g4_report']),release['g4_report_sha256'])
    inventory={r['clip_id']:r for r in read_strict_json(release['g4_report'])['cache_inventory']}
    originals={str(Path(r['feature_path'])):r for r in ready}
    mapped={str(resolver.resolve(r)):r for r in ready}
    import torch
    torch.manual_seed(42)
    from scripts import dayB_inference
    original_load=dayB_inference.load_features
    def mapped_load(path):
        row=originals.get(str(path)) or mapped.get(str(path))
        if row is None:raise PermissionError('Feature outside approved validation cohort')
        resolved=resolver.resolve(row);identity=inventory[row['clip_id']]
        require_sha256(resolved,identity['feature_sha256'])
        require_sha256(resolved.parent/'meta.json',identity['metadata_sha256'])
        return original_load(resolved)
    if a.kind=='validation':
        dayB_inference.load_features=mapped_load
        entry=dayB_inference
    else:
        from scripts import dayB_model_benchmark
        dayB_model_benchmark.load_features=mapped_load
        entry=dayB_model_benchmark
    # No tensors, quantizer, model, feature contents or inference loop is replaced.
    sys.argv=[entry.__file__]+rest
    entry.main()


if __name__=='__main__':main()
