"""User-run exact-byte restoration from backups; never fabricate or run inference."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import uuid
import zipfile


def sha_bytes(data):return hashlib.sha256(data).hexdigest()


def sha_file(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):d.update(b)
    return d.hexdigest()


def validate_destination(root, path, expected):
    root=Path(root).resolve(strict=True);path=Path(path)
    if not path.is_absolute():path=root/path
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise PermissionError('Unsafe destination: '+str(path))
    if os.path.lexists(path):
        if not path.is_file() or sha_file(path)!=expected:
            raise FileExistsError('Existing destination differs; preserving it: '+str(path))
        return path,True
    return path,False


def write_original(root,path,data,expected):
    path,existing=validate_destination(root,path,expected)
    if existing:return 'PRESERVED_MATCHING_FILE'
    if sha_bytes(data)!=expected:raise ValueError('Original source hash mismatch')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.parent/('.restore_'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        os.link(temporary,path)  # Atomic, exclusive; never replace a destination.
        if sha_file(path)!=expected:raise RuntimeError('Restored output hash mismatch')
    finally:
        temporary.unlink(missing_ok=True)
    return 'RESTORED_EXACT_ORIGINAL_BYTES'


def restore_from_backups(root, targets, archive_paths, loose_paths=()):
    """Verify all required sources/destinations first, then restore missing files.

    targets: label -> {path, sha256, names}. No code from ZIPs is executed and
    no archive is extracted wholesale. Canonical manifest bytes are not parsed.
    """
    root=Path(root).resolve(strict=True);plan={};sources={};report=[]
    for label,item in targets.items():
        path,exists=validate_destination(root,Path(item['path']),item['sha256'])
        plan[label]=(path,exists)
    missing={name for name,(_,exists) in plan.items() if not exists}
    for path in map(Path,loose_paths):
        if not path.is_file() or path.is_symlink():continue
        labels=[name for name in missing if path.name in targets[name]['names']]
        if labels and path.stat().st_size<=32*1024*1024:
            data=path.read_bytes();digest=sha_bytes(data)
            for name in labels:
                if digest==targets[name]['sha256']:
                    sources[name]=(data,str(path));missing.remove(name)
    for archive in map(Path,archive_paths):
        if not missing:break
        with zipfile.ZipFile(archive) as z:
            if len(z.infolist())>20000:continue
            for info in z.infolist():
                p=PurePosixPath(info.filename)
                if (info.is_dir() or p.is_absolute() or '..' in p.parts
                        or any(part.lower() in {'test','train'} for part in p.parts)
                        or info.file_size>32*1024*1024):continue
                labels=[name for name in missing if p.name in targets[name]['names']]
                if not labels:continue
                data=z.read(info);digest=sha_bytes(data)
                for name in labels:
                    if digest==targets[name]['sha256']:
                        sources[name]=(data,str(archive)+'!'+info.filename);missing.remove(name)
    if missing:
        raise FileNotFoundError('No verified original found for: '+', '.join(sorted(missing))+
            '. No destination was written. Upload the provided Verified Original Inputs ZIP; do not regenerate these files.')
    for label,item in targets.items():
        path,exists=plan[label]
        if exists:state='PRESERVED_MATCHING_FILE';origin='existing destination'
        else:
            data,origin=sources[label];state=write_original(root,path,data,item['sha256'])
        report.append({'asset':label,'path':str(path),'sha256':sha_file(path),'status':state,'source':origin})
    return report


def targets_from_release(c):
    targets={key:{'path':c[key],'sha256':c[key+'_sha256'],'names':[Path(c[key]).name]}
        for key in ('g5_context','canonical_manifest','validation_manifest','ready_manifest','g4_report')}
    targets['g5_context']['names']+=['accepted_g5_context.json','accepted_g5_execution_context.json']
    for asset in c['runtime_assets']:
        name=Path(asset['path']).name
        label='checkpoint' if name.endswith('.pth') else 'model_source' if name=='models.py' else 'model_config'
        targets[label]={'path':asset['path'],'sha256':asset['sha256'],'names':[name]}
    targets['reference_predictions']={'path':c['g5_run']+'/predictions/dayB_reference_val.jsonl',
        'sha256':c['dayb_reference_anchor']['prediction_file_sha256'],
        'names':['dayB_reference_val.jsonl','reference_val.jsonl']}
    return targets


def run_colab_restoration():
    content=Path('/content');addon=content/'AFML_DayC_C8_2_CPU_Pilot_Drift_v2'
    release_path=addon/'configs/dayC/member1/colab_pilot_release.json'
    if sha_file(release_path)!='ea6edfdab55ac88c7f6fa08ce1f472819b6a335b4dbd66aeb758d133b7a93bd6':
        raise ValueError('V2 release does not match approved configuration')
    c=json.loads(release_path.read_text())
    report=restore_from_backups(content,targets_from_release(c),sorted(content.glob('*.zip')),
        [addon/'assets/accepted_g5_context.json',content/'dfd_lavdf_reduceonplateau_64_4_1_5_True.pth'])
    print('ORIGINAL_FILES_RESTORED\n'+json.dumps(report,indent=2))
    ready=Path(c['ready_manifest']);rows=[json.loads(line) for line in ready.read_text().splitlines() if line.strip()]
    if len(rows)!=1200 or any(row['split']!='val' for row in rows):raise ValueError('Not the approved full validation cohort')
    cache=Path(c['cache_root']);present=0
    for row in rows:
        feature=Path(row['feature_path'])
        if feature!=cache/'val'/row['clip_id']/'features.npz':raise ValueError('Original manifest path differs')
        present+=feature.is_file()
    print('VALIDATION_FEATURE_FILES_PRESENT',present,'OF',1200)
    if present!=1200:
        print('CACHE STILL MISSING. The metadata/code ZIPs cannot generate the 1,200 cached feature tensors. No Drive paths were created or manifests rewritten.')
    else:print('All feature paths exist. Return this output for exact-cache/preflight verification before a pilot run.')
    print('STOP: restoration only. No inference, training, extraction, installation or full-validation run executed.')
