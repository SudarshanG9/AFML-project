"""Copy original approved validation caches into ONE handoff ZIP. No inference/test access."""
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import re
import zipfile


def require(ok,message):
    if not ok:raise ValueError(message)


def safe(name):
    p=PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and str(p)==name,'Unsafe ZIP path')
    return name


def validation_ids(data,expected_count):
    rows=[json.loads(line) for line in data.splitlines() if line.strip()]
    require(len(rows)==expected_count,'Wrong approved validation count')
    require(all(r['split']=='val' and isinstance(r['clip_id'],str) and re.fullmatch('[A-Za-z0-9_-]+',r['clip_id']) for r in rows),'Invalid/non-validation cohort')
    ids={r['clip_id'] for r in rows};require(len(ids)==expected_count,'Duplicate validation ID')
    return sorted(ids)


def copy_checked(z,path,name,expected,validation_root=None):
    require(path.is_file(),'Missing original validation file: '+str(path))
    root=Path(validation_root) if validation_root is not None else path.parent.parent
    physical=root.resolve(strict=True)
    require('test' not in physical.parts,'Validation root resolves into a test directory')
    require(path.resolve(strict=True).is_relative_to(physical),'Cache escapes approved validation root')
    # Drive mount/project ancestors may be symlinks. Below the explicitly
    # approved validation root, clip and file symlinks remain forbidden.
    current=path
    while current!=root:
        require(not current.is_symlink(),'Symlink inside validation cache: '+str(current))
        require(current!=current.parent,'Cache is outside validation root')
        current=current.parent
    digest=hashlib.sha256()
    with path.open('rb') as src,z.open(name,'w',force_zip64=True) as dst:
        for block in iter(lambda:src.read(1024*1024),b''):digest.update(block);dst.write(block)
    require(digest.hexdigest()==expected,'Original validation cache hash mismatch: '+str(path))
    return digest.hexdigest()


def assemble(base,cache_root,out,expected_count=1200):
    """One read per original cache file; preserve failure .partial; never overwrite originals."""
    base=Path(base);out=Path(out);cache_root=Path(cache_root)
    require(not out.exists() and not out.is_symlink(),'Output exists; preserve it')
    partial=out.with_suffix('.partial');require(not partial.exists(),'Partial archive exists; preserve failure evidence')
    with zipfile.ZipFile(base) as source:
        names=source.namelist();require(len(names)==len(set(names)),'Duplicate source ZIP members')
        for name in names:safe(name)
        checks=json.loads(source.read('HANDOFF_FILES_SHA256.json'))
        require(set(names)==set(checks)|{'HANDOFF_FILES_SHA256.json'},'Unlisted source ZIP entries')
        for name,digest in checks.items():require(hashlib.sha256(source.read(name)).hexdigest()==digest,'Handoff source changed: '+name)
        ids=validation_ids(source.read('data/manifests/manifest_whole_val.jsonl'),expected_count)
        inventory=json.loads(source.read('validation/accepted_B2_cache_verification.json'))
        require(inventory['status']=='PASS' and inventory['test_access'] is False and inventory['verified_count']==expected_count,'Unaccepted validation inventory')
        records=inventory['records'];require(len(records)==expected_count and len({r['clip_id'] for r in records})==expected_count,'Invalid cache identity inventory')
        approved={r['clip_id']:r for r in records};require(set(approved)==set(ids),'Validation IDs differ from accepted inventory')
        for r in records:
            for key in ['feature_sha256','metadata_sha256']:require(re.fullmatch('[0-9a-f]{64}',r[key]) is not None,'Missing original validation hash')
        out.parent.mkdir(parents=True,exist_ok=True);hashes={};cache_files=[]
        with zipfile.ZipFile(partial,'x',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as target:
            for name in names:
                if name in {'HANDOFF_FILES_SHA256.json','SHA256SUMS.txt','VALIDATION_CACHE_COPY_STATUS.json'}:continue
                data=source.read(name);target.writestr(name,data);hashes[name]=hashlib.sha256(data).hexdigest()
            for index,clip in enumerate(ids,1):
                folder=cache_root/'val'/clip
                for filename,key in [('features.npz','feature_sha256'),('meta.json','metadata_sha256')]:
                    name='dimodif_emb/val/'+clip+'/'+filename
                    require(name not in hashes,'Cache already packaged')
                    hashes[name]=copy_checked(target,folder/filename,name,approved[clip][key],cache_root/'val');cache_files.append(name)
                if index%100==0:print('ORIGINAL_VALIDATION_CACHES_COPIED='+str(index)+'/'+str(expected_count),flush=True)
            status={'status':'COMPLETE','validation_clips':expected_count,'original_feature_and_metadata_files':len(cache_files),
                'all_file_hashes_match_accepted_B2_inventory':True,'feature_regeneration':False,'inference':False,
                'test_directory_access':False,'source_manifest_modified':False,'original_capsule':'opaque sealed reference only; do not extract/parse its canonical test rows before authorization'}
            data=(json.dumps(status,sort_keys=True,indent=2)+'\n').encode();target.writestr('VALIDATION_CACHE_COPY_STATUS.json',data);hashes['VALIDATION_CACHE_COPY_STATUS.json']=hashlib.sha256(data).hexdigest()
            sums=''.join(digest+'  '+name+'\n' for name,digest in sorted(hashes.items())).encode();target.writestr('SHA256SUMS.txt',sums);hashes['SHA256SUMS.txt']=hashlib.sha256(sums).hexdigest()
            target.writestr('HANDOFF_FILES_SHA256.json',(json.dumps(hashes,sort_keys=True,indent=2)+'\n').encode())
    with zipfile.ZipFile(partial) as finished:
        require(set(finished.namelist())==set(hashes)|{'HANDOFF_FILES_SHA256.json'},'Incomplete final ZIP')
        for name,digest in hashes.items():
            actual=hashlib.sha256()
            with finished.open(name) as f:
                for block in iter(lambda:f.read(1024*1024),b''):actual.update(block)
            require(actual.hexdigest()==digest,'Final archive verification failed: '+name)
    # Exclusive publication on the same Colab filesystem; never overwrite an older ZIP.
    os.link(partial,out);partial.unlink()
    print('FINAL_MEMBER3_ZIP='+str(out),flush=True)
    return out


def complete(base):
    return assemble(base,Path('/content/afml_drive/MyDrive/AFML_Project/dimodif_emb'),
        Path('/content/AFML_DayC_Member3_Export/artifacts/dayC/member1/handoffs/AFML_Member3_COMPLETE.zip'))
