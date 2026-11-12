"""Read-only original-asset discovery. No writes, extraction, model imports or inference."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import zipfile


def stream_sha(stream):
    d=hashlib.sha256()
    for chunk in iter(lambda:stream.read(1048576),b''):
        d.update(chunk)
    return d.hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return stream_sha(stream)


def ids_sha(ids):
    return hashlib.sha256(('\n'.join(sorted(ids))+'\n').encode()).hexdigest()


def aggregate_hash(pairs):
    return hashlib.sha256(''.join(clip+'\t'+value+'\n' for clip,value in sorted(pairs)).encode()).hexdigest()


def safe_member(name):
    p=PurePosixPath(name)
    return not p.is_absolute() and '..' not in p.parts and not any(x.lower() in {'test','train'} for x in p.parts)


def validation_ids(data):
    rows=[json.loads(line) for line in data.splitlines() if line.strip()]
    if len(rows)!=1200 or any(row.get('split')!='val' for row in rows):
        raise ValueError('Approved ready metadata must be 1,200 validation rows')
    ids={row['clip_id'] for row in rows}
    if len(ids)!=1200:
        raise ValueError('Duplicate approved validation IDs')
    return ids


def recover_inventory(spec):
    """Search authorized roots by filename, then hash only relevant candidates.

    A bounded/incomplete search never establishes that an asset does not exist.
    No original path is rebased. Cache checks use all 1,200 approved IDs, or
    an exact match to their independently recorded ID-set digest.
    """
    result={'status':'READ_ONLY_DISCOVERY','exact_original_files':[], 'other_candidates':[],
        'archives':[], 'validation_caches':[], 'archive_cache_candidates':[], 'warnings':[],
        'search_incomplete':False,'asset_writes':False,'archives_extracted':False,
        'model_or_feature_tensors_loaded':False,'test_rows_or_test_features_opened':False}
    names={}
    for label,item in spec['assets'].items():
        for name in item['names']:names.setdefault(name,[]).append(label)
    seen_files=set();seen_dirs=set();cache_roots=set();cohort=None

    def observe(name,sha,location,size,kind,reader):
        nonlocal cohort
        exact=[label for label in names.get(name,[]) if spec['assets'][label]['sha256']==sha]
        row={'location':location,'sha256':sha,'bytes':size,'kind':kind}
        if exact:
            row['assets']=exact
            row['recorded_runtime_location']=any(location==spec['assets'][label].get('recorded_path') for label in exact)
            row['interpretation']='verified original bytes; other locations are backup copies, not substituted runtime paths'
            result['exact_original_files'].append(row)
            if 'ready_manifest' in exact:
                data=reader()
                if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Ready metadata changed during inspection')
                ids=validation_ids(data)
                if ids_sha(ids)!=spec['validation_id_set_sha256']:raise ValueError('Approved ID digest differs')
                cohort=ids
        elif len(result['other_candidates'])<250:
            row['status']='NOT_ACCEPTED_EXPECTED_HASH_MISMATCH';result['other_candidates'].append(row)
        return exact

    def inspect_archive(path):
        record={'path':str(path),'bytes':path.stat().st_size}
        if record['bytes']<=512*1024*1024:
            record['sha256']=file_sha(path)
            record['known_original_container_match']=record['sha256']==spec['original_archives'].get(path.name)
        else:record['container_hash']='not read: above 512 MiB; inspect member metadata only'
        result['archives'].append(record)
        with zipfile.ZipFile(path) as z:
            members=z.infolist()
            if len(members)>20000:
                result['warnings'].append('Archive member limit: '+str(path));result['search_incomplete']=True;return
            cache_members={}
            for info in members:
                name=info.filename;p=PurePosixPath(name)
                if info.is_dir() or not safe_member(name):continue
                if len(p.parts)>=3 and p.parts[-3]=='val' and p.name=='features.npz':
                    cache_members.setdefault(str(p.parent.parent),set()).add(p.parent.name)
                if p.name not in names or info.file_size>64*1024*1024:continue
                with z.open(info) as f:sha=stream_sha(f)
                exact=observe(p.name,sha,str(path)+'!'+name,info.file_size,'archive_member',lambda n=name:z.read(n))
                if any(label in exact for label in ('g5_package_manifest','g4_package_manifest')):
                    frozen=json.loads(z.read(info));prefix=name[:-len('PACKAGE_MANIFEST.json')]
                    expected={row['path']:row['sha256'] for row in frozen['files']}
                    actual={i.filename[len(prefix):] for i in members if i.filename.startswith(prefix) and not i.is_dir()}
                    good=actual==set(expected)|{'PACKAGE_MANIFEST.json'}
                    for rel,want in expected.items():
                        full=prefix+rel
                        if not safe_member(full) or full not in z.namelist():good=False;break
                        with z.open(full) as f:
                            if stream_sha(f)!=want:good=False;break
                    record.setdefault('frozen_package_checks',[]).append({'prefix':prefix,'all_recorded_members_match':good,'expected_members':len(expected)+1})
            for prefix,ids in cache_members.items():
                selected=cohort if cohort is not None and cohort.issubset(ids) else None
                if selected is None and len(ids)==1200 and ids_sha(ids)==spec['validation_id_set_sha256']:
                    selected=ids
                cache={'archive':str(path),'prefix':prefix,'feature_member_count':len(ids),
                    'approved_id_set_match':selected is not None,'feature_bytes_verified':False,
                    'usable_without_user_restoration':False,'status':'UNVERIFIED_ARCHIVE_CACHE'}
                if selected is not None:
                    try:
                        features=[];metas=[]
                        for i,clip in enumerate(sorted(selected),1):
                            for filename,target in (('features.npz',features),('meta.json',metas)):
                                member=prefix+'/'+clip+'/'+filename
                                info=z.getinfo(member)
                                if info.file_size>64*1024*1024:
                                    raise ValueError('Cache member exceeds 64 MiB inspection limit')
                                with z.open(member) as stream:target.append((clip,stream_sha(stream)))
                            if i%100==0:print('Read-only archived cache hashes:',str(path),i,'/ 1200',flush=True)
                        cache.update(all_feature_bytes_match=aggregate_hash(features)==spec['validation_feature_hash_aggregate_sha256'],
                            all_metadata_bytes_match=aggregate_hash(metas)==spec['validation_metadata_hash_aggregate_sha256'])
                        cache['feature_bytes_verified']=cache['all_feature_bytes_match'] and cache['all_metadata_bytes_match']
                        cache['status']='EXACT_1200_ARCHIVED_CACHE_BYTES' if cache['feature_bytes_verified'] else 'ARCHIVE_CACHE_BYTES_DIFFER'
                    except (KeyError,ValueError,OSError) as exc:cache.update(status='INCOMPLETE_OR_UNVERIFIABLE_ARCHIVE_CACHE',error=str(exc))
                result['archive_cache_candidates'].append(cache)

    def inspect_file(path):
        key=str(path)
        if key in seen_files or path.is_symlink():return
        seen_files.add(key)
        if path.name in names:
            if path.stat().st_size>64*1024*1024:
                result['warnings'].append('Candidate too large: '+key);return
            sha=file_sha(path);observe(path.name,sha,key,path.stat().st_size,'existing_file',path.read_bytes)
        elif path.suffix.lower()=='.zip' and (re.search(r'(afml|g4|g5|g6|dayb|dayc|dimodif)',path.name.lower())
                or any(re.search(r'(afml|g5|dayb)',part.lower()) for part in path.parent.parts)):
            inspect_archive(path)

    print('Checking recorded paths, then bounded project-backup filenames. No assets will be written.',flush=True)
    result['recorded_locations']=[]
    for label,item in spec['assets'].items():
        if 'recorded_path' not in item:continue
        path=Path(item['recorded_path']);row={'asset':label,'path':str(path),'exists':path.is_file()}
        result['recorded_locations'].append(row)
        if row['exists']:
            try:inspect_file(path)
            except (OSError,ValueError,KeyError,zipfile.BadZipFile) as exc:result['warnings'].append(str(path)+': '+str(exc))
    original_cache=Path(spec['cache_root'])
    if (original_cache/'val').is_dir():cache_roots.add(original_cache)
    result['search_roots']=[{'path':s,'exists':Path(s).is_dir()} for s in spec['search_roots']]
    def walk_error(error):
        result['warnings'].append(str(error));result['search_incomplete']=True
    visited=0
    for root_text in spec['search_roots']:
        root=Path(root_text)
        if not root.is_dir():continue
        for current,dirs,files in os.walk(root,followlinks=False,onerror=walk_error):
            current=Path(current)
            if str(current) in seen_dirs:dirs[:]=[];continue
            seen_dirs.add(str(current));visited+=len(dirs)+len(files)+1
            if visited>spec['max_entries']:
                dirs[:]=[];result['search_incomplete']=True;break
            depth=len(current.relative_to(root).parts)
            dirs[:]=[d for d in dirs if d.lower() not in {'test','train','.git','.cache','.trash','__pycache__','.ipynb_checkpoints'}
                     and not d.startswith(('pytest','tmp_')) and not (current==Path('/content') and d=='drive')]
            if 'val' in dirs:
                cache_roots.add(current);dirs.remove('val')  # Never recursively read media/tensors.
            if depth>=spec['max_depth']:
                if dirs:result['search_incomplete']=True
                dirs[:]=[]
            for name in files:
                try:inspect_file(current/name)
                except (OSError,ValueError,KeyError,zipfile.BadZipFile) as exc:result['warnings'].append(str(current/name)+': '+str(exc))
        if visited>spec['max_entries']:break
    result['metadata_entries_considered']=visited
    for base in sorted(cache_roots):
        val=base/'val';record={'root':str(base),'recorded_root':str(base)==spec['cache_root']}
        try:
            ids=cohort
            if ids is None:
                present_ids={p.name for p in val.iterdir() if p.is_dir() and re.fullmatch(r'[0-9a-f]{16}',p.name) and (p/'features.npz').is_file()}
                if len(present_ids)==1200 and ids_sha(present_ids)==spec['validation_id_set_sha256']:ids=present_ids
                else:
                    record.update(status='UNVERIFIED_NEEDS_ORIGINAL_READY_MANIFEST',candidate_feature_directories=len(present_ids));result['validation_caches'].append(record);continue
            feature_hashes=[];meta_hashes=[];missing=[]
            for i,clip in enumerate(sorted(ids),1):
                folder=val/clip;feature=folder/'features.npz';meta=folder/'meta.json'
                for p in (feature,meta):
                    if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(val.resolve()):missing.append(str(p));continue
                    target=feature_hashes if p==feature else meta_hashes
                    target.append((clip,file_sha(p)))
                if i%100==0:print('Read-only cache hashes:',str(base),i,'/ 1200',flush=True)
            record.update(feature_files_present=len(feature_hashes),metadata_files_present=len(meta_hashes),
                missing_path_count=len(missing),missing_path_examples=missing[:20],approved_id_set_match=True,
                all_feature_bytes_match=len(feature_hashes)==1200 and aggregate_hash(feature_hashes)==spec['validation_feature_hash_aggregate_sha256'],
                all_metadata_bytes_match=len(meta_hashes)==1200 and aggregate_hash(meta_hashes)==spec['validation_metadata_hash_aggregate_sha256'])
            record['status']='EXACT_1200_CACHE_BYTES' if record['all_feature_bytes_match'] and record['all_metadata_bytes_match'] else 'INCOMPLETE_OR_DIFFERENT_CACHE'
        except OSError as exc:record.update(status='UNREADABLE',error=str(exc))
        result['validation_caches'].append(record)
    labels={label for row in result['exact_original_files'] for label in row['assets']}
    result['assets_not_found_in_inspected_locations']=[label for label in spec['assets'] if label not in labels]
    result['caution']='Discovery only. Exact backup bytes do not authorize path rewriting, extraction, copying or inference. A bounded search cannot prove global absence.'
    print('READ_ONLY_RECOVERY_REPORT\n'+json.dumps(result,indent=2,allow_nan=False),flush=True)
    return result
