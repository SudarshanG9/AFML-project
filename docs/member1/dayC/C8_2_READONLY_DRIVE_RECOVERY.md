# B1 / C8.2 — verified originals and one read-only Drive recovery cell

**Status: EXTERNAL HANDOFF REQUIRED.** This is original-file discovery, not regeneration. No checkpoint, manifest, context, cache, model source or historical provenance is created, rewritten, substituted, extracted or copied by the cell. No inference is run. All agent-generated code/reports stay inside the local clone; you run the cell manually in Colab.

## What the supplied G4 backup establishes

`/Users/sudarshan/Downloads/G4_20261111T174507Z_816ab424` contains seven preserved evidence files. All six recorded hashes match the accepted G4 outputs; `output_hashes.json` itself is byte-identical to the preserved original. Its preflight records PASS, 1,200 expected and verified validation clips, zero errors, and 1,200 cache inventory rows. The supplied folder does **not** contain the feature tensors, checkpoint or G5 execution context. `execution_environment.json` is G4 environment evidence, a different file and role; it must not be renamed or passed off as G5 `execution_context.json`.

The user-supplied Colab inventory is recorded verbatim in structured form under this recovery preparation's `external_inventory_user_supplied.json`: original G5 context, manifests, G4 report, checkpoint/config/source and recorded cache path absent; packaged accepted context present with the accepted SHA. `/content/drive` existing alone does not establish that the original Drive cache or backup is mounted/reachable.

## Recoverable original files already verified locally

| Asset | Exact existing local path | Accepted SHA-256 |
|---|---|---|
| g5_context | `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayB/final_review/DayB_FINAL_CPU_20261112T031627Z_91e5f723/received_evidence/provenance/g5/execution_context.json` | `c03488cb787b1230bd86274dc7a36d39983574c7e2bc996c931dcfc8a835bd7c` |
| canonical_manifest | `/Users/sudarshan/Documents/GitHub/AFML-project/data/manifests/manifest_whole.jsonl` | `8339b2476a71e62c6c6fca7d937ba4cc7a704b09d26f08506c20d1f10de8f3dd` |
| validation_manifest | `/Users/sudarshan/Documents/GitHub/AFML-project/data/manifests/manifest_whole_val.jsonl` | `22d1d9e02403cfdbb05f2116da16bc4015f540547a4ac00bf6b7127c119e359c` |
| ready_manifest | `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayB/g4_colab_review/G4_20261111T174507Z_816ab424/dayB_val_ready.jsonl` | `c79ad908517878ebf0f3aa0e0dac053a01d9d5c4eb264583db26096985890481` |
| g4_report | `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayB/g4_colab_review/G4_20261111T174507Z_816ab424/preflight_report.json` | `559445575ff51495dbb79ae175737ad1596fcc41afab058734d138bd01fb9c01` |
| checkpoint | `/Users/sudarshan/Documents/GitHub/AFML-project/third_party/dimodif/ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.pth` | `e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e` |
| model_config | `/Users/sudarshan/Documents/GitHub/AFML-project/third_party/dimodif/ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.json` | `bea128dc7dad6b72bc0487dcc45d03ae3322e48fd597de1a5cd83ad036f30651` |
| model_source | `/Users/sudarshan/Documents/GitHub/AFML-project/third_party/dimodif/src/models.py` | `e74b5e3b5fcee24353f9940dddd48d9ac1982ced725664db2518f63a6d853778` |
| reference_predictions | `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/predictions/reference_val.jsonl` | `e146432023638635b53dc517f3b4b06ecd87a6599efe3709ed14b145c7458ff3` |

The G5 context is **an accepted exact-copy backup input by content**, independently matching Day B output hashes and the C0-approved specification. Its three preserved copies and the packaged reference copy all match `c03488cb787b1230bd86274dc7a36d39983574c7e2bc996c931dcfc8a835bd7c`. This does not prove the original G5 run exists in the current Colab session and does not silently authorize replacing its recorded path with a different role/location.

Verified local archived originals:

- `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayB/local_deliverables/g5_colab/AFML_G5_CPU_Colab_20261111T180139Z.zip` — SHA-256 `3b44fc5a1db22733716a5ee4d2577227747e60fe97cf03a29fdc92a291fa2469`; 129 members, all matching the frozen approved manifest.
- `/Users/sudarshan/Documents/GitHub/AFML-project/artifacts/dayB/local_deliverables/g4_colab/AFML_G4_Colab_20261111T153141Z.zip` — SHA-256 `70001af7f259aa8fd02a161246f97b1850f4cbd1e6e32a1b3f6fd7ec305b0fca`; 113 members, all matching the frozen approved manifest.

The G5 archive contains the canonical/validation/ready manifests, G4 evidence, original model config/source and original project implementation. The G4 archive contains canonical/validation manifests and config/source. **Neither archive contains the checkpoint or validation feature cache.** The checkpoint is a separate exact local file above. The preserved G6 add-on archive includes the exact G5 context under `accepted_g5_execution_context.json`; this is a copy of original bytes, not a reconstructed context.

The full accepted validation cache is still not present in the clone. The previously inspected local pilot contains 10 matching feature/metadata files; that is not a substitute for 1,200. The G4 report's hashes cannot regenerate missing tensors. Do not run training, feature extraction or checkpoint replacement to make this gate pass.

## ONE read-only Colab cell

Paste the entire cell below in the current Colab session. It needs no upload, installation or inference package execution. It uses only Python's standard library, reads existing files and ZIP members in place, and prints one `READ_ONLY_RECOVERY_REPORT`.

It checks recorded locations first, then bounded filename searches under `/content` and the existing Drive MyDrive/Shareddrives/shortcut roots. It never descends test/train directories or reads held-out cache members. Canonical manifest contents are only hashed, never parsed for test rows. Only relevant asset filenames and project-named ZIPs are opened; ordinary unrelated files are not read.

When a complete candidate validation cache is found, it verifies all **1,200** feature and metadata byte hashes against independently derived G4 aggregate fingerprints, without creating a manifest or loading arrays. A matching ID-set fingerprint can identify the complete cohort even when the original ready manifest has not yet been located. If a larger cache contains other IDs and the ready manifest is unavailable, it reports that coverage is unverified rather than selecting a guessed subset. Archive caches can also be hashed without extraction. Hashing a complete cache can take time; progress prints every 100 clips.

Search limits (depth 10, 40,000 metadata entries, 20,000 members per archive) are disclosed; `search_incomplete`, warnings and unverified statuses must be retained. Large archive containers need not be rehashed wholesale, but relevant original members can still be checked. Cache members above the 64 MiB inspection limit are reported unverified. A bounded search cannot prove global absence. These limits do not reduce any research experiment.

```python
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


# Independently verified expected hashes from preserved Day B evidence.
SPEC = {'assets': {'canonical_manifest': {'names': ['manifest_whole.jsonl'],
                                   'recorded_path': '/content/AFML_G5_CPU_Colab_20261111T180139Z/data/manifests/manifest_whole.jsonl',
                                   'sha256': '8339b2476a71e62c6c6fca7d937ba4cc7a704b09d26f08506c20d1f10de8f3dd'},
            'checkpoint': {'names': ['dfd_lavdf_reduceonplateau_64_4_1_5_True.pth'],
                           'recorded_path': '/content/g5_results/G5_CPU_20261111T181935Z_422cf72f/runtime/dimodif/ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.pth',
                           'sha256': 'e1e59006d1bb6ad36a3859af5486a6649ed41bba23258b7c41e24f873be7948e'},
            'g4_package_manifest': {'names': ['PACKAGE_MANIFEST.json'],
                                    'sha256': '29b00e9731dfa12265b607502130712e91525afc3b64bd2f309137e5f591b43a'},
            'g4_report': {'names': ['preflight_report.json'],
                          'recorded_path': '/content/AFML_G5_CPU_Colab_20261111T180139Z/data/g4_evidence/preflight_report.json',
                          'sha256': '559445575ff51495dbb79ae175737ad1596fcc41afab058734d138bd01fb9c01'},
            'g5_context': {'names': ['execution_context.json',
                                     'accepted_g5_execution_context.json',
                                     'accepted_g5_context.json'],
                           'recorded_path': '/content/g5_results/G5_CPU_20261111T181935Z_422cf72f/execution_context.json',
                           'sha256': 'c03488cb787b1230bd86274dc7a36d39983574c7e2bc996c931dcfc8a835bd7c'},
            'g5_package_manifest': {'names': ['PACKAGE_MANIFEST.json'],
                                    'sha256': '6d389dc16e7dc46f658cc81c776ceece5913913ec1bfcf9c43b0bcf0989e9863'},
            'model_config': {'names': ['dfd_lavdf_reduceonplateau_64_4_1_5_True.json'],
                             'recorded_path': '/content/g5_results/G5_CPU_20261111T181935Z_422cf72f/runtime/dimodif/ckpt/dfd/dfd_lavdf_reduceonplateau_64_4_1_5_True.json',
                             'sha256': 'bea128dc7dad6b72bc0487dcc45d03ae3322e48fd597de1a5cd83ad036f30651'},
            'model_source': {'names': ['models.py'],
                             'recorded_path': '/content/g5_results/G5_CPU_20261111T181935Z_422cf72f/runtime/dimodif/src/models.py',
                             'sha256': 'e74b5e3b5fcee24353f9940dddd48d9ac1982ced725664db2518f63a6d853778'},
            'ready_manifest': {'names': ['dayB_val_ready.jsonl'],
                               'recorded_path': '/content/AFML_G5_CPU_Colab_20261111T180139Z/data/g4_evidence/dayB_val_ready.jsonl',
                               'sha256': 'c79ad908517878ebf0f3aa0e0dac053a01d9d5c4eb264583db26096985890481'},
            'reference_predictions': {'names': ['dayB_reference_val.jsonl', 'reference_val.jsonl'],
                                      'recorded_path': '/content/g5_results/G5_CPU_20261111T181935Z_422cf72f/predictions/dayB_reference_val.jsonl',
                                      'sha256': 'e146432023638635b53dc517f3b4b06ecd87a6599efe3709ed14b145c7458ff3'},
            'validation_manifest': {'names': ['manifest_whole_val.jsonl'],
                                    'recorded_path': '/content/AFML_G5_CPU_Colab_20261111T180139Z/data/manifests/manifest_whole_val.jsonl',
                                    'sha256': '22d1d9e02403cfdbb05f2116da16bc4015f540547a4ac00bf6b7127c119e359c'}},
 'cache_root': '/content/drive/.shortcut-targets-by-id/1yKf-S0Yyi8ZU7uLx6JbYKZpL95Kn3Mb7/dimodif_emb',
 'canonical_manifest_rows_must_not_be_parsed': True,
 'expected_count': 1200,
 'max_depth': 10,
 'max_entries': 40000,
 'no_asset_writes': True,
 'original_archives': {'AFML_G4_Colab_20261111T153141Z.zip': '70001af7f259aa8fd02a161246f97b1850f4cbd1e6e32a1b3f6fd7ec305b0fca',
                       'AFML_G5_CPU_Colab_20261111T180139Z.zip': '3b44fc5a1db22733716a5ee4d2577227747e60fe97cf03a29fdc92a291fa2469',
                       'AFML_G6_CPU_Addon.zip': '48408b4041d141542f9f9b2789fbb10aefbfd23920f615c11295c4d24388dada'},
 'schema_id': 'avfuse-member1-readonly-recovery/1.0',
 'search_roots': ['/content',
                  '/content/drive/MyDrive',
                  '/content/drive/Shareddrives',
                  '/content/drive/.shortcut-targets-by-id'],
 'validation_feature_hash_aggregate_sha256': '5c42a20ba42304ccddc0f0c4600cb3f3ae6ab0789047a44a5fe3171cbdfe3ffe',
 'validation_id_set_sha256': 'a64c2347aef98758db9776cd31fd514730c7233a129eb3b3bc18406c4ca30650',
 'validation_metadata_hash_aggregate_sha256': '231e91f79002b751f74627706cffbb7b1f91b7d245abbe425740db376b1cf131'}

REPORT = recover_inventory(SPEC)
```

## Return / next action

Return the printed `READ_ONLY_RECOVERY_REPORT` and any warnings. **Do not rerun the pilot or restore anything yet.** The report distinguishes exact existing assets, exact original archive members, wrong hashes, cache byte verification and unverified locations. Once the actual originals are located, any user-managed restoration can use verified original bytes; it need not regenerate data. This cell itself performs no such restoration and never changes `feature_path` values or expected hashes.

C8A remains unapproved. After exact original inputs are available at authorized locations, finish C8.2 preflight/pilot and review real evidence before requesting C8A. B2 and Day 9/10 remain blocked.

## Local independent checks

**215 Member 1 tests passed, 0 failed, 0 skipped.** Recovery tests prove no input writes or archive extraction, original-copy versus runtime-location distinction, opaque canonical hashing, avoidance of test/train paths, incomplete-search reporting, full 1,200-file hashing and all 1,200 archived features/metadata verified without extraction. The actual supplied G4 backup was also read with this scanner: ready manifest/report recognized by exact SHA, and all seven backup files unchanged afterward. No live Drive operation or real-model inference was performed by the agent.
