"""Rebuild frozen G4/G5 package contents from repository code + private inputs.

Packaging only: no inference, downloads, accounts, Git writes or protocol edits.
Original package manifests and code bytes are retained; ZIP timestamps may differ.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

NAMES = {'g4':'AFML_G4_Colab_20261111T153141Z', 'g5':'AFML_G5_CPU_Colab_20261111T180139Z'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_path(repo, gate, name, private_root):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe package path: '+name)
    special = {
        'requirements-g4.txt':repo/'requirements/dayB-g4.txt',
        'requirements-g5-cpu.txt':repo/'requirements/dayB-colab-cpu.txt',
        'COLAB_G4_INSTRUCTIONS.md':repo/'docs/dayB/COLAB_G4_PACKAGE_INSTRUCTIONS.md',
        'COLAB_G5_CPU_INSTRUCTIONS.md':repo/'docs/dayB/COLAB_G5_PACKAGE_INSTRUCTIONS.md',
    }
    if name.startswith(gate+'/') and name.endswith('.py'):
        return repo/'scripts/dayB_colab'/relative
    if name in special:
        return special[name]
    if name.startswith(('src/','scripts/','tests/')):
        # Required code must exist in the uploadable source tree, never only in
        # an ignored private snapshot. No fallback for Python implementation.
        return repo/relative
    candidates = ([private_root/relative] if private_root is not None else []) + [repo/relative]
    return next((p for p in candidates if p.is_file()),candidates[-1])


def build(repo, gate, output, private_root=None):
    repo, output = Path(repo), Path(output)
    gate = gate.lower()
    manifest_path = repo/'configs/dayB'/('frozen_'+gate+'_package_manifest.json')
    manifest = json.loads(manifest_path.read_text())
    selected, problems, seen = [], [], set()
    for row in manifest['files']:
        name = row['path']
        if name in seen:
            raise ValueError('Duplicate manifest path: '+name)
        seen.add(name)
        source = source_path(repo,gate,name,private_root)
        if not source.is_file() or source.is_symlink():
            problems.append(name+' — missing regular file')
        elif digest(source)!=row['sha256']:
            problems.append(name+' — hash mismatch; preserve the approved snapshot')
        else:
            selected.append((source,name))
    if problems:
        raise RuntimeError('Cannot rebuild the approved package; no ZIP created:\n'+'\n'.join(problems))
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for source,name in selected:
            z.write(source,NAMES[gate]+'/'+name)
        z.write(manifest_path,NAMES[gate]+'/PACKAGE_MANIFEST.json')
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:
            raise RuntimeError('Archive integrity check failed')
    return {'archive':str(output.resolve()),'bytes':output.stat().st_size,'sha256':digest(output),
            'frozen_package_manifest_sha256':digest(manifest_path),'files':len(selected)+1,
            'note':'Package contents match the frozen manifest; ZIP container bytes/timestamps can differ.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gate',choices=['g4','g5'],required=True)
    p.add_argument('--private-root',type=Path,
                   help='Locally restored original package payload for private manifests/evidence/model source assets')
    p.add_argument('--out',type=Path,required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    print(json.dumps(build(repo,a.gate,a.out,a.private_root),indent=2))


if __name__=='__main__':
    main()
