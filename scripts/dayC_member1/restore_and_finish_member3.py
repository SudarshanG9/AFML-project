"""Run manually in Colab: restore the one original-evidence ZIP and finish its validation-cache handoff."""
from pathlib import Path,PurePosixPath
import hashlib
import json
import runpy
import stat
import zipfile

ARCHIVE=Path('/content/AFML_Member3_Handoff.zip')
EXPECTED_SHA256='7fe10726c45dd458c2610dc9ccb627043b6b5efd75f375c42c3f68b94d59b47d'
PROJECT=Path('/content/AFML_DayC_Member3_Export')
CACHE_ROOT=Path('/content/afml_drive/MyDrive/AFML_Project/dimodif_emb')


def require(ok,message):
    if not ok:raise RuntimeError(message)


def restore(archive,project,expected=EXPECTED_SHA256):
    archive=Path(archive);project=Path(project)
    require(archive.is_file(),'Upload AFML_Member3_Handoff.zip to /content first.')
    require(hashlib.sha256(archive.read_bytes()).hexdigest()==expected,'Wrong or damaged handoff ZIP. Re-upload the original; do not rebuild it.')
    for p in (project,*project.parents):require(not p.is_symlink(),'Symlink extraction root')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();require(len(names)==len(set(names)),'Duplicate ZIP members')
        hashes=json.loads(z.read('HANDOFF_FILES_SHA256.json'))
        require(set(names)==set(hashes)|{'HANDOFF_FILES_SHA256.json'},'Unlisted ZIP files')
        for info in z.infolist():
            p=PurePosixPath(info.filename)
            require(not p.is_absolute() and '..' not in p.parts and str(p)==info.filename,'Unsafe ZIP path')
            require(not stat.S_ISLNK(info.external_attr>>16),'Symlink in ZIP')
            data=z.read(info.filename)
            if info.filename in hashes:require(hashlib.sha256(data).hexdigest()==hashes[info.filename],'Source hash mismatch: '+info.filename)
        project.mkdir(parents=True,exist_ok=True)
        for name in names:
            target=project/name
            for parent in (target,*target.parents):require(not parent.is_symlink(),'Symlink in restore destination')
            require(target.resolve().is_relative_to(project.resolve()),'Destination escapes project')
            data=z.read(name)
            if target.exists():
                require(target.is_file() and target.read_bytes()==data,'Existing changed file preserved; stop: '+str(target))
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as out:out.write(data)
    print('RESTORED_PROJECT='+str(project),flush=True)
    print('Original models, QKV, source, validation metadata and completed audit evidence restored. No test data accessed.',flush=True)
    return project


def main():
    restore(ARCHIVE,PROJECT)
    from google.colab import drive,files
    if not (CACHE_ROOT/'val').is_dir():
        print('Connect the existing Google Drive account containing AFML_Project. No cache will be created or regenerated.',flush=True)
        drive.mount('/content/afml_drive')
    require((CACHE_ROOT/'val').is_dir(),
        'Original validation cache is not available at '+str(CACHE_ROOT/'val')+
        '. Select the correct Drive account/mount. Metadata/model ZIPs cannot regenerate the original caches. No test directory was inspected.')
    require(not (PROJECT/'complete_member3_handoff.py').is_symlink(),'Unsafe completion helper')
    helper=runpy.run_path(str(PROJECT/'complete_member3_handoff.py'),run_name='member3_validation_copy')
    # The existing tested helper checks all 1200 original feature+metadata hashes.
    # It only reads cache_root/val/<approved_id>, never root/test or raw media.
    final=helper['complete'](ARCHIVE)
    print('SEND ONLY THIS FINAL ZIP TO MEMBER 3: '+str(final),flush=True)
    files.download(str(final))


if __name__=='__main__':main()
