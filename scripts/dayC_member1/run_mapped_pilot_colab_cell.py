"""Copy into Colab manually; never execute in the local clone."""
from pathlib import Path
from google.colab import files
import hashlib, json, os, re, stat, subprocess, sys, zipfile

archive = Path('/content/AFML_DayC_C8_2_CPU_Pilot_Mapped_v3_Verified.zip')
addon = Path('/content/AFML_DayC_C8_2_CPU_Pilot_Mapped_v3')
assert archive.is_file(), 'Upload the Verified ZIP into /content first'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '8cb453863ffad1f688318729ca158497b04e2b15161f752faf1521f6a09bc33a', 'Archive hash mismatch'
assert not addon.exists(), 'Preserve existing evidence. Do not reuse or rerun this package folder.'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    names = z.namelist()
    assert len(names) == len(set(names))
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
    assert {Path(n).parts[0] for n in names} == {addon.name}
    assert all(not stat.S_ISLNK(i.external_attr >> 16) for i in z.infolist())
    z.extractall('/content')
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
    PYTHONPATH=str(addon/'src') + os.pathsep + str(addon),
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg', GIT_OPTIONAL_LOCKS='0')
print('Verifying all 1200 validation feature + metadata hashes; then one approved pilot.')
print('No install, mount, manifest rewrite, test access or full-validation inference.')
lines = []
with subprocess.Popen([sys.executable, '-B', str(addon/'scripts/dayC_member1/colab_query_pilot.py')],
        cwd=addon, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1) as process:
    for line in process.stdout:
        lines.append(line)
        print(line, end='', flush=True)
    returncode = process.wait()
print('Exit code:', returncode)
matches = re.findall(r'^RETURN REVIEW ZIP: (.+)$', ''.join(lines), flags=re.MULTILINE)
if matches:
    review = Path(matches[-1].strip()).resolve()
    assert review.is_file() and review.is_relative_to(addon.resolve())
    with zipfile.ZipFile(review) as z:
        hashes = json.loads(z.read('output_hashes.json'))
        assert set(z.namelist()) == set(hashes) | {'output_hashes.json'}
        for name, expected in hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
        if 'failure.json' in z.namelist():
            print('HALTED:', json.loads(z.read('failure.json')))
        else:
            proof = json.loads(z.read('cache_verification.json'))
            pilot = json.loads(z.read('pilot.json'))
            assert proof['status'] == 'PASS' and proof['verified_count'] == len(proof['records']) == 1200
            assert pilot['status'] == 'PILOT_COMPLETE_PENDING_C8A'
            print('1200 exact validation identities verified; single pilot complete, pending review.')
    print('REVIEW_ZIP_SHA256:', hashlib.sha256(review.read_bytes()).hexdigest())
    files.download(str(review))
else:
    print('No review ZIP. Return this complete output; preserve the directory and do not rerun.')
print('STOP: return the review ZIP even on failure. Await evidence audit and C8A before B2.')
