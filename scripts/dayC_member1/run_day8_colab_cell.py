"""Copy into Colab manually; never execute in the local clone."""
from pathlib import Path
from google.colab import files
import hashlib, json, os, re, stat, subprocess, sys, zipfile

archive = Path('/content/AFML_DayC_B2_C8A_Release.zip')
addon = Path('/content/AFML_DayC_B2_Full_Validation')
assert archive.is_file(), 'Upload the C8A Release ZIP into /content first'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == 'dbd44345b383645b12f6a03dae7778165ec0f82df6018dfec4cd27e7706b0882', 'Archive checksum mismatch'
assert not addon.exists(), 'Preserve any existing B2 attempt. Do not reuse or rerun this folder.'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    names = z.namelist()
    assert len(names) == len(set(names))
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
    assert {Path(n).parts[0] for n in names} == {addon.name}
    assert all(not stat.S_ISLNK(i.external_attr >> 16) for i in z.infolist())
    z.extractall('/content')
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
    PYTHONPATH=str(addon/'src') + os.pathsep + str(addon),
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg', GIT_OPTIONAL_LOCKS='0')
print('B2: exact 1200 cache identities, one full query-validation run, then 3 x 4-mode benchmarks.')
print('Keep the existing /content/afml_drive mount connected. No installs, mounts or test access.')
print('Inference progress is saved in <printed B2_RUN_DIR>/validation/query_validation_run.log.')
lines = []
with subprocess.Popen([sys.executable, '-B', str(addon/'scripts/dayC_member1/colab_day8.py')],
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
            print('HALTED; preserve evidence and do not rerun:', json.loads(z.read('failure.json')))
        else:
            completion = json.loads(z.read('validation/validation_completion.json'))
            assert completion['count'] == 1200 and completion['status'] == 'COMPLETE'
            print('Full query validation and 12 benchmark records produced; B2 evaluation remains pending.')
    print('RETURN_ZIP_SHA256:', hashlib.sha256(review.read_bytes()).hexdigest())
    files.download(str(review))
else:
    print('No return ZIP. Return the complete output above; preserve this folder and do not rerun.')
print('Return DAYC_B2_review.zip even on failure. Continue B2 after audit; no C8B review yet.')
