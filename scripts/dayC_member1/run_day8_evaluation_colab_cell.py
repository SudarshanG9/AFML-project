"""Manual evaluation-only Colab cell; source prepared locally."""
from pathlib import Path
from google.colab import files
import hashlib, json, os, re, stat, subprocess, sys, zipfile

archive = Path('/content/AFML_DayC_B2_Evaluation.zip')
addon = Path('/content/AFML_DayC_B2_Evaluation')
assert archive.is_file(), 'Upload AFML_DayC_B2_Evaluation.zip first'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '1f30245c2393d586cda4f60d051422357a93c5d724dc62e29a1280505c22671b', 'Evaluation ZIP checksum mismatch'
assert Path('/content/AFML_DayC_B2_Full_Validation').is_dir(), 'Keep the completed B2 runtime and saved outputs; do not rerun inference'
assert not addon.exists(), 'Preserve any existing evaluation attempt; do not reuse the folder automatically'
with zipfile.ZipFile(archive) as z:
    names = z.namelist()
    assert z.testzip() is None and len(names) == len(set(names))
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
    assert {Path(n).parts[0] for n in names} == {addon.name}
    assert all(not stat.S_ISLNK(i.external_attr >> 16) for i in z.infolist())
    z.extractall('/content')
manifest = json.loads((addon/'EVALUATION_MANIFEST.json').read_text())
for name, expected in manifest['files'].items():
    assert hashlib.sha256((addon/name).read_bytes()).hexdigest() == expected
print('B2 evaluation only: saved 1200 predictions, 1000 bootstrap replicates, seed 0.')
print('No model inference, feature-cache access, benchmark rerun, recalibration or test scoring.')
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg', GIT_OPTIONAL_LOCKS='0')
lines = []
with subprocess.Popen([sys.executable, '-B', str(addon/'finish_day8_evaluation.py')],
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
    assert review.is_file() and review.is_relative_to(Path('/content/AFML_DayC_B2_Full_Validation').resolve())
    with zipfile.ZipFile(review) as z:
        hashes = json.loads(z.read('output_hashes.json'))
        assert set(z.namelist()) == set(hashes) | {'output_hashes.json'}
        for name, expected in hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
        if 'failure.json' in z.namelist():
            print('HALTED:', json.loads(z.read('failure.json')))
        else:
            completion = json.loads(z.read('evaluation_completion.json'))
            assert completion['status'] == 'B2_EVIDENCE_COMPLETE_PENDING_C8B_REVIEW'
            print('B2 calculations complete. Return evidence for the full-block C8B review.')
    print('RETURN_ZIP_SHA256:', hashlib.sha256(review.read_bytes()).hexdigest())
    files.download(str(review))
else:
    print('No return ZIP. Return complete output; preserve every directory and do not rerun inference.')
print('STOP: no B3/Day9 policy freeze or held-out work before the full-block C8B acknowledgement.')
