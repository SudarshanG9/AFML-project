"""Manual joint B2/B3 audit cell; run only in approved Colab CPU environment."""
from pathlib import Path
from google.colab import files
import hashlib, json, os, re, stat, subprocess, sys, zipfile

archive = Path('/content/AFML_DayC_B2_B3_Combined_Audit.zip')
addon = Path('/content/AFML_DayC_B2_B3_Combined_Audit')
assert archive.is_file(), 'Upload the combined audit ZIP into /content first'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '9d4fe84cb29d1d73cd07f410147a2593a2d0b2897b3eaaa63d47ff2a4c1abd25', 'Combined ZIP checksum mismatch'
assert not addon.exists(), 'Preserve any existing joint audit attempt; do not overwrite its directory'
with zipfile.ZipFile(archive) as z:
    names = z.namelist()
    assert z.testzip() is None and len(names) == len(set(names))
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
    assert {Path(n).parts[0] for n in names} == {addon.name}
    assert all(not stat.S_ISLNK(i.external_attr >> 16) for i in z.infolist())
    z.extractall('/content')
manifest = json.loads((addon/'JOINT_MANIFEST.json').read_text())
for name, expected in manifest['files'].items():
    assert not Path(name).is_absolute() and '..' not in Path(name).parts
    assert hashlib.sha256((addon/name).read_bytes()).hexdigest() == expected
project = addon/'execution_project'
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
    PYTHONPATH=str(project/'src') + os.pathsep + str(project),
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg', GIT_OPTIONAL_LOCKS='0')
print('JOINT B2/B3 AUDIT: saved-score statistics, 1000 source-bootstrap replicates, seed 0; then unapproved B3 draft.')
print('No Drive mount, checkpoint, cached tensors, model inference, benchmark rerun, final freeze or test scoring.')
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
    assert review.is_file() and review.is_relative_to(addon.resolve())
    with zipfile.ZipFile(review) as z:
        hashes = json.loads(z.read('output_hashes.json'))
        assert len(z.namelist()) == len(set(z.namelist()))
        assert set(z.namelist()) == set(hashes) | {'output_hashes.json'}
        for name, expected in hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
        if 'failure.json' in z.namelist():
            print('HALTED:', json.loads(z.read('failure.json')))
        else:
            complete = json.loads(z.read('evaluation_completion.json'))
            proposal = json.loads(z.read('day9_preparation/policy_proposal.json'))
            assert complete['n'] == 1200 and complete['n_boot'] == 1000 and complete['seed'] == 0
            assert proposal['status'] == 'PROPOSED_NOT_APPROVED'
            assert proposal['ready_for_freeze'] is False and proposal['selected_interpretation'] is None
            print('B2 statistics and B3 draft are ready for joint review. Missing real owner releases remain explicit.')
    print('RETURN_ZIP_SHA256:', hashlib.sha256(review.read_bytes()).hexdigest())
    files.download(str(review))
else:
    print('No return ZIP. Return this complete output; preserve the directory. Do not rerun inference or benchmarks.')
print('STOP: return DAYC_B2_B3_review.zip even on failure. No final freeze or held-out test authorization is inferred.')
