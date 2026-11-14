import hashlib
import json
from pathlib import Path
import zipfile

import pytest
from scripts.dayB_build_colab_package import build


def fixture_tree(tmp_path):
    repo=tmp_path/'repo';private=tmp_path/'private'
    code=repo/'scripts/dayB_colab/g5/verify_package.py'
    data=private/'data/approved.json'
    code.parent.mkdir(parents=True);data.parent.mkdir(parents=True)
    code.write_text('# unit packaging fixture\n')
    data.write_text('{"unit":true}\n')
    rows=[{'path':'g5/verify_package.py','sha256':hashlib.sha256(code.read_bytes()).hexdigest()},
          {'path':'data/approved.json','sha256':hashlib.sha256(data.read_bytes()).hexdigest()}]
    manifest=repo/'configs/dayB/frozen_g5_package_manifest.json'
    manifest.parent.mkdir(parents=True);manifest.write_text(json.dumps({'files':rows}))
    return repo,private,data


def test_canonical_code_and_private_data_rebuild_exact_contents(tmp_path):
    repo,private,data=fixture_tree(tmp_path);out=tmp_path/'package.zip'
    result=build(repo,'g5',out,private)
    assert result['files']==3
    with zipfile.ZipFile(out) as z:
        assert z.read('AFML_G5_CPU_Colab_20261111T180139Z/data/approved.json')==data.read_bytes()


def test_tampered_private_input_does_not_create_partial_archive(tmp_path):
    repo,private,data=fixture_tree(tmp_path);data.write_text('tampered')
    out=tmp_path/'package.zip'
    with pytest.raises(RuntimeError,match='hash mismatch'):
        build(repo,'g5',out,private)
    assert not out.exists()


def test_private_code_cannot_substitute_missing_uploadable_source(tmp_path):
    repo,private,_=fixture_tree(tmp_path)
    code=repo/'scripts/dayB_colab/g5/verify_package.py'
    hidden=private/'g5/verify_package.py';hidden.parent.mkdir()
    hidden.write_bytes(code.read_bytes());code.unlink()
    with pytest.raises(RuntimeError,match='missing regular file'):
        build(repo,'g5',tmp_path/'package.zip',private)
