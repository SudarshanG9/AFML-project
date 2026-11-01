import pytest
from avfuse.member2_data.manifest import verify_manifest
from avfuse.member2_data.evaluate import metrics_for_fake_type


def row(clip, group, split, manipulation_type):
    return {"clip_id":clip,"dataset":"toy","source_id":group,"identity_id":group,
            "split":split,"label":int(manipulation_type != "real"),
            "manipulation_type":manipulation_type,"feature_path":f"{clip}.pt"}


def test_no_source_overlap():
    verify_manifest([row('a','A','val','real'),row('b','B','test','audio_fake')])
    with pytest.raises(ValueError,match='source-group leakage'):
        verify_manifest([row('a','A','val','real'),row('b','A','test','audio_fake')])


def test_label_consistency_rejected():
    bad=row('a','A','val','real');bad['label']=1
    with pytest.raises(Exception):
        verify_manifest([bad])


def test_fake_type_uses_real_negative_class():
    manifest=[row('a','A','val','real'),row('b','B','val','audio_fake'),
              row('c','C','val','video_fake')]
    preds=[{"clip_id":"a","score_fake":0.1},{"clip_id":"b","score_fake":0.9},
           {"clip_id":"c","score_fake":0.8}]
    m=metrics_for_fake_type(preds,manifest,'audio_fake',0.5)
    assert m['auroc']==1.0 and m['n_real']==1 and m['n_fake']==1
