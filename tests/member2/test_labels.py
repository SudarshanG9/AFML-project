import pytest
from avfuse.member2_data.labels import BY_CATEGORY, label_of, manipulation_type


def test_flags_map_to_shared_types():
    assert manipulation_type(0, 0) == "real"
    assert manipulation_type(0, 1) == "audio_fake"
    assert manipulation_type(1, 0) == "video_fake"
    assert manipulation_type(True, True) == "both_fake"


def test_labels_follow_types():
    assert label_of("real") == 0 and all(label_of(t) == 1 for t in ("audio_fake", "video_fake", "both_fake"))
    with pytest.raises(ValueError):
        label_of("fake_audio")


def test_legacy_categories_cover_all_types():
    assert set(BY_CATEGORY.values()) == {"real", "audio_fake", "video_fake", "both_fake"}
