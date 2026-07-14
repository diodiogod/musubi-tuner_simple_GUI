from continuation_names import continuation_name, dynamic_suffix_name, split_continuation_name


def test_continuation_names_advance_with_a_single_dynamic_suffix():
    assert continuation_name("portrait") == "portrait-cont"
    assert continuation_name("portrait-cont") == "portrait-cont2"
    assert continuation_name("portrait-cont2") == "portrait-cont3"


def test_legacy_repeated_suffixes_are_collapsed_on_next_continuation():
    assert split_continuation_name("portrait-cont-cont") == ("portrait", 2)
    assert continuation_name("portrait-cont-cont") == "portrait-cont3"
    assert continuation_name("portrait-cont2-cont") == "portrait-cont4"


def test_cont_inside_a_real_name_is_not_treated_as_a_suffix():
    assert continuation_name("content-model") == "content-model-cont"


def test_repeated_job_names_use_the_same_canonical_counter_rule():
    assert dynamic_suffix_name("portrait", "rerun") == "portrait-rerun"
    assert dynamic_suffix_name("portrait-rerun", "rerun") == "portrait-rerun2"
    assert dynamic_suffix_name("portrait-rerun-rerun", "rerun") == "portrait-rerun3"
