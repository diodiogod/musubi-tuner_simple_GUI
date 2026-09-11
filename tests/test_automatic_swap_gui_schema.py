from modern_gui.settings import MINIMAX_H3_DEFAULTS, settings_schema
from modern_gui.stages import prepare_standard_stage


def test_automatic_memory_fields_are_runtime_and_default_to_fixed():
    schema = settings_schema({"training_mode": "MiniMax H3 (Experimental)", **MINIMAX_H3_DEFAULTS})
    fields = {field["key"]: (section, field) for section in schema["sections"] for field in section["fields"]}
    assert MINIMAX_H3_DEFAULTS["minimax_h3_block_memory_mode"] == "Fixed blocks (existing)"
    mode_section, mode_field = fields["minimax_h3_block_memory_mode"]
    assert mode_section["id"] == "runtime"
    for key in ("reserve_gb", "min_blocks", "max_blocks"):
        assert fields["minimax_h3_auto_swap_" + key][0]["id"] == "runtime"


def test_standard_stage_keeps_automatic_configuration():
    base = {**MINIMAX_H3_DEFAULTS, "output_name": "test",
            "minimax_h3_block_memory_mode": "Automatic (experimental)",
            "minimax_h3_auto_swap_reserve_gb": "3.5"}
    stage = prepare_standard_stage(base, {"dataset_config": "test.toml", "epochs": 1}, 0)
    assert stage["minimax_h3_block_memory_mode"] == "Automatic (experimental)"
    assert stage["minimax_h3_auto_swap_reserve_gb"] == "3.5"
