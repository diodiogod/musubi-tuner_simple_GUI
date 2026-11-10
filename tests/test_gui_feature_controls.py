"""Feature gates, locale decimals and focused help at GUI/backend boundaries."""
import copy
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from backends.regularization_switches import FEATURES, effective_settings, normalize_switches, toggle_settings
from backends.numeric_settings import decimal_text, normalize_numbers


@pytest.mark.parametrize("switch,spec", FEATURES.items())
def test_legacy_recipe_and_explicit_off_preserve_strength(switch, spec):
    strength, first, _ = spec
    old = {strength: "0.025"}
    assert normalize_switches(old)[switch] is True
    disabled = {**old, switch: False}
    assert effective_settings(disabled)[strength] == "0"
    assert disabled[strength] == "0.025"
    assert toggle_settings(disabled, switch, True)[strength] == "0.025"
    assert toggle_settings({strength: "0"}, switch, True)[strength] == first
    assert normalize_switches({}) == {}  # partial preset updates do not reset other switches


@pytest.mark.parametrize("family", ["krea", "moss", "h3"])
@pytest.mark.parametrize("on", [False, True, None])
def test_backend_switches_control_real_cli_strengths(family, on):
    from backends import krea2, moss_image, minimax_h3
    backend = {"krea": krea2, "moss": moss_image, "h3": minimax_h3}[family]
    prefix = "moss" if family == "moss" else "krea2"
    settings = {"output_dir": ".", "output_name": "switch-test", "_preview_only": True,
                "network_type": "LoRA", "learning_rate": "0.0001",
                f"{prefix}_weight_noise_sigma": "0,0125", f"{prefix}_depth_anchor_weight": "0,01"}
    if on is not None:
        settings.update({f"{prefix}_weight_noise_enabled": on, f"{prefix}_depth_anchor_enabled": on})
    if on is False:
        settings[f"{prefix}_depth_anchor_input_size"] = "inactive invalid value"
    original = copy.deepcopy(settings)
    command = backend.build_commands(settings)[0]
    assert command[command.index("--weight_noise_sigma") + 1] == ("0" if on is False else "0.0125")
    assert command[command.index("--depth_anchor_weight") + 1] == ("0" if on is False else "0.01")
    assert "inactive invalid value" not in command
    assert settings == original


def test_recipe_roundtrip_migrates_before_defaults_and_retains_off_strength(monkeypatch, tmp_path):
    from modern_gui import settings as module
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"training_mode": "MOSS-Image (Experimental)",
                               "moss_depth_anchor_weight": "0,01", "moss_weight_noise_sigma": "0,0125"}))
    monkeypatch.setattr(module, "LAST_SETTINGS", path)
    loaded = module.load_settings()
    assert loaded["moss_depth_anchor_enabled"] is True
    assert loaded["moss_weight_noise_enabled"] is True
    loaded["moss_depth_anchor_enabled"] = False
    module.save_settings(loaded)
    restored = module.load_settings()
    assert restored["moss_depth_anchor_enabled"] is False
    assert restored["moss_depth_anchor_weight"] == "0.01"
    schema = module.settings_schema(restored)
    fields = {f["key"]: f for section in schema["sections"] for f in section["fields"]}
    assert fields["moss_depth_anchor_enabled"]["type"] == "boolean"
    assert fields["moss_depth_anchor_weight"]["enabled_by"] == "moss_depth_anchor_enabled"
    assert fields["krea2_depth_vae_device"]["enabled_by"] == "krea2_depth_anchor_enabled"


@pytest.mark.parametrize("switch,spec", FEATURES.items())
def test_classic_switches_and_loading_a_legacy_strength_use_real_widgets(switch, spec):
    import tkinter as tk
    from tkinter import ttk
    from musubi_tuner_gui import MusubiTunerGUI
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("A Tk display is required for the Classic GUI widget check")
    root.withdraw()
    try:
        app = object.__new__(MusubiTunerGUI)
        app.root = root
        strength, initial, _ = spec
        control = ttk.Checkbutton(root)
        control.var = tk.BooleanVar(root, False)
        entry = ttk.Combobox(root) if switch.startswith("krea2") else ttk.Entry(root)
        if isinstance(entry, ttk.Combobox):
            entry.set("0.025")
        else:
            entry.insert(0, "0.025")
        app.entries = {switch: control, strength: entry}
        app._update_staged_summary = lambda: None
        app._sync_starting_point_from_settings = lambda _settings: None
        app._on_moss_timestep_schedule_changed = lambda: None
        app.update_button_states = app._update_regularization_switch_states
        app._update_regularization_switch_states()
        assert str(entry.cget("state")) == "disabled"
        # A legacy recipe turns its nonzero strength on, even over a disabled widget.
        app.set_values({strength: "0,01"})
        assert control.var.get() is True
        assert entry.get() == "0.01"
        assert str(entry.cget("state")) != "disabled"
        control.var.set(False)
        app._on_regularization_toggle(switch)
        app._update_regularization_switch_states()
        assert entry.get() == "0.01"
        assert str(entry.cget("state")) == "disabled"
    finally:
        root.destroy()


@pytest.mark.parametrize("raw,expected", [("0,0005", "0.0005"), ("0.0005", "0.0005"),
    ("5e-4", "5e-4"), ("1,25e-5", "1.25e-5"), (".25", ".25"), (",25", ".25"),
    ("0,", "0."), ("", ""), ("0,,5", "0,,5")])
def test_decimal_notation(raw, expected):
    assert decimal_text(raw) == expected


def test_numeric_normalization_does_not_rewrite_paths_prompts_or_optimizer_tuples():
    values = {"learning_rate": "0,0005", "moss_ema_decay": "0,98", "sample_every_n_epochs": "0,5",
              "output_dir": "J:/folder,1", "training_comment": "0,0005 is a value",
              "optimizer_args": "betas=(0.9,0.99)", "moss_sample_skip_layers": "22,24"}
    normalized = normalize_numbers(values)
    assert normalized["learning_rate"] == "0.0005"
    assert normalized["moss_ema_decay"] == "0.98"
    assert normalized["sample_every_n_epochs"] == "0.5"
    for key in ("output_dir", "training_comment", "optimizer_args", "moss_sample_skip_layers"):
        assert normalized[key] == values[key]


def test_original_automagic_comma_error_is_fixed_at_validation_and_launch():
    from backends.moss_image import build_commands
    from modern_gui.validation import validate_training_settings
    from modern_gui.settings import settings_schema
    settings = {"training_mode": "MOSS-Image (Experimental)", "optimizer_type": "Automagic3",
                "learning_rate": "0,0005", "moss_automagic3_min_lr": "0,00005", "moss_automagic3_max_lr": "0,001",
                "lr_scheduler": "constant", "lr_warmup_steps": "0", "network_type": "LoKr",
                "network_dim_low": "10000", "network_alpha_low": "10000", "lokr_factor": "4",
                "output_dir": ".", "output_name": "locale-test", "_preview_only": True}
    errors = validate_training_settings(settings)["errors"]
    assert not any(e["key"] in {"learning_rate", "optimizer_type", "moss_automagic3_min_lr", "moss_automagic3_max_lr"} for e in errors)
    command = build_commands(settings)[0]
    assert command[command.index("--learning_rate") + 1] == "0.0005"
    assert "min_lr=5e-05" in command
    assert "max_lr=0.001" in command
    schema = settings_schema({**settings, "learning_rate": "1e-4"})
    lr = next(f for s in schema["sections"] for f in s["fields"] if f["key"] == "learning_rate")
    assert lr["decimal"] is True  # scientific defaults must not silently select a text field
    errors = validate_training_settings({**settings, "learning_rate": "0,,0005"})["errors"]
    assert len([e for e in errors if e["key"] == "learning_rate"]) == 1
    assert not any(e["key"] == "optimizer_type" for e in errors)


def test_performance_help_is_specific_and_small_enough_to_read():
    from backends.performance_help import DETAILS
    from modern_gui.settings import _field_for
    displayed = []
    for key, detail in DETAILS.items():
        field = _field_for(key, False if key in {"compile", "gradient_checkpointing"} else "0")
        assert field["help_detail"] == detail
        assert len(detail.split()) <= 110
        displayed.append(detail)
    assert len(set(displayed)) == len(displayed)


def test_browser_decimal_helpers_accept_commas_dots_and_scientific_defaults():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed for browser helper verification")
    source = Path("modern_gui/static/app.js").read_text(encoding="utf-8")
    functions = "\n".join(line for line in source.splitlines() if line.startswith(("function looksNumeric(", "function canonicalDecimal(", "function performanceFieldKeys(")))
    functions += "\n" + re.search(r"function normalizeMossTimestepSchedule\(settings\)\{.*?\n\}", source, re.DOTALL).group()
    script = functions + '''
const state={schema:{sections:[{fields:[{key:"learning_rate",decimal:true},{key:"moss_depth_anchor_weight",decimal:true}]}],feature_switches:{moss_depth_anchor_enabled:{strength:"moss_depth_anchor_weight"}}}};
const imported=normalizeMossTimestepSchedule({learning_rate:"0,0005",moss_depth_anchor_weight:"0,01",output_dir:"J:/folder,1"});
if(imported.learning_rate!=="0.0005"||imported.moss_depth_anchor_weight!=="0.01"||imported.moss_depth_anchor_enabled!==true||imported.output_dir!=="J:/folder,1")throw Error("Imported decimal recipe changed meaning");
const explicitOff=normalizeMossTimestepSchedule({moss_depth_anchor_weight:"0,01",moss_depth_anchor_enabled:false});
if(explicitOff.moss_depth_anchor_enabled!==false)throw Error("Explicitly disabled feature reactivated");
for (const [input,expected] of [["0,0005","0.0005"],["0.0005","0.0005"],["5e-4","5e-4"],["1,2e-5","1.2e-5"]]) {
 if (!looksNumeric(input) || canonicalDecimal(input)!==expected) throw Error(input);
}
if (looksNumeric("0,,5")) throw Error("bad number accepted");
for(const mode of ["Krea 2","Wan 2.2","MOSS-Image (Experimental)"]){
 const keys=performanceFieldKeys(mode);
 for(const required of ["compile","gradient_checkpointing","mixed_precision","attention_mechanism","blocks_to_swap"])
  if(!keys.includes(required))throw Error(mode+" lost "+required);
 if(new Set(keys).size!==keys.length)throw Error("duplicate field");
 if(mode.startsWith("MOSS")&&!keys.includes("moss_fast_lora"))throw Error("MOSS controls missing");
}
'''
    result = subprocess.run([node, "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
