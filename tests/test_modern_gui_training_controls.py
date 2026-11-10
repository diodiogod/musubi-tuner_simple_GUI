import json
import shutil
import subprocess
from pathlib import Path

import pytest

from modern_gui import settings


def test_gradient_controls_have_numeric_constraints_and_distinct_help():
    schema = settings.settings_schema({"gradient_accumulation_steps": "4", "max_grad_norm": "0.5"})
    fields = {field["key"]: field for section in schema["sections"] for field in section["fields"]}
    accumulation, clipping = (fields[key] for key in ("gradient_accumulation_steps", "max_grad_norm"))
    assert accumulation["type"] == "number" and accumulation["step"] == 1 and accumulation["min"] == 1
    assert clipping["type"] == "number" and clipping["step"] == .1 and clipping["min"] == 0
    assert "fewer optimizer updates per epoch" in accumulation["help_detail"]
    assert "0 disables clipping" in clipping["help"]
    assert accumulation["help_detail"] != clipping["help_detail"]


def test_loading_old_recipe_supplies_missing_controls_without_resetting_choices(monkeypatch, tmp_path):
    recipe = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "LAST_SETTINGS", recipe)
    recipe.write_text(json.dumps({"training_mode": "MOSS-Image (Experimental)"}))
    loaded = settings.load_settings()
    assert loaded["gradient_accumulation_steps"] == "1" and loaded["max_grad_norm"] == "1.0"
    recipe.write_text(json.dumps({"gradient_accumulation_steps": "4", "max_grad_norm": "0,5"}))
    loaded = settings.load_settings()
    assert loaded["gradient_accumulation_steps"] == "4" and loaded["max_grad_norm"] == "0.5"


def test_batch_summary_uses_active_sources_overrides_and_mixed_update_range():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for frontend summary checks")
    path = Path("modern_gui/static/training-batch.js").resolve()
    script = "const {summary}=require(" + json.dumps(str(path)) + ");\n" + """
const assert=require('node:assert/strict');
assert.equal(summary(null,4),null);
assert.match(summary({datasets:[]},1).text,/No active/);
const defaults={general:{batch_size:2},datasets:[{},{}]};
assert.match(summary(defaults,4).text,/Batch size: 2.*Effective batch: 8/);
const mixed={general:{batch_size:1},datasets:[{batch_size:1},{batch_size:2}],disabled_datasets:[{batch_size:99}]};
assert.match(summary(mixed,4).text,/Batch sizes: 1, 2.*Effective batch: 4–8/);
assert.match(summary(mixed,4).detail,/mixed updates/);
assert.match(summary({datasets:[{batch_size:0}]},1).text,/Review/);
assert.match(summary(defaults,1.5).detail,/whole number/);
assert.match(summary({datasets:[{batch_size:1}]},1).detail,/1 batch per/);
assert.match(summary({datasets:[{batch_size:1}]},4).detail,/4 batches per/);
assert.match(summary(defaults,4).detail,/Partial buckets/);
"""
    result = subprocess.run([node, "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
