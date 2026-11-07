from modern_gui.monitor import is_training_progress_line, parse_cache_progress, parse_training_line
from modern_gui.jobs import JobSupervisor


def test_repa_frontend_renders_structured_and_already_running_server_metrics():
    import shutil
    import subprocess

    import pytest

    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed to execute the frontend monitor helpers")
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync('modern_gui/static/app.js','utf8');
function definition(name){const start=source.indexOf('function '+name+'('),end=source.indexOf('\nfunction ',start+1);assert(start>=0&&end>start);return source.slice(start,end)}
const nodes=Object.fromEntries(['#metric-repa','#metric-repa-group','#repa-metrics-help','#metric-depth','#metric-depth-group','#metric-dop','#metric-dop-group'].map(key=>[key,{dataset:{}}]));
let help=null;
const context={$:key=>nodes[key],findField:()=>({metrics_help:'Explain the teacher agreement.'}),openHelp:value=>{help=value}};vm.createContext(context);
vm.runInContext(['parseProgressLine','renderRepaMetrics','renderOptionalMetrics','openRepaMetricsHelp'].map(definition).join('\n'),context);
const logged='steps: 1%|#| 1/100 [00:03<05:00, 3.0s/it, loss/repa=0.54, loss/repa_weighted=0.27, repa/cosine_similarity=0.46]';
context.renderRepaMetrics({},logged);
assert.equal(nodes['#metric-repa'].textContent,'0.54000');
assert.equal(nodes['#metric-repa-group'].hidden,false);
context.openRepaMetricsHelp();
assert(help.help_detail.includes('REPA loss: 0.54000'));
assert(help.help_detail.includes('Added loss: 0.27000'));
assert(help.help_detail.includes('Visual-teacher similarity: 0.460'));
assert(help.help_detail.includes('Explain the teacher agreement.'));
context.renderRepaMetrics({repa_loss:0,repa_weighted:0,repa_similarity:-0.25},logged);
assert.equal(nodes['#metric-repa'].textContent,'0.00000');
assert.equal(nodes['#repa-metrics-help'].dataset.repaWeighted,'0.00000');
assert.equal(nodes['#repa-metrics-help'].dataset.repaSimilarity,'-0.250');
context.renderRepaMetrics({},'Loading teacher: 100%|#| 1/1 [00:03<00:00, 3.0s/it, loss/repa=0.54]');
assert.equal(nodes['#metric-repa'].textContent,'—');
context.renderRepaMetrics({repa_loss:NaN,repa_weighted:Infinity});
assert.equal(nodes['#metric-repa-group'].hidden,true);
assert.equal(nodes['#repa-metrics-help'].dataset.repaWeighted,'—');
assert.equal(nodes['#repa-metrics-help'].dataset.repaSimilarity,'—');
context.renderOptionalMetrics({depth_loss:0.2});
assert.equal(nodes['#metric-depth-group'].hidden,false);
assert.equal(nodes['#metric-dop-group'].hidden,true);
context.renderOptionalMetrics({dop_loss:0});
assert.equal(nodes['#metric-depth-group'].hidden,true);
assert.equal(nodes['#metric-dop-group'].hidden,false);
assert.equal(nodes['#metric-dop'].textContent,'0.00000');
context.renderOptionalMetrics({dop_loss:NaN,depth_loss:Infinity});
assert.equal(nodes['#metric-depth-group'].hidden,true);
assert.equal(nodes['#metric-dop-group'].hidden,true);
'''
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


def test_repa_monitor_help_explains_agreement_and_weighted_loss():
    from modern_gui.settings import settings_schema
    from backends.moss_settings import DEFAULTS, MODE, REPA_METRICS_HELP

    schema = settings_schema({**DEFAULTS, "training_mode": MODE})
    field = next(field for section in schema["sections"] for field in section["fields"]
                 if field["key"] == "moss_repa_enabled")
    assert field["metrics_help"] == REPA_METRICS_HELP
    assert "not an image-quality score" in field["metrics_help"]
    assert "loss 0.54" in field["metrics_help"]


def test_parser_ignores_unrelated_model_loading_bar():
    line = "Loading krea2_turbo.safetensors: 100%|##########| 686/686 [00:23<00:00, 29.13key/s]"
    assert parse_training_line(line) == {}


def test_parser_extracts_training_depth_and_dop_metrics():
    line = (
        "steps:  14%|#4| 47/329 [01:47<10:43, 2.28s/it, "
        "avr_loss=0.119, loss/depth_anchor=0.238, loss/dop=0.044, loss/dop_weighted=0.011]"
    )

    parsed = parse_training_line(line)

    assert parsed == {
        "step": 47,
        "total_steps": 329,
        "loss": 0.119,
        "depth_loss": 0.238,
        "dop_loss": 0.044,
        "dop_weighted": 0.011,
    }


def test_parser_extracts_epochs_and_face_steps():
    assert parse_training_line("epoch 3/10") == {"epoch": 3, "total_epochs": 10}
    assert parse_training_line("step=12/80 loss=0.22") == {"step": 12, "total_steps": 80}


def test_parser_extracts_distinct_repa_metrics_and_diffusion_loss():
    parsed = parse_training_line(
        "steps: 1%|#| 98/25290 [05:03<21:42:58, 3.10s/it, avr_loss=0.942, "
        "loss/diffusion=1.07, loss/repa=5.41e-1, loss/repa_weighted=0.2705, repa/cosine_similarity=0.459]"
    )
    assert parsed["loss"] == 0.942
    assert parsed["diffusion_loss"] == 1.07
    assert parsed["repa_loss"] == 0.541
    assert parsed["repa_weighted"] == 0.2705
    assert parsed["repa_similarity"] == 0.459


def test_parser_ignores_invalid_repa_numbers_and_partial_metric_names():
    assert parse_training_line(
        "loss/repa=nan loss/repa_weighted=1e309 repa/cosine_similarity=inf "
        "other_loss/repa=0.2 loss/repa=0.3broken loss/repa=--2"
    ) == {}
    assert parse_training_line("loss/repa=0, loss/repa_weighted=0, repa/cosine_similarity=-.25") == {
        "repa_loss": 0, "repa_weighted": 0, "repa_similarity": -0.25,
    }


def test_supervisor_keeps_repa_metrics_in_history(monkeypatch, tmp_path):
    from modern_gui import jobs

    monkeypatch.setattr(jobs, "HISTORY_PATH", tmp_path / "jobs.json")
    supervisor = JobSupervisor()
    supervisor._active = {
        "id": "repa-monitor", "name": "repa-monitor", "status": "running",
        "kind": "training", "settings": {"rename_final_artifacts_to_epoch": False},
        "metrics": {"loss_history": []},
    }
    supervisor._append_log("output", "steps: 1%|#| 1/100 [00:03<05:00, 3.0s/it, "
                           "avr_loss=0.9, loss/repa=0.54, loss/repa_weighted=0.27, repa/cosine_similarity=0.46]")
    supervisor._finish(0)
    metrics = supervisor.history()[0]["metrics"]
    assert metrics["repa_loss"] == 0.54
    assert metrics["repa_weighted"] == 0.27
    assert metrics["repa_similarity"] == 0.46


def test_parser_extracts_cache_progress_without_treating_it_as_training_steps():
    parsed = parse_cache_progress(" 42%|####      | 42/100 [00:10<00:14, 4.20it/s]")

    assert parsed["current"] == 42
    assert parsed["total"] == 100
    assert parsed["percent"] == 42
    assert parsed["eta"] == "00:14"
    assert parsed["rate"] == "4.20it/s"
    assert parse_training_line(" 42%|####      | 42/100 [00:10<00:14, 4.20it/s]") == {}


def test_parser_extracts_cache_count_when_tqdm_has_no_total():
    parsed = parse_cache_progress("99it [00:18, 5.26it/s]")

    assert parsed["current"] == 99
    assert parsed["total"] == 0
    assert parsed["rate"] == "5.26it/s"


def test_only_main_training_bar_is_replaceable_progress():
    assert is_training_progress_line("steps:  25%|##5| 373/1510 [16:50<51:20, 2.71s/it]")
    assert not is_training_progress_line("Loading weights: 25%|##5| 3/12 [00:02<00:06]")
    assert not is_training_progress_line("epoch 1/2")


def test_job_monitor_replaces_transient_progress_and_same_step_loss():
    supervisor = JobSupervisor()
    supervisor._active = {"metrics": {"loss_history": []}}
    supervisor._append_log("system", "training started")
    supervisor._append_log("output", "steps:  1%|#| 1/100 [00:01<01:39, 1.0s/it, avr_loss=0.3]")
    supervisor._append_log("output", "steps:  2%|#| 2/100 [00:02<01:38, 1.0s/it, avr_loss=0.2]")
    supervisor._append_log("output", "steps:  2%|#| 2/100 [00:02<01:38, 1.0s/it, avr_loss=0.19]")

    snapshot = supervisor.snapshot()

    assert [entry["message"] for entry in snapshot["log"]] == [
        "training started",
        "steps:  2%|#| 2/100 [00:02<01:38, 1.0s/it, avr_loss=0.19]",
    ]
    assert supervisor._active["metrics"]["loss_history"] == [[1, 0.3], [2, 0.19]]


def test_job_monitor_keeps_cache_progress_during_cache_phase():
    supervisor = JobSupervisor()
    supervisor._active = {"phase": "Caching text", "metrics": {"loss_history": []}}

    supervisor._append_log("output", "42%|####| 42/100 [00:10<00:14, 4.20it/s]")

    assert supervisor._active["metrics"]["cache_progress"] == {
        "current": 42,
        "total": 100,
        "percent": 42,
        "elapsed": "00:10",
        "eta": "00:14",
        "rate": "4.20it/s",
        "text": "42%|####| 42/100 [00:10<00:14, 4.20it/s]",
        "kind": "text",
    }


def test_job_monitor_discards_empty_carriage_return_output():
    supervisor = JobSupervisor()
    supervisor._active = {"metrics": {"loss_history": []}}

    supervisor._append_log("output", "\r")
    supervisor._append_log("output", "")
    supervisor._append_log("output", "useful output")

    assert [entry["message"] for entry in supervisor.snapshot()["log"]] == ["useful output"]


def test_stop_after_current_epoch_waits_for_the_following_epoch_header():
    class Process:
        def __init__(self):
            self.signals = []

        def poll(self):
            return None

        def send_signal(self, signal):
            self.signals.append(signal)

    supervisor = JobSupervisor()
    process = Process()
    supervisor._active = {
        "status": "running",
        "kind": "training",
        "metrics": {"epoch": 1, "total_epochs": 4},
        "settings": {"max_train_epochs": "4"},
    }
    supervisor._process = process

    armed = supervisor.stop_after_current_epoch(True)
    assert armed["stop_after_epoch"] == 1

    supervisor._append_log("output", "epoch 1/4")
    assert supervisor._active["status"] == "running"
    assert not supervisor._active["stop_after_epoch_triggered"]
    assert process.signals == []

    supervisor._append_log("output", "generating sample images at step / sample: 20")
    supervisor._append_log("output", "epoch 2/4")
    assert supervisor._active["status"] == "stopping"
    assert supervisor._active["stop_after_epoch_triggered"]
    assert supervisor._stop_requested
    assert process.signals
