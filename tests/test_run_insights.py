import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from job_performance import parse_training_metrics
from modern_gui import run_insights as insights
from modern_gui import server


def test_raw_loss_does_not_replace_reported_average():
    assert parse_training_metrics("avr_loss=0.2, loss/current=0.4, lr=1e-4]") == {
        "loss": .2, "raw_loss": .4, "learning_rate": .0001}
    assert parse_training_metrics("loss/current=0, lr=nan") == {"loss": 0, "raw_loss": 0}


def test_recover_chart_deduplicates_updates_and_keeps_legacy_honest(tmp_path, monkeypatch):
    monkeypatch.setattr("job_performance.JOB_LOG_DIR", tmp_path)
    path = tmp_path / "train.log"
    path.write_text("\n".join([
        "[time] [output] epoch 1/2",
        "[time] [output] steps: 50%|##| 1/2 [00:01, 1s/it, avr_loss=.2, loss/current=.4, lr=1e-4]",
        "[time] [output] steps: 50%|##| 1/2 [00:01, 1s/it, avr_loss=.21, loss/current=.42, lr=1e-4]",
        "[time] [output] epoch 2/2",
        "[time] [output] steps: 100%|##| 2/2 [00:02, 1s/it, avr_loss=.1, loss/current=.1, lr=0]",
    ]), encoding="utf-8")
    job = {"console_log_path": str(path)}
    chart = insights.run_chart(job)
    assert chart["points"] == [[1, 1, .42, .21, .0001], [2, 2, .1, .1, 0]]
    assert chart["has_raw"] and chart["has_lr"]
    # Cache invalidates when the log changes.
    path.write_text("steps: 100%|##| 2/2 [00:02, 1s/it, avr_loss=.05]", encoding="utf-8")
    old = insights.run_chart(job)
    assert old["points"] == [[2, None, None, .05, None]]
    assert not old["has_raw"] and not old["has_lr"]
    assert insights.run_chart({"loss_history": [[1, .2], [2, .1]]})["points"] == [[1, None, None, .2, None], [2, None, None, .1, None]]


def test_notes_stable_identity_unicode_and_independent_of_recipes(tmp_path, monkeypatch):
    monkeypatch.setattr(insights, "NOTES_PATH", tmp_path / "notes.json")
    a = {"id": "first", "_source": "web", "_history_index": 0, "settings": {"training_comment": "original"}}
    b = dict(a, id="second")
    insights.save_note(a, insights.run_key(a), "Época 8 — melhor. 中文")
    insights.save_note(b, insights.run_key(b), "Try less LR")
    reordered = dict(a, _history_index=4)
    assert insights.decorate_run(reordered, insights.load_notes())["user_note"] == "Época 8 — melhor. 中文"
    assert a["settings"]["training_comment"] == "original"
    with pytest.raises(ValueError, match="History changed"):
        insights.save_note(a, insights.run_key(b), "Wrong run")
    with pytest.raises(ValueError, match="20,000"):
        insights.save_note(a, insights.run_key(a), "x" * 20001)
    insights.NOTES_PATH.write_text("broken JSON", encoding="utf-8")
    with pytest.raises(ValueError, match="left unchanged"):
        insights.save_note(a, insights.run_key(a), "Do not overwrite existing notes")
    assert insights.NOTES_PATH.read_text(encoding="utf-8") == "broken JSON"


def test_files_show_only_this_run_and_do_not_open_weights(tmp_path):
    (tmp_path / "example.safetensors").write_bytes(b"metadata not needed")
    (tmp_path / "example-000001-state").mkdir()
    (tmp_path / "other.safetensors").write_bytes(b"another run")
    (tmp_path / "example2.safetensors").write_bytes(b"another run with a similar name")
    assert {p["name"] for p in insights.run_files({"settings": {"output_dir": str(tmp_path), "output_name": "example"}})["checkpoints"]} == {"example.safetensors", "example-000001-state"}


def test_notes_api_refuses_stale_index_and_persists_note(tmp_path, monkeypatch):
    monkeypatch.setattr(insights, "NOTES_PATH", tmp_path / "notes.json")
    jobs = [{"id": "first", "kind": "training", "metrics": {"loss_history": [[1, .2], [2, .1]]}}, {"id": "second"}]
    monkeypatch.setattr(server.SUPERVISOR, "history", lambda: jobs)
    monkeypatch.setattr(server, "load_desktop_history", lambda: [])
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.MusubiWebHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_port}"
    def request(path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        return json.load(urllib.request.urlopen(urllib.request.Request(base+path, data=data, headers={"Content-Type": "application/json"})))
    try:
        job = request("/api/jobs")["jobs"][0]
        payload = {"source": "web", "index": 0, "key": job["insights_key"], "note": "A good run"}
        request("/api/jobs/notes", payload)
        assert request("/api/jobs")["jobs"][0]["user_note"] == "A good run"
        assert len(request(f'/api/jobs/insights?source=web&index=0&key={job["insights_key"]}')["chart"]["points"]) == 2
        jobs.reverse()
        with pytest.raises(urllib.error.HTTPError):
            request("/api/jobs/notes", payload)
        assert len(insights.load_notes()) == 1
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_chart_math_handles_sparse_raw_zero_lr_outliers_and_epochs():
    if not shutil.which("node"):
        pytest.skip("Node.js is required for frontend math checks")
    source = Path(__file__).resolve().parents[1] / "modern_gui/static/loss-chart.js"
    script = r'''
const assert=require('node:assert/strict'), c=require(process.argv[1]);
assert.deepEqual(c.ema([[1,2],[2,4],[3,8]],.5),[[1,2],[2,3],[3,5.5]]);
const data={points:[[1,1,2,2,0],[2,1,4,3,.001],[3,2,6,4,.001]]};
assert.deepEqual(c.epochMeans(data.points),[[1.5,3,1],[3,6,2]]);
assert.equal(c.normalize(data).hasLR,true);
assert.equal(c.normalize([[1,.2],[2,.1]]).hasRaw,false);
assert.equal(c.model([[1,.2],[2,.1]],{epoch:true}).series.length,1);
assert.ok(c.model(data,{log:true,lr:true}).lr.length===3);
assert.ok(c.domain([0,-1,1,10],true,false).every(Number.isFinite));
const values=[...Array(100)].map((_,i)=>1+i/100);values.push(1000);
assert.ok(c.domain(values,false,true)[1]<3);
assert.ok(c.domain(values,false,false)[1]>900);
assert.ok(!c.spark([[1,.2],[2,.1]]).includes('NaN'));
const historical=c.model([[1,9],[5000,1],[5001,1],[5002,1]],{clip:true});
assert.ok(historical.range[1]>=9); // Never crop the beginning of a sparse historical trend.
'''
    subprocess.run(["node", "-e", script, str(source)], check=True, capture_output=True, text=True)
