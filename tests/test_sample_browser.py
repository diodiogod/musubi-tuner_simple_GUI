import json
from pathlib import Path
import shutil
import subprocess
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from urllib.parse import urlencode

import pytest
from PIL import Image

from modern_gui import sample_metadata
from modern_gui.samples import discover_samples


def test_saved_prompt_labels_and_changed_captions_stay_separate(tmp_path, monkeypatch):
    history = tmp_path / "history.json"
    root = tmp_path / "outputs"
    run = root / "portrait"
    run.mkdir(parents=True)
    for epoch in (1, 2):
        Image.new("RGB", (64, 64)).save(run / f"portrait_e{epoch:06}_00_2026072{epoch}120000_42.png")
    history.write_text(json.dumps([
        {"started_at": "2026-07-20T00:00:00+00:00", "settings": {"output_name": "portrait", "output_dir": str(root), "sample_prompts_data": [{"prompt": "A person outdoors", "_library_name": "Outdoor portrait"}]}},
        {"started_at": "2026-07-22T00:00:00+00:00", "settings": {"output_name": "portrait", "output_dir": str(root), "sample_prompts_data": [{"prompt": "A person indoors", "_library_name": "Indoor portrait"}]}},
    ]), encoding="utf-8")
    monkeypatch.setattr(sample_metadata, "HISTORY_PATHS", (history,))
    result = discover_samples(str(root), "portrait")
    assert len(result["groups"]) == 2
    assert {g["prompt_name"] for g in result["groups"]} == {"Outdoor portrait", "Indoor portrait"}
    assert {i["metadata_source"] for g in result["groups"] for i in g["items"]} == {"Saved run settings"}


def test_current_run_scope_filters_flat_outputs_before_limiting(tmp_path):
    (tmp_path / "current_e000001_00_20260721120000_42.png").write_bytes(b"current")
    (tmp_path / "current2_e000001_00_20260721120000_42.png").write_bytes(b"other")
    previous = tmp_path / "old"
    previous.mkdir()
    (previous / "old_e000001_00_20260721120000_42.png").write_bytes(b"old")
    result = discover_samples(str(tmp_path), "current", current_run_only=True)
    assert result["count"] == 1
    result = discover_samples(str(tmp_path), "current", current_run_only=True, source_paths=[str(previous)])
    assert result["count"] == 2
    result = discover_samples(str(tmp_path / "not-created"), "future-run", current_run_only=True, source_paths=[str(previous)])
    assert result["count"] == 1  # A missing current output must not hide an explicitly selected old folder.


def test_images_and_unknown_prompts_stay_readable_without_history(tmp_path, monkeypatch):
    monkeypatch.setattr(sample_metadata, "HISTORY_PATHS", ())
    Image.new("RGB", (64, 32)).save(tmp_path / "example_e000001_00_20260721120000_0.png")
    result = discover_samples(str(tmp_path))
    item = result["groups"][0]["items"][0]
    assert item["width"] == 64 and item["height"] == 32
    assert "prompt_text" not in item or item["prompt_text"] == ""
    assert item["seed"] == "0"


def test_gallery_matrix_and_selection_preserve_identity_and_all_variants():
    if not shutil.which("node"):
        pytest.skip("Node.js is required for sampling UI model checks")
    module = Path(__file__).resolve().parents[1] / "modern_gui/static/sample-browser.js"
    script = r'''
const assert=require('node:assert/strict'), m=require(process.argv[1]);
const item=(path,sequence,prompt=0,source='run',kind='epoch')=>({path,url:path,sequence,sequence_kind:kind,prompt_index:prompt,source_path:source,source_label:source,media_kind:'image',modified:sequence});
const a=item('a',1), b=item('b',3), duplicate=item('b-again',3), other=item('c',1,1), olderRun=item('d',1,0,'older');
const data={groups:[{key:'p0',prompt_index:0,items:[b,a,duplicate,olderRun]},{key:'p1',prompt_index:1,items:[other]}],ungrouped:[]};
const matrix=m.matrices(data)[0];
assert.equal(matrix.columns.length,3); // epoch 1 and 3, plus another source's epoch 1
assert.equal(matrix.rows[0].cells.get(JSON.stringify(['run','epoch',3])).length,2); // never overwrite a take
assert.equal(matrix.rows[1].cells.get(JSON.stringify(['run','epoch',3])),undefined); // missing stays missing
assert.deepEqual(m.chronological([b,a]).map(i=>i.path),['a','b']);
assert.deepEqual(m.toggleSelection(['a','b'],'c'),['a','c']); // replacing B keeps A pinned
assert.deepEqual(m.toggleSelection(['a','b'],'a'),['b']);
assert.equal(m.comparisonContext(data,['a','c']).items.length,5);
assert.equal(m.comparisonContext(data,['a','missing']),null);
assert.equal(m.comparisonContext({groups:[],ungrouped:[a,{...b,media_kind:'video'}]},['a','b']),null);
assert.equal(m.promptTitle({}), 'Preview');
assert.equal(m.allItems({groups:[{items:[a]}],ungrouped:[a]}).length,1);
assert.deepEqual(m.matrices({groups:[{key:'mixed',items:[a,item('step',2,0,'run','step')]}]}).map(x=>x.kind),['epoch','step']);
'''
    subprocess.run(["node", "-e", script, str(module)], check=True, capture_output=True, text=True)


def test_sample_api_source_filter_keeps_unselected_folders_out(tmp_path, monkeypatch):
    from modern_gui import server
    current, previous, unrelated = [tmp_path / name for name in ("current", "previous", "unrelated")]
    for path in (current, previous, unrelated):
        path.mkdir()
        (path / f"{path.name}_e000001_00_20260721120000_42.png").write_bytes(b"image")
    monkeypatch.setattr(server, "load_sample_sources", lambda: [{"path": str(p), "label": p.name} for p in (previous, unrelated)])
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.MusubiWebHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    def get(extra):
        query = urlencode({"output_dir": str(tmp_path), "output_name": "current", "run_only": "1", **extra}, doseq=True)
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_port}/api/samples?{query}"))
    try:
        assert get({"include_sources": "0"})["count"] == 1
        chosen = get({"include_sources": "1", "source_filter": "1", "source": [str(previous)]})
        assert chosen["count"] == 2
        assert {p["label"] for p in chosen["sources"]} == {"current", "previous"}
        assert get({"include_sources": "1", "source_filter": "1", "source": [str(tmp_path / "not-registered")]})["count"] == 1
        assert get({"include_sources": "1"})["count"] == 3  # Existing API clients keep their contract.
    finally:
        httpd.shutdown()
        httpd.server_close()
