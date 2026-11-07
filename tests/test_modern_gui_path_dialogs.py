import sys
from types import SimpleNamespace

import pytest

from modern_gui.path_dialogs import choose_path, initial_location


def test_model_file_starts_in_parent_folder_with_filename(tmp_path):
    model = tmp_path / "a model.safetensors"
    model.touch()
    assert initial_location(f'"{model}"') == {"initialdir": str(tmp_path), "initialfile": model.name}
    assert initial_location(str(model), directory=True) == {"initialdir": str(tmp_path)}


def test_missing_or_existing_folders_never_become_file_initialdir(tmp_path):
    assert initial_location(str(tmp_path)) == {"initialdir": str(tmp_path)}
    assert initial_location(str(tmp_path / "moved" / "model.safetensors")) == {"initialdir": str(tmp_path)}
    assert initial_location("") == {}
    assert initial_location(str(tmp_path / "new.safetensors")) == {
        "initialdir": str(tmp_path), "initialfile": "new.safetensors",
    }


@pytest.mark.parametrize("kind", ["file", "directory"])
@pytest.mark.parametrize("fail", [False, True])
def test_picker_uses_only_tk_and_always_destroys_hidden_owner(monkeypatch, tmp_path, kind, fail):
    calls = []

    class Root:
        def withdraw(self):
            calls.append("hidden")

        def title(self, value):
            pass

        def attributes(self, *args):
            pass

        def destroy(self):
            calls.append("destroyed")

    def dialog(**options):
        calls.append(options)
        if fail:
            raise RuntimeError("dialog failed")
        return ""

    dialogs = SimpleNamespace(askopenfilename=dialog, askdirectory=dialog)
    monkeypatch.setitem(sys.modules, "tkinter", SimpleNamespace(Tk=Root, filedialog=dialogs))
    before = set(sys.modules)
    if fail:
        with pytest.raises(RuntimeError, match="dialog failed"):
            choose_path(kind, str(tmp_path / "model.safetensors"))
    else:
        assert choose_path(kind, str(tmp_path / "model.safetensors")) == ""
    assert calls[0] == "hidden" and calls[-1] == "destroyed"
    assert calls[1]["initialdir"] == str(tmp_path)
    assert ("initialfile" in calls[1]) == (kind == "file")
    assert "musubi_tuner_gui" not in set(sys.modules) - before
