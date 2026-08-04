from types import SimpleNamespace

from musubi_tuner_gui import MusubiTunerGUI


def test_cuda_visible_devices_maps_logical_accelerate_gpu_to_physical_index():
    gui = object.__new__(MusubiTunerGUI)

    hint = gui._resolve_vram_gpu_hint(
        {"CUDA_VISIBLE_DEVICES": "1,0"},
        ["accelerate", "launch", "--gpu_ids", "1", "train.py"],
        [object(), object()],
    )

    assert hint == (0, "CUDA_VISIBLE_DEVICES")


def test_unmapped_default_uses_torch_order(monkeypatch):
    gui = object.__new__(MusubiTunerGUI)
    monkeypatch.setattr(gui, "_torch_nvml_index_map", lambda _handles: [1, 0])

    hint = gui._resolve_vram_gpu_hint({}, ["accelerate", "launch", "train.py"], [object(), object()])

    assert hint == (1, "Torch/NVML")


def test_unmapped_default_reports_unknown_when_torch_mapping_is_unavailable(monkeypatch):
    gui = object.__new__(MusubiTunerGUI)
    monkeypatch.setattr(gui, "_torch_nvml_index_map", lambda _handles: None)

    hint = gui._resolve_vram_gpu_hint({}, ["accelerate", "launch", "train.py"], [object(), object()])

    assert hint == (None, "unknown")


def test_monitor_does_not_choose_gpu_from_incidental_memory_delta():
    gui = object.__new__(MusubiTunerGUI)
    gui._vram_gpu_hint = None
    gui._vram_gpu_hint_source = None
    gui._training_process_ids = lambda: {1234}
    gui._gpu_process_ids = lambda _handle: set()

    result = gui._detect_training_gpu(
        [object(), object()],
        [SimpleNamespace(used=100), SimpleNamespace(used=10_000_000_000)],
    )

    assert result == 0


def test_explicit_gpu_hint_wins_over_process_detection():
    gui = object.__new__(MusubiTunerGUI)
    gui._vram_gpu_hint = 1
    gui._vram_gpu_hint_source = "CUDA_VISIBLE_DEVICES"

    result = gui._detect_training_gpu([object(), object()], [])

    assert result == 1
