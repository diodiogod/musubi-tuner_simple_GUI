"""Build the real Classic widgets without loading/saving user recipes."""
import tkinter as tk
from unittest.mock import patch

from musubi_tuner_gui import MusubiTunerGUI


def test_classic_automatic_controls_preserve_fixed_value():
    root = tk.Tk()
    root.withdraw()
    try:
        with patch.object(MusubiTunerGUI, "_load_last_settings"), patch.object(MusubiTunerGUI, "_load_job_history"):
            app = MusubiTunerGUI(root)
        app.training_mode_var.set("MiniMax H3 (Experimental)")
        app.update_button_states()
        fixed = app.entries["blocks_to_swap"]
        original = fixed.get()
        mode = app.entries["minimax_h3_block_memory_mode"]
        mode.set("Automatic (experimental)")
        app.update_button_states()
        assert str(fixed.cget("state")) == "disabled"
        assert fixed.get() == original
        assert app.entries["minimax_h3_auto_swap_reserve_gb"].get() == "2.0"
        assert not app.hidden_frames["h3_auto_swap_advanced"].winfo_manager()
        mode.set("Fixed blocks (existing)")
        app.update_button_states()
        assert str(fixed.cget("state")) == "normal"
        assert fixed.get() == original
    finally:
        root.destroy()
