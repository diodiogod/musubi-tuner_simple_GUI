import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import threading
import json
import os
import re
import time
import sys
from pathlib import Path

from backends import wan as wan_backend, flux2 as flux2_backend, krea2 as krea2_backend
from backends.flux2 import FLUX2_VERSION_MAP

# --- Dependency Check ---
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib
    matplotlib.use("TkAgg")
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

try:
    import pynvml
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False

# --- Helper Class for Tooltips ---
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        try:
            x, y, _, _ = self.widget.bbox("insert")
            x += self.widget.winfo_rootx() + 25
            y += self.widget.winfo_rooty() + 25
        except Exception:
            x = self.widget.winfo_rootx() + 25
            y = self.widget.winfo_rooty() + 25
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, justify='left',
                         background="#FFFFE0", relief='solid', borderwidth=1,
                         font=("Calibri", "10", "normal"), wraplength=400)
        label.pack(ipadx=1)

    def hide_tooltip(self, event):
        if self.tooltip:
            self.tooltip.destroy()
        self.tooltip = None

# --- Main Application ---
class MusubiTunerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Musubi Tuner GUI - WAN 2.2 LoRA Training")
        self.root.geometry("1200x900")

        self.entries = {}
        self.hidden_frames = {}
        self.training_mode_var = tk.StringVar(value="Wan 2.2")
        self.setup_styles()

        self.current_process = None
        self.monitoring_active = False
        self.vram_thread = None
        self.loss_data = []
        self.peak_vram = 0
        self.command_sequence = []
        self.last_line_was_progress = False
        self.current_step = 0
        self.current_total_steps = 0
        self.current_epoch_num = 0
        self.current_epoch_total = 0
        self.sample_watcher_active = False
        self._sample_watcher_thread = None
        self._last_sample_files = []
        self._sample_list_frame = None
        self._sample_prompts_data = []  # list of dicts
        self._temp_prompts_file = None  # path to auto-written temp .txt

        self.create_interface()
        self.load_default_settings()
        self._load_last_settings()
        self.update_button_states()

    def setup_styles(self):
        BG_COLOR = '#2B2B2B'; TEXT_COLOR = '#D3D3D3'; FIELD_BG_COLOR = '#3C3F41'
        SELECT_BG_COLOR = '#4A6185'; BORDER_COLOR = '#555555'; ERROR_BORDER = '#E53935'
        
        self.root.configure(bg=BG_COLOR)
        style = ttk.Style()
        try: style.theme_use('clam')
        except Exception: pass
        
        style.configure('.', background=BG_COLOR, foreground=TEXT_COLOR, font=('Calibri', 9))
        style.configure('TLabel', font=('Calibri', 10)); style.configure('TFrame', background=BG_COLOR)
        style.configure('TLabelframe', background=BG_COLOR, bordercolor=BORDER_COLOR, relief='solid', borderwidth=1)
        style.configure('TLabelframe.Label', background=BG_COLOR, foreground=TEXT_COLOR, font=('Calibri', 11, 'bold'))
        style.configure('TNotebook', background=BG_COLOR, borderwidth=0)
        style.configure('TNotebook.Tab', background='#3C3F41', foreground=TEXT_COLOR, padding=[10, 5], borderwidth=0)
        style.map('TNotebook.Tab', background=[('selected', BG_COLOR)])
        style.configure('TButton', background='#3C3F41', foreground=TEXT_COLOR, font=('Calibri', 10), borderwidth=1, relief='solid')
        style.map('TButton', background=[('active', '#4E5254'), ('pressed', '#585C5E')], bordercolor=[('active', BORDER_COLOR)], foreground=[('disabled', '#6A6A6A')])
        style.configure('TEntry', foreground=TEXT_COLOR, fieldbackground=FIELD_BG_COLOR, insertcolor=TEXT_COLOR, borderwidth=1, relief='solid', bordercolor=BORDER_COLOR, padding=3)
        style.map('TCombobox', fieldbackground=[('readonly', FIELD_BG_COLOR)], foreground=[('readonly', TEXT_COLOR)], selectbackground=[('readonly', SELECT_BG_COLOR)])
        self.root.option_add('*TCombobox*Listbox.background', FIELD_BG_COLOR); self.root.option_add('*TCombobox*Listbox.foreground', TEXT_COLOR)
        self.root.option_add('*TCombobox*Listbox.selectBackground', SELECT_BG_COLOR); self.root.option_add('*TCombobox*Listbox.selectForeground', TEXT_COLOR)
        style.configure('TCheckbutton', font=('Calibri', 10)); style.configure('Title.TLabel', font=('Calibri', 16, 'bold'))
        style.configure('Status.TLabel', font=('Calibri', 11, 'bold')); style.configure('TProgressbar', thickness=20, background=SELECT_BG_COLOR, troughcolor=FIELD_BG_COLOR)
        style.configure('Invalid.TEntry', fieldbackground=FIELD_BG_COLOR, bordercolor=ERROR_BORDER, foreground=TEXT_COLOR, relief='solid', borderwidth=1)
        style.configure('Valid.TEntry', fieldbackground=FIELD_BG_COLOR, bordercolor=BORDER_COLOR, foreground=TEXT_COLOR, relief='solid', borderwidth=1)

    def create_interface(self):
        self.root.grid_columnconfigure(0, weight=1); self.root.grid_rowconfigure(0, weight=1)
        canvas = tk.Canvas(self.root, bg='#2B2B2B', highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", tags="frame")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig('frame', width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew"); scrollbar.grid(row=0, column=1, sticky="ns")
        self.root.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        main_frame = ttk.Frame(scrollable_frame); main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.title_label = ttk.Label(main_frame, text="Musubi Tuner - WAN 2.2 LoRA Training", style='Title.TLabel')
        self.title_label.pack(pady=(0, 10), anchor='w')

        # Training Mode selector
        mode_frame = ttk.LabelFrame(main_frame, text="Training Mode"); mode_frame.pack(fill="x", pady=(0, 10))
        mode_inner = ttk.Frame(mode_frame); mode_inner.pack(fill="x", padx=10, pady=8)
        ttk.Label(mode_inner, text="Mode:").pack(side="left", padx=(0, 8))
        self.mode_combo = ttk.Combobox(mode_inner, textvariable=self.training_mode_var,
                                       values=["Wan 2.2", "Flux.2 Klein", "Flux.2 Dev", "Krea 2"],
                                       state="readonly", width=20)
        self.mode_combo.pack(side="left"); self.mode_combo.bind("<MouseWheel>", lambda e: "break")
        self.mode_combo.bind("<<ComboboxSelected>>", self.on_training_mode_change)
        self.mode_note_label = ttk.Label(mode_inner, text="", foreground="#AAAAAA")
        self.mode_note_label.pack(side="left", padx=(15, 0))

        self.create_settings_buttons(main_frame)

        self.notebook = ttk.Notebook(main_frame); self.notebook.pack(fill="both", expand=True, pady=(10, 0))

        self.create_model_paths_tab()
        self.create_training_params_tab()
        self.create_advanced_tab()
        self.create_samples_tab()
        self.create_run_monitor_tab()
        self.create_convert_lora_tab()
        self.create_accelerate_config_tab()

    def create_settings_buttons(self, parent):
        button_frame = ttk.Frame(parent); button_frame.pack(fill="x", pady=(0, 10), anchor='w')
        ttk.Button(button_frame, text="Load Settings", command=self.load_settings).pack(side="left", padx=(0, 5))
        ttk.Button(button_frame, text="Save Settings", command=self.save_settings).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.load_default_settings).pack(side="left", padx=5)

    def _add_widget(self, parent, key, label, tooltip, kind='entry', options=None, is_required=False, validate_num=False, is_path=False, is_dir=False, default_val=False, command=None):
        frame = ttk.Frame(parent); frame.pack(fill="x", padx=5, pady=(5, 8))
        if kind != 'checkbox': ttk.Label(frame, text=label).pack(anchor="w")
        
        widget = None
        if kind == 'path_entry':
            path_frame = ttk.Frame(frame); path_frame.pack(fill="x", pady=(2, 0))
            widget = ttk.Entry(path_frame)
            widget.pack(side="left", fill="x", expand=True)
            filetypes = options if isinstance(options, list) else None
            def browse():
                path = filedialog.askdirectory() if is_dir else filedialog.askopenfilename(filetypes=filetypes)
                if path: widget.delete(0, tk.END); widget.insert(0, path); self.update_button_states()
            ttk.Button(path_frame, text="Browse", command=browse).pack(side="right", padx=(5, 0))
        elif kind == 'combobox':
            widget = ttk.Combobox(frame, values=options, state="readonly")
            if options: widget.set(options[0])
            widget.pack(fill="x", pady=(2, 0)); widget.bind("<MouseWheel>", lambda e: "break")
            if command: widget.bind("<<ComboboxSelected>>", command)
        elif kind == 'checkbox':
            var = tk.BooleanVar(value=default_val)
            def chained_command(event=None):
                if command and callable(command): command()
                self.update_button_states()
            widget = ttk.Checkbutton(frame, text=label, variable=var, command=chained_command)
            widget.var = var; widget.pack(anchor="w", padx=5, pady=2)
        else:
            vcmd = (self.root.register(self.validate_number), '%P') if validate_num else None
            widget = ttk.Entry(frame, validate="key", validatecommand=vcmd); widget.pack(fill="x", pady=(2, 0))

        if tooltip: ToolTip(widget, tooltip)
        self.entries[key] = widget
        widget.is_required = is_required; widget.is_path = is_path
        if isinstance(widget, ttk.Entry):
            widget.bind("<FocusOut>", self.update_button_states); widget.bind("<KeyRelease>", self.update_button_states)
        return widget
    
    def create_model_paths_tab(self):
        frame = ttk.Frame(self.notebook); self.notebook.add(frame, text="Model Paths & Dataset")

        dataset_frame = ttk.LabelFrame(frame, text="Dataset Configuration"); dataset_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(dataset_frame, "dataset_config", "Dataset Config (TOML):", "Path to .toml dataset configuration file.", kind='path_entry', options=[("TOML files", "*.toml")], is_required=True, is_path=True)

        # ---- WAN 2.2 DiT section ----
        self.hidden_frames['wan_dit'] = ttk.LabelFrame(frame, text="DiT Model Selection")
        self.hidden_frames['wan_dit'].pack(fill="x", padx=10, pady=10)
        self._add_widget(self.hidden_frames['wan_dit'], "is_i2v", "Is I2V Training?", "IMPORTANT: I2V models REQUIRE VIDEO data (multiple frames per sample), not static images. If you only have images, use T2V task instead. I2V is for training with video datasets.", kind='checkbox', command=self.update_button_states)

        high_noise_frame = ttk.LabelFrame(self.hidden_frames['wan_dit'], text="High Noise Model (T2V: 875-1000 / I2V: 900-1000)"); high_noise_frame.pack(fill="x", padx=5, pady=5)
        self._add_widget(high_noise_frame, "train_high_noise", "Train High Noise Model", "Enable to train the high noise model.", kind='checkbox', command=self.update_button_states)
        self._add_widget(high_noise_frame, "dit_high_noise", "DiT High Noise Model Path:", "Path to the high noise DiT model.", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_path=True)
        self._add_widget(high_noise_frame, "min_timestep_high", "Min Timestep:", "Minimum timestep for this model. (e.g., 875)", validate_num=True)
        self._add_widget(high_noise_frame, "max_timestep_high", "Max Timestep:", "Maximum timestep for this model. (e.g., 1000)", validate_num=True)

        low_noise_frame = ttk.LabelFrame(self.hidden_frames['wan_dit'], text="Low Noise Model (T2V: 0-875 / I2V: 0-900)"); low_noise_frame.pack(fill="x", padx=5, pady=(5, 10))
        self._add_widget(low_noise_frame, "train_low_noise", "Train Low Noise Model", "Enable to train the low noise model.", kind='checkbox', command=self.update_button_states)
        self._add_widget(low_noise_frame, "dit_low_noise", "DiT Low Noise Model Path:", "Path to the low noise DiT model.", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_path=True)
        self._add_widget(low_noise_frame, "min_timestep_low", "Min Timestep:", "Minimum timestep for this model. (e.g., 0)", validate_num=True)
        self._add_widget(low_noise_frame, "max_timestep_low", "Max Timestep:", "Maximum timestep for this model. (e.g., 875)", validate_num=True)

        # ---- WAN 2.2 other model paths ----
        self.hidden_frames['wan_models'] = ttk.LabelFrame(frame, text="Text Encoder & CLIP")
        self.hidden_frames['wan_models'].pack(fill="x", padx=10, pady=10)
        self._add_widget(self.hidden_frames['wan_models'], "clip_model", "CLIP Model (Optional):", "Path to optional CLIP model. Required for Wan2.1 I2V training (not needed for Wan2.2). Only needed if you're doing I2V with video data.", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_path=True)
        self._add_widget(self.hidden_frames['wan_models'], "t5_model", "T5 Text Encoder:", "Path to T5 text encoder model. Required.", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_required=True, is_path=True)

        # ---- FLUX.2 model paths section ----
        self.hidden_frames['flux2_model_paths'] = ttk.LabelFrame(frame, text="Flux.2 Model Paths")
        # (packed/hidden by on_training_mode_change)

        flux2_ver_frame = ttk.Frame(self.hidden_frames['flux2_model_paths']); flux2_ver_frame.pack(fill="x", padx=5, pady=(8, 2))
        ttk.Label(flux2_ver_frame, text="Model Version:").pack(anchor="w")
        flux2_ver_combo = ttk.Combobox(flux2_ver_frame, values=list(FLUX2_VERSION_MAP.keys()), state="readonly")
        flux2_ver_combo.set("Klein Base 4B ★"); flux2_ver_combo.pack(fill="x", pady=(2, 0))
        flux2_ver_combo.bind("<MouseWheel>", lambda e: "break")
        flux2_ver_combo.bind("<<ComboboxSelected>>", self.update_button_states)
        self.entries["flux2_model_version"] = flux2_ver_combo

        self.flux2_note_label = ttk.Label(self.hidden_frames['flux2_model_paths'],
                                           text="★ Base variants recommended for training — distilled models (4B/9B) are for inference only.",
                                           foreground="#FFCC66", font=("Calibri", 9, "italic"))
        self.flux2_note_label.pack(anchor="w", padx=8, pady=(0, 4))

        self._add_widget(self.hidden_frames['flux2_model_paths'], "flux2_dit_model", "DiT Model:", "Path to the Flux.2 DiT model (.safetensors).", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_required=True, is_path=True)
        self._add_widget(self.hidden_frames['flux2_model_paths'], "flux2_text_encoder", "Text Encoder (Qwen3 or Mistral3):", "Path to the Qwen3 or Mistral3 text encoder directory or safetensors file.", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_required=True, is_path=True)
        self._add_widget(self.hidden_frames['flux2_model_paths'], "fp8_text_encoder", "FP8 Text Encoder", "Load the text encoder in FP8 precision to reduce VRAM.", kind='checkbox')

        # ---- Krea 2 model paths section ----
        self.hidden_frames['krea2_model_paths'] = ttk.LabelFrame(frame, text="Krea 2 Model Paths")

        self.krea2_note_label = ttk.Label(
            self.hidden_frames['krea2_model_paths'],
            text="Train on RAW DiT. Qwen-Image VAE is required. Qwen3-VL text encoder is only required for text re-caching and sample generation. Upstream starting point: bf16, rank 32, alpha 32, timestep_sampling=krea2_shift.",
            foreground="#FFCC66", font=("Calibri", 9, "italic")
        )
        self.krea2_note_label.pack(anchor="w", padx=8, pady=(8, 4))

        self._add_widget(self.hidden_frames['krea2_model_paths'], "krea2_dit_model", "RAW DiT Model:", "Path to the Krea 2 RAW DiT model (.safetensors). Required for training.", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_required=True, is_path=True)
        self._add_widget(self.hidden_frames['krea2_model_paths'], "krea2_text_encoder", "Text Encoder (Qwen3-VL-4B):", "Path to the Qwen3-VL-4B-Instruct safetensors file. Required for text re-caching and sample generation during training.", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_path=True)
        self._add_widget(self.hidden_frames['krea2_model_paths'], "krea2_turbo_dit", "Turbo DiT (Optional):", "Optional distilled Turbo DiT safetensors path. Used only for sample generation during training to preview Turbo inference behavior.", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_path=True)
        self._add_widget(self.hidden_frames['krea2_model_paths'], "krea2_turbo_dit_cache", "Cache Turbo DiT in RAM", "Keeps the optional Turbo DiT weights resident in CPU RAM for faster sampling. Only relevant when Turbo DiT is set.", kind='checkbox')
        self._add_widget(self.hidden_frames['krea2_model_paths'], "krea2_projector_diff", "Projector Patch (Optional):", "Optional tiny Krea 2 projector diff safetensors patch. Applied to the RAW training base model and also to optional Turbo sample generation so previews stay consistent.", kind='path_entry', options=[("Safetensors", "*.safetensors")], is_path=True)
        self._add_widget(self.hidden_frames['krea2_model_paths'], "krea2_projector_diff_strength", "Patch Strength:", "Multiplier for the optional projector patch. Example: 2.5", validate_num=True)

        # VAE shared by both modes — store reference for pack ordering
        models_frame = ttk.LabelFrame(frame, text="VAE Model"); models_frame.pack(fill="x", padx=10, pady=10)
        self._vae_frame = models_frame
        self._add_widget(models_frame, "vae_model", "VAE Model:", "Path to VAE model (.safetensors or .pt). Required for training and caching.", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_required=True, is_path=True)

        output_frame = ttk.LabelFrame(frame, text="Output Configuration"); output_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(output_frame, "output_dir", "Output Directory:", "Base directory to save trained LoRAs. A subfolder will be automatically created.", kind='path_entry', is_dir=True, is_required=True, is_path=True)
        self._add_widget(output_frame, "output_name", "Output Name:", "Base filename for output LoRA (e.g., 'my_character'). Suffixes like '_LowNoise' will be added automatically.", is_required=True)

    def create_training_params_tab(self):
        frame = ttk.Frame(self.notebook); self.notebook.add(frame, text="Training Parameters")
        basic_frame = ttk.LabelFrame(frame, text="Basic Training Parameters"); basic_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(basic_frame, "learning_rate", "Learning Rate:", "The speed at which the model learns. Common values are 1e-4, 2e-4, 3e-4.", is_required=True, validate_num=True)
        self._add_widget(basic_frame, "max_train_epochs", "Max Train Epochs:", "The total number of times the training process will iterate over the entire dataset.", is_required=True, validate_num=True)
        self._add_widget(basic_frame, "save_every_n_epochs", "Save Every N Epochs:", "Frequency of saving checkpoints based on epochs. '1' saves after every epoch.", validate_num=True)
        self._add_widget(basic_frame, "save_every_n_steps", "Save Every N Steps:", "Frequency of saving checkpoints based on steps. Leave empty to disable.", validate_num=True)
        self._add_widget(basic_frame, "seed", "Seed:", "A number to ensure reproducible training results. Any integer will do.", validate_num=True)
        
        network_container = ttk.Frame(frame); network_container.pack(fill="x", padx=10, pady=10)
        network_type_frame = ttk.LabelFrame(network_container, text="Network Type"); network_type_frame.pack(fill="x", pady=(0, 5))
        self._add_widget(network_type_frame, "network_type", "Network Type:", "LoRA: standard, efficient. LoHa: uses Hadamard product, often better quality — use lower ranks (4-32). LoKr: uses Kronecker product, more expressive.", kind='combobox', options=["LoRA", "LoHa", "LoKr"], command=self.update_button_states)

        self.hidden_frames['lokr_factor'] = ttk.Frame(network_container)
        self._add_widget(self.hidden_frames['lokr_factor'], "lokr_factor", "LoKr Factor:", "Controls how LoKr splits weight dimensions via Kronecker factorization. -1 = auto (recommended). Positive values force a specific factor (e.g., 4, 8).", validate_num=False)

        self.hidden_frames['low_noise_lora_params'] = ttk.LabelFrame(network_container, text="Low Noise Network Parameters")
        self._add_widget(self.hidden_frames['low_noise_lora_params'], "network_dim_low", "Network Dimension (Rank):", "Controls network capacity. LoRA: 32-128 typical. LoHa: use lower values (4-32) — the Hadamard product squares expressiveness so smaller ranks go further. LoKr: similar range to LoRA.", is_required=True, validate_num=True)
        self._add_widget(self.hidden_frames['low_noise_lora_params'], "network_alpha_low", "Network Alpha:", "Scaling factor for network weights. Often set to half of Network Dimension, or equal to it for LoHa/LoKr.", is_required=True, validate_num=True)

        self.hidden_frames['high_noise_lora_params'] = ttk.LabelFrame(network_container, text="High Noise Network Parameters")
        self._add_widget(self.hidden_frames['high_noise_lora_params'], "network_dim_high", "Network Dimension (Rank):", "Leave blank to use the same as the Low Noise model. If different, a separate training run will be executed.", is_required=False, validate_num=True)
        self._add_widget(self.hidden_frames['high_noise_lora_params'], "network_alpha_high", "Network Alpha:", "Leave blank to use the same as the Low Noise model.", is_required=False, validate_num=True)

        optimizer_frame = ttk.LabelFrame(frame, text="Optimizer Settings"); optimizer_frame.pack(fill="x", padx=10, pady=10)
        # Optimizer Type: preset dropdown + optional custom path entry
        _opt_presets = ["adamw", "adamw8bit", "adafactor", "lion", "prodigy",
                        "prodigyplus.prodigy_plus_schedulefree.ProdigyPlusScheduleFree", "Custom..."]
        _opt_frame = ttk.Frame(optimizer_frame); _opt_frame.pack(fill="x", padx=5, pady=(5, 0))
        ttk.Label(_opt_frame, text="Optimizer Type:").pack(anchor="w")
        _opt_var = tk.StringVar(value="adamw8bit")   # what get_settings() reads
        self.entries["optimizer_type"] = _opt_var
        _opt_combo_var = tk.StringVar(value="adamw8bit")
        _opt_combo = ttk.Combobox(_opt_frame, textvariable=_opt_combo_var,
                                  values=_opt_presets, state="readonly")
        _opt_combo.pack(fill="x", pady=(2, 0)); _opt_combo.bind("<MouseWheel>", lambda e: "break")
        ToolTip(_opt_combo, "'adamw8bit' is a memory-efficient and stable default. 'prodigy' can also work well. Choose 'Custom...' to type any full import path.")
        _custom_frame = ttk.Frame(_opt_frame)
        _custom_entry = ttk.Entry(_custom_frame)
        _custom_entry.pack(fill="x")
        ToolTip(_custom_entry, "Full Python import path, e.g. prodigyplus.prodigy_plus_schedulefree.ProdigyPlusScheduleFree")
        def _on_opt_combo_change(*_):
            sel = _opt_combo_var.get()
            if sel == "Custom...":
                _custom_frame.pack(fill="x", pady=(2, 4))
                _opt_var.set(_custom_entry.get())
            else:
                _custom_frame.forget()
                _opt_var.set(sel)
        def _on_custom_entry_change(*_):
            if _opt_combo_var.get() == "Custom...":
                _opt_var.set(_custom_entry.get())
        _opt_combo.bind("<<ComboboxSelected>>", _on_opt_combo_change)
        _custom_entry.bind("<KeyRelease>", _on_custom_entry_change)
        # restore helper: called by set_values via StringVar — we also need to sync combo
        def _opt_var_trace(*_):
            v = _opt_var.get()
            if v in _opt_presets and v != "Custom...":
                _opt_combo_var.set(v); _custom_frame.forget()
            elif v not in _opt_presets or v == "Custom...":
                _opt_combo_var.set("Custom...")
                _custom_frame.pack(fill="x", pady=(2, 4))
                _custom_entry.delete(0, tk.END); _custom_entry.insert(0, v)
        _opt_var.trace_add("write", _opt_var_trace)
        # --- ADDED --- max_grad_norm widget
        self._add_widget(optimizer_frame, "max_grad_norm", "Max Grad Norm:", "Clips the gradient norm to prevent gradients from exploding, which can stabilize training. '1.0' is a good default. '0' disables it.", validate_num=True)
        self._add_widget(optimizer_frame, "optimizer_args", "Optimizer Args:", (
            "Additional arguments for the optimizer. Separate multiple args with space, comma, or semicolon.\n\n"
            "ProdigyPlusScheduleFree recommended args:\n"
            "  weight_decay=0.01\n"
            "  d_coef=1.0          (tuning knob 0.1-2.0; lower=slower, higher=aggressive)\n"
            "  use_stableadamw=True\n"
            "  use_speed=True\n"
            "  betas=(0.95,0.99)   (optional, default is (0.9,0.99); more stable)\n\n"
            "Example: weight_decay=0.01, d_coef=1.0, use_stableadamw=True, use_speed=True\n\n"
            "Note: do NOT put spaces around '=' signs."
        ), kind='entry')
        
        lr_frame = ttk.LabelFrame(frame, text="Learning Rate Scheduler"); lr_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(lr_frame, "lr_scheduler", "LR Scheduler:", "Algorithm to adjust learning rate during training. 'cosine' is a reliable choice.", kind='combobox', options=["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup"], command=self.update_button_states)
        self.hidden_frames['lr_warmup'] = ttk.Frame(lr_frame)
        self._add_widget(self.hidden_frames['lr_warmup'], "lr_warmup_steps", "Warmup Steps:", "Number of initial steps where the learning rate gradually increases. Can be a fixed number or a ratio (e.g., 0.1 for 10% of total steps).", validate_num=True)
        self.hidden_frames['lr_restarts'] = ttk.Frame(lr_frame)
        self._add_widget(self.hidden_frames['lr_restarts'], "lr_scheduler_num_cycles", "Restart Cycles:", "Number of times the learning rate will be reset for the 'cosine_with_restarts' scheduler.", validate_num=True)
        self._add_widget(lr_frame, "lr_scheduler_power", "LR Scheduler Power:", "The exponent for the polynomial decay. Only used by the 'polynomial' scheduler.", validate_num=True)
        self._add_widget(lr_frame, "lr_scheduler_min_lr_ratio", "Min LR Ratio:", "The minimum learning rate as a ratio of the initial learning rate.", validate_num=True)

    def create_advanced_tab(self):
        frame = ttk.Frame(self.notebook); self.notebook.add(frame, text="Advanced Settings")
        memory_frame = ttk.LabelFrame(frame, text="Memory & Performance"); memory_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(memory_frame, "mixed_precision", "Mixed Precision:", "Use 'fp16' or 'bf16' to reduce VRAM usage and speed up training. 'fp16' is common, 'bf16' is better on newer GPUs.", kind='combobox', options=["no", "fp16", "bf16"])
        self._add_widget(memory_frame, "gradient_checkpointing", "Gradient Checkpointing", "Drastically reduces VRAM usage by re-calculating gradients on the backward pass. Highly recommended.", kind='checkbox', default_val=True)
        self._add_widget(memory_frame, "persistent_data_loader_workers", "Persistent Data Loader Workers", "Keeps data loader processes alive between epochs to speed up data loading, at the cost of slightly higher RAM usage.", kind='checkbox')
        self._add_widget(memory_frame, "gradient_accumulation_steps", "Gradient Accumulation Steps:", "Simulates a larger batch size by accumulating gradients over several steps. E.g., a batch size of 1 with 4 accumulation steps simulates a batch size of 4.", validate_num=True)
        self._add_widget(memory_frame, "max_data_loader_n_workers", "Max Data Loader Workers:", "Number of CPU threads to load data. '2' is a safe default. Higher values can speed up loading but use more RAM.", validate_num=True)
        self._add_widget(memory_frame, "offload_inactive_dit", "Offload Inactive DiT Model", "When training both models in a combined run, offloads the inactive DiT model to CPU to save VRAM. Disables 'Blocks to Swap'.", kind='checkbox', command=self.update_button_states)
        self._add_widget(memory_frame, "blocks_to_swap", "Blocks to Swap:", "Number of DiT blocks to offload to CPU memory to save VRAM. Can slow down training. (e.g., 10)", validate_num=True)
        
        flow_frame = ttk.LabelFrame(frame, text="Flow Matching Parameters"); flow_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(flow_frame, "timestep_sampling", "Timestep Sampling:", "Method for selecting timesteps during training. 'shift' is recommended for Wan/Flux. 'krea2_shift' matches Krea 2's resolution-aware schedule.", kind='combobox', options=["uniform", "shift", "sigma", "logsnr", "qinglong_flux", "krea2_shift"])
        self._add_widget(flow_frame, "num_timestep_buckets", "Timestep Buckets:", "Enables stratified sampling by dividing timesteps into buckets. Can improve training stability, especially with small datasets. (e.g., 10)", validate_num=True)
        self.hidden_frames['timestep_boundary'] = ttk.Frame(flow_frame)
        self._add_widget(self.hidden_frames['timestep_boundary'], "timestep_boundary", "Timestep Boundary:", "The integer timestep where the model switches from low to high noise (e.g., 875). Only for combined runs.", validate_num=True)
        self._add_widget(flow_frame, "discrete_flow_shift", "Discrete Flow Shift:", "Shift value for 'shift' sampling. The documentation recommends 3.0.", validate_num=True)
        self._add_widget(flow_frame, "preserve_distribution_shape", "Preserve Distribution Shape", "Prevents distortion of the timestep distribution. Recommended when training only one model (e.g., only low noise).", kind='checkbox')
        
        attention_frame = ttk.LabelFrame(frame, text="Attention Mechanism"); attention_frame.pack(fill="x", padx=10, pady=10)
        self.attention_var = tk.StringVar(value="xformers")
        self.entries['attention_mechanism'] = self.attention_var
        attention_options = [("None", "none"), ("xFormers", "xformers"), ("Flash Attention", "flash_attn"), ("SDPA", "sdpa")]
        for text, value in attention_options:
            rb = ttk.Radiobutton(attention_frame, text=text, variable=self.attention_var, value=value)
            rb.pack(anchor="w", padx=5, pady=2); ToolTip(rb, f"Optimized attention mechanism to save VRAM and increase speed. xFormers or Flash Attention are recommended if available.")

        logging_frame = ttk.LabelFrame(frame, text="Logging (TensorBoard / W&B)"); logging_frame.pack(fill="x", padx=10, pady=10)
        log_with_widget = self._add_widget(logging_frame, "log_with", "Log With:", "Enable logging with TensorBoard or Weights & Biases to monitor training progress.", kind='combobox', options=["none", "tensorboard", "wandb", "all"])
        log_with_widget.bind('<<ComboboxSelected>>', self.update_button_states)
        self._add_widget(logging_frame, "logging_dir", "Logging Directory:", "Directory to save logs. Required if 'Log With' is not 'none'.", kind='path_entry', is_dir=True, is_path=True)
        self._add_widget(logging_frame, "log_prefix", "Log Prefix:", "Optional prefix for log filenames or wandb run names.", kind='entry')

        other_frame = ttk.LabelFrame(frame, text="Other Options"); other_frame.pack(fill="x", padx=10, pady=10)
        fp8_frame = ttk.Frame(other_frame); fp8_frame.pack(fill='x')
        self._add_widget(fp8_frame, "fp8_base", "FP8 Base", "Use FP8 precision for the base model. Select a compatible mixed precision (fp16 or bf16).", kind='checkbox')
        self._add_widget(fp8_frame, "fp8_scaled", "FP8 Scaled", "Use scaled FP8 training with block-wise quantization for better accuracy.", kind='checkbox')
        self.hidden_frames['fp8_t5_frame'] = ttk.Frame(fp8_frame)
        self.hidden_frames['fp8_t5_frame'].pack(fill='x')
        self._add_widget(self.hidden_frames['fp8_t5_frame'], "fp8_t5", "FP8 T5", "Use FP8 precision for the T5 text encoder.", kind='checkbox')
        self._add_widget(fp8_frame, "fp8_llm", "FP8 LLM", "Use FP8 precision for LLM components. Only helps if NOT using cached text encoder outputs (rarely useful for WAN training).", kind='checkbox')

        # WAN 2.2 specific options
        wan22_frame = ttk.Frame(other_frame); wan22_frame.pack(fill='x')
        self._add_widget(wan22_frame, "force_v2_1_time_embedding", "Force v2.1 Time Embedding", "Use Wan2.1 time embedding format for Wan2.2 (reduces VRAM usage).", kind='checkbox')

        self._add_widget(other_frame, "save_state", "Save State", "Save the complete training state (optimizer, etc.) to allow resuming later.", kind='checkbox', default_val=True)
        
        resume_frame = ttk.LabelFrame(frame, text="Resume Training"); resume_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(resume_frame, "resume_path", "Resume from State:", "Path to a saved state folder to continue a previous training run.", kind='path_entry', is_dir=True, is_path=True)
        self._add_widget(resume_frame, "network_weights", "Network Weights:", "Load pre-trained LoRA weights to continue training from them (fine-tuning a LoRA).", kind='path_entry', options=[("Weight files", "*.safetensors")], is_path=True)

    def create_samples_tab(self):
        tab_frame = ttk.Frame(self.notebook); self.notebook.add(tab_frame, text="Samples")

        # --- Frequency controls ---
        freq_frame = ttk.LabelFrame(tab_frame, text="Sampling Frequency"); freq_frame.pack(fill="x", padx=10, pady=10)
        freq_inner = ttk.Frame(freq_frame); freq_inner.pack(fill="x", padx=8, pady=8)
        vcmd = (self.root.register(self.validate_number), '%P')

        ttk.Label(freq_inner, text="Every N Epochs:").pack(side="left", padx=(0, 4))
        ep_entry = ttk.Entry(freq_inner, width=7, validate="key", validatecommand=vcmd)
        ep_entry.pack(side="left", padx=(0, 18))
        ep_entry.bind("<FocusOut>", self.update_button_states); ep_entry.bind("<KeyRelease>", self.update_button_states)
        ep_entry.is_required = False; ep_entry.is_path = False
        self.entries["sample_every_n_epochs"] = ep_entry
        ToolTip(ep_entry, "Generate a sample every N training epochs.")

        ttk.Label(freq_inner, text="Every N Steps:").pack(side="left", padx=(0, 4))
        st_entry = ttk.Entry(freq_inner, width=7, validate="key", validatecommand=vcmd)
        st_entry.pack(side="left", padx=(0, 18))
        st_entry.bind("<FocusOut>", self.update_button_states); st_entry.bind("<KeyRelease>", self.update_button_states)
        st_entry.is_required = False; st_entry.is_path = False
        self.entries["sample_every_n_steps"] = st_entry
        ToolTip(st_entry, "Generate a sample every N training steps.")

        af_var = tk.BooleanVar(value=False)
        af_cb = ttk.Checkbutton(freq_inner, text="Sample at First", variable=af_var,
                                command=self.update_button_states)
        af_cb.var = af_var; af_cb.pack(side="left")
        af_cb.is_required = False; af_cb.is_path = False
        self.entries["sample_at_first"] = af_cb
        ToolTip(af_cb, "Generate one sample before training starts (step 0).")

        # --- Prompt editor ---
        prompts_frame = ttk.LabelFrame(tab_frame, text="Sample Prompts"); prompts_frame.pack(fill="x", padx=10, pady=(0, 10))

        prompt_btn_row = ttk.Frame(prompts_frame); prompt_btn_row.pack(fill="x", padx=5, pady=(6, 4))
        ttk.Button(prompt_btn_row, text="+ Add Prompt", command=self._add_sample_prompt_dialog).pack(side="left")
        ttk.Button(prompt_btn_row, text="Enable All", command=lambda: self._set_all_sample_prompts_enabled(True)).pack(side="left", padx=(6, 0))
        ttk.Button(prompt_btn_row, text="Disable All", command=lambda: self._set_all_sample_prompts_enabled(False)).pack(side="left", padx=(6, 0))
        ttk.Label(prompt_btn_row, text="Unchecked prompts stay saved but are skipped during sampling.",
                  foreground="#888888", font=("Calibri", 9, "italic")).pack(side="left", padx=(12, 0))

        plist_container = ttk.Frame(prompts_frame); plist_container.pack(fill="x", padx=5, pady=(0, 6))
        plist_canvas = tk.Canvas(plist_container, bg='#2B2B2B', highlightthickness=0, height=160)
        plist_sb = ttk.Scrollbar(plist_container, orient="vertical", command=plist_canvas.yview)
        self._prompt_list_inner = ttk.Frame(plist_canvas)
        self._prompt_list_inner.bind("<Configure>", lambda e: plist_canvas.configure(scrollregion=plist_canvas.bbox("all")))
        plist_canvas.create_window((0, 0), window=self._prompt_list_inner, anchor="nw", tags="pframe")
        plist_canvas.bind("<Configure>", lambda e: plist_canvas.itemconfig("pframe", width=e.width))
        plist_canvas.configure(yscrollcommand=plist_sb.set)
        plist_canvas.pack(side="left", fill="x", expand=True); plist_sb.pack(side="right", fill="y")
        self._prompt_list_canvas = plist_canvas
        self._rebuild_prompt_list()

        # --- Outputs section ---
        outputs_frame = ttk.LabelFrame(tab_frame, text="Sample Outputs"); outputs_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        btn_row = ttk.Frame(outputs_frame); btn_row.pack(fill="x", padx=5, pady=5)
        ttk.Button(btn_row, text="Open Output Folder", command=self._open_output_folder).pack(side="left", padx=(0, 5))
        ttk.Button(btn_row, text="Refresh", command=self._refresh_sample_list).pack(side="right")

        list_container = ttk.Frame(outputs_frame); list_container.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        canvas = tk.Canvas(list_container, bg='#2B2B2B', highlightthickness=0, height=200)
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=canvas.yview)
        self._sample_list_frame = ttk.Frame(canvas)
        self._sample_list_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._sample_list_frame, anchor="nw", tags="sframe")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig("sframe", width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")
        self._sample_canvas = canvas

        ttk.Label(self._sample_list_frame, text="No samples yet. Start training to generate samples.",
                  foreground="#777777").pack(padx=10, pady=10)

    # ---------- Prompt list helpers ----------

    def _rebuild_prompt_list(self):
        for w in self._prompt_list_inner.winfo_children():
            w.destroy()
        if not self._sample_prompts_data:
            ttk.Label(self._prompt_list_inner, text="No prompts added yet.",
                      foreground="#777777").pack(padx=10, pady=8)
            return
        for idx, p in enumerate(self._sample_prompts_data):
            row = ttk.Frame(self._prompt_list_inner); row.pack(fill="x", padx=4, pady=2)
            enabled_var = tk.BooleanVar(value=p.get("enabled", True))
            def _toggle_enabled(i=idx, var=enabled_var):
                self._sample_prompts_data[i]["enabled"] = bool(var.get())
                self.update_button_states()
            enabled_cb = ttk.Checkbutton(row, text="", variable=enabled_var, command=_toggle_enabled)
            enabled_cb.pack(side="left", padx=(0, 6))
            # Summary label
            summary = p.get("prompt", "")[:55] + ("…" if len(p.get("prompt", "")) > 55 else "")
            params = []
            if p.get("width") or p.get("height"):
                params.append(f"{p.get('width','?')}×{p.get('height','?')}")
            if self.training_mode_var.get() != "Krea 2" and p.get("frames"): params.append(f"{p['frames']}f")
            if p.get("steps"): params.append(f"s{p['steps']}")
            if p.get("guidance"): params.append(f"g{p['guidance']}")
            if self.training_mode_var.get() == "Krea 2":
                if p.get("mu"): params.append(f"mu={p['mu']}")
                if p.get("y1"): params.append(f"y1={p['y1']}")
                if p.get("y2"): params.append(f"y2={p['y2']}")
            if p.get("seed"): params.append(f"seed={p['seed']}")
            if not p.get("enabled", True): params.append("disabled")
            tag = "  |  " + "  ".join(params) if params else ""
            ttk.Label(row, text=summary + tag, anchor="w",
                      foreground="#CCCCCC" if p.get("enabled", True) else "#777777").pack(side="left", fill="x", expand=True)
            ttk.Button(row, text="Test", width=5,
                       command=lambda i=idx: self._test_sample_prompt(i)).pack(side="right", padx=(3, 0))
            ttk.Button(row, text="Dup", width=4,
                       command=lambda i=idx: self._duplicate_sample_prompt(i)).pack(side="right", padx=(3, 0))
            ttk.Button(row, text="Edit", width=5,
                       command=lambda i=idx: self._edit_sample_prompt_dialog(i)).pack(side="right", padx=(3, 0))
            ttk.Button(row, text="Del", width=4,
                       command=lambda i=idx: self._delete_sample_prompt(i)).pack(side="right", padx=(3, 0))

    def _set_all_sample_prompts_enabled(self, enabled):
        for prompt in self._sample_prompts_data:
            prompt["enabled"] = bool(enabled)
        self._rebuild_prompt_list()
        self.update_button_states()

    def _count_enabled_sample_prompts(self):
        return sum(1 for p in self._sample_prompts_data if p.get("enabled", True))

    def _delete_sample_prompt(self, idx):
        self._sample_prompts_data.pop(idx)
        self._rebuild_prompt_list()
        self.update_button_states()

    def _duplicate_sample_prompt(self, idx):
        duplicated = dict(self._sample_prompts_data[idx])
        duplicated["enabled"] = self._sample_prompts_data[idx].get("enabled", True)
        self._sample_prompts_data.insert(idx + 1, duplicated)
        self._rebuild_prompt_list()
        self.update_button_states()

    def _test_sample_prompt(self, idx):
        if self.current_process:
            messagebox.showwarning("Process Running", "Stop the current process before launching a test sample.")
            return

        settings = self.get_settings()
        mode = settings.get("training_mode", "Wan 2.2")
        if mode != "Krea 2":
            messagebox.showinfo("Not Available", "Sample test generation is currently implemented for Krea 2 only.")
            return

        prompt_data = self._sample_prompts_data[idx]
        try:
            command = self._build_krea2_test_sample_command(settings, prompt_data)
        except ValueError as e:
            messagebox.showerror("Krea 2 Test Sample", str(e))
            return

        self.output_text.delete("1.0", tk.END)
        self.run_status_var.set("🧪 Krea 2 Test Sample")
        self.progress_label_var.set("Running test generation...")
        self.run_process(command, on_complete=self._on_test_sample_complete, output_widget=self.output_text)

    def _build_krea2_test_sample_command(self, settings, prompt_data):
        required = {
            "DiT model": settings.get("krea2_dit_model"),
            "VAE model": settings.get("vae_model"),
            "Text encoder": settings.get("krea2_text_encoder"),
        }
        missing = [name for name, path in required.items() if not path or not os.path.exists(path)]
        if missing:
            raise ValueError("Missing required Krea 2 paths for test sampling:\n- " + "\n- ".join(missing))

        python_executable = sys.executable or "python"
        dit_path = settings.get("krea2_turbo_dit") if settings.get("krea2_turbo_dit") else settings.get("krea2_dit_model")
        if not dit_path or not os.path.exists(dit_path):
            raise ValueError("The selected Krea 2 inference DiT path does not exist.")

        output_root = Path(settings.get("output_dir", "")).expanduser()
        output_name = settings.get("output_name", "").strip() or "krea2_test"
        save_path = output_root / output_name / "sample_test"
        save_path.mkdir(parents=True, exist_ok=True)

        attn_map = {
            "sdpa": "torch",
            "flash_attn": "flash",
            "sage_attn": "sageattn",
            "xformers": "xformers",
        }
        attn_mode = attn_map.get(settings.get("attention_mechanism"), "torch")

        command = [
            python_executable,
            "src/musubi_tuner/krea2_generate_image.py",
            prompt_data.get("prompt", ""),
            "--dit", dit_path,
            "--vae", settings["vae_model"],
            "--text_encoder", settings["krea2_text_encoder"],
            "--save_path", str(save_path),
            "--attn_mode", attn_mode,
        ]

        if settings.get("fp8_scaled"):
            command.append("--fp8_scaled")

        blocks_to_swap = str(settings.get("blocks_to_swap") or "").strip()
        if blocks_to_swap and blocks_to_swap != "0":
            command.extend(["--blocks_to_swap", blocks_to_swap])

        if settings.get("krea2_projector_diff"):
            command.extend(["--projector_diff", settings["krea2_projector_diff"]])
            strength = str(settings.get("krea2_projector_diff_strength") or "").strip()
            if strength:
                command.extend(["--projector_diff_strength", strength])

        network_weights = str(settings.get("network_weights") or "").strip()
        if network_weights and os.path.exists(network_weights):
            command.extend(["--lora_weight", network_weights])

        negative_prompt = prompt_data.get("neg", "").strip()
        if negative_prompt:
            command.extend(["--negative_prompt", negative_prompt])

        width = str(prompt_data.get("width", "")).strip()
        if width:
            command.extend(["--width", width])

        height = str(prompt_data.get("height", "")).strip()
        if height:
            command.extend(["--height", height])

        steps = str(prompt_data.get("steps", "")).strip()
        if steps:
            command.extend(["--steps", steps])

        guidance = str(prompt_data.get("guidance", "")).strip()
        if guidance:
            command.extend(["--guidance_scale", guidance])

        seed = str(prompt_data.get("seed", "")).strip()
        if seed:
            command.extend(["--seed", seed])

        mu = str(prompt_data.get("mu", "")).strip()
        y1 = str(prompt_data.get("y1", "")).strip()
        y2 = str(prompt_data.get("y2", "")).strip()
        if mu:
            command.extend(["--mu", mu])
        if y1:
            command.extend(["--y1", y1])
        if y2:
            command.extend(["--y2", y2])

        return command

    def _on_test_sample_complete(self, return_code):
        if return_code == 0:
            self.output_text.insert(tk.END, "\n--- Test sample completed successfully. ---\n")
            self._refresh_sample_list()
        else:
            self.output_text.insert(tk.END, f"\n--- Test sample failed with code {return_code}. ---\n")
        self.stop_all_activity()

    def _add_sample_prompt_dialog(self):
        self._open_prompt_dialog(None)

    def _edit_sample_prompt_dialog(self, idx):
        self._open_prompt_dialog(idx)

    def _open_prompt_dialog(self, idx):
        """Open a modal dialog to add/edit a sample prompt."""
        existing = self._sample_prompts_data[idx] if idx is not None else {}
        mode = self.training_mode_var.get()
        is_krea2 = mode == "Krea 2"
        field_tooltips = {
            "guidance": "Classifier-free guidance scale. For Krea 2 RAW, leaving it empty uses the default 5.5. For Turbo previews, 1.0 is usually the safer value.",
            "mu": "Direct timestep-shift value. If you set Mu, it overrides Y1 and Y2 for this prompt.",
            "y1": "Minimum-resolution timestep-shift endpoint. Used only when Mu is empty.",
            "y2": "Maximum-resolution timestep-shift endpoint. Used only when Mu is empty. Krea 2 defaults to Y1=0.5 and Y2=1.15.",
            "steps": "Number of denoising steps. Krea 2 RAW commonly uses around 28. Turbo previews commonly use around 8.",
        }
        dlg = tk.Toplevel(self.root)
        dlg.title("Edit Sample Prompt" if idx is not None else "Add Sample Prompt")
        dlg.geometry("620x520" if is_krea2 else "560x480")
        dlg.configure(bg='#2B2B2B')
        dlg.resizable(False, False)
        dlg.grab_set()

        def lbl(parent, text):
            ttk.Label(parent, text=text).pack(anchor="w", padx=10, pady=(8, 1))

        def entry(parent, default="", width=None):
            kw = {"width": width} if width else {}
            e = ttk.Entry(parent, **kw)
            if default: e.insert(0, str(default))
            e.pack(fill="x", padx=10, pady=(0, 2))
            return e

        lbl(dlg, "Prompt text *")
        e_prompt = ttk.Entry(dlg)
        e_prompt.insert(0, existing.get("prompt", ""))
        e_prompt.pack(fill="x", padx=10, pady=(0, 2))

        lbl(dlg, "Negative prompt  (optional)")
        e_neg = ttk.Entry(dlg)
        e_neg.insert(0, existing.get("neg", ""))
        e_neg.pack(fill="x", padx=10, pady=(0, 2))

        row1 = ttk.Frame(dlg); row1.pack(fill="x", padx=10, pady=(8, 0))
        row1_fields = [
            ("Width", "width", "1024" if is_krea2 else "512", 6),
            ("Height", "height", "1024" if is_krea2 else "512", 6),
            ("Steps", "steps", "28" if is_krea2 else "20", 5),
            ("Guidance", "guidance", "5.5" if is_krea2 else "5.0", 6),
        ]
        if not is_krea2:
            row1_fields.insert(2, ("Frames", "frames", "25", 5))
        for text, key, default, w in row1_fields:
            col = ttk.Frame(row1); col.pack(side="left", padx=(0, 12))
            label_widget = ttk.Label(col, text=text)
            label_widget.pack(anchor="w")
            e = ttk.Entry(col, width=w)
            e.insert(0, str(existing.get(key, default)))
            e.pack()
            if is_krea2 and key in field_tooltips:
                ToolTip(label_widget, field_tooltips[key])
                ToolTip(e, field_tooltips[key])
            row1.__dict__[f"e_{key}"] = e

        row2 = ttk.Frame(dlg); row2.pack(fill="x", padx=10, pady=(8, 0))
        if is_krea2:
            row2_fields = [
                ("Mu", "mu", existing.get("mu", existing.get("flow_shift", "")), 6),
                ("Y1", "y1", existing.get("y1", ""), 6),
                ("Y2", "y2", existing.get("y2", ""), 6),
                ("Seed", "seed", existing.get("seed", ""), 8),
            ]
        else:
            row2_fields = [
                ("Flow Shift", "flow_shift", existing.get("flow_shift", ""), 6),
                ("CFG Scale", "cfg_scale", existing.get("cfg_scale", ""), 6),
                ("Seed", "seed", existing.get("seed", ""), 8),
            ]
        for text, key, default, w in row2_fields:
            col = ttk.Frame(row2); col.pack(side="left", padx=(0, 12))
            label_widget = ttk.Label(col, text=text)
            label_widget.pack(anchor="w")
            e = ttk.Entry(col, width=w)
            val = default
            if val: e.insert(0, str(val))
            e.pack()
            if is_krea2 and key in field_tooltips:
                ToolTip(label_widget, field_tooltips[key])
                ToolTip(e, field_tooltips[key])
            row2.__dict__[f"e_{key}"] = e

        e_img_entry = None
        if is_krea2:
            ttk.Label(
                dlg,
                text="Krea 2 prompt notes: leave Frames empty; use Guidance only; Mu/Y1/Y2 are optional timestep-shift controls.",
                foreground="#888888",
                font=("Calibri", 9, "italic"),
                wraplength=580,
            ).pack(anchor="w", padx=10, pady=(12, 0))
        else:
            lbl(dlg, "Image path  (I2V only — optional)")
            e_img = ttk.Frame(dlg); e_img.pack(fill="x", padx=10, pady=(0, 2))
            e_img_entry = ttk.Entry(e_img)
            e_img_entry.insert(0, existing.get("image_path", ""))
            e_img_entry.pack(side="left", fill="x", expand=True)
            def _browse_img():
                p = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")])
                if p: e_img_entry.delete(0, tk.END); e_img_entry.insert(0, p)
            ttk.Button(e_img, text="Browse", command=_browse_img).pack(side="right", padx=(5, 0))

        def _save():
            prompt_text = e_prompt.get().strip()
            if not prompt_text:
                messagebox.showerror("Validation", "Prompt text cannot be empty.", parent=dlg); return
            data = {"prompt": prompt_text}
            data["neg"]        = e_neg.get().strip()
            data["width"]      = row1.e_width.get().strip()
            data["height"]     = row1.e_height.get().strip()
            data["steps"]      = row1.e_steps.get().strip()
            data["guidance"]   = row1.e_guidance.get().strip()
            if not is_krea2:
                data["frames"]     = row1.e_frames.get().strip()
                data["flow_shift"] = row2.e_flow_shift.get().strip()
                data["cfg_scale"]  = row2.e_cfg_scale.get().strip()
            else:
                data["mu"]         = row2.e_mu.get().strip()
                data["y1"]         = row2.e_y1.get().strip()
                data["y2"]         = row2.e_y2.get().strip()
            data["seed"]       = row2.e_seed.get().strip()
            data["image_path"] = e_img_entry.get().strip() if e_img_entry is not None else ""
            data["enabled"]    = existing.get("enabled", True)
            # Remove empty optional keys
            data = {k: v for k, v in data.items() if v != "" or k == "enabled"}
            if idx is not None:
                self._sample_prompts_data[idx] = data
            else:
                self._sample_prompts_data.append(data)
            self._rebuild_prompt_list()
            self.update_button_states()
            dlg.destroy()

        btn_row = ttk.Frame(dlg); btn_row.pack(pady=14)
        ttk.Button(btn_row, text="Save", command=_save).pack(side="left", padx=6)
        ttk.Button(btn_row, text="Cancel", command=dlg.destroy).pack(side="left", padx=6)
        dlg.bind("<Return>", lambda e: _save())
        e_prompt.focus_set()

    def _build_sample_prompts_txt(self):
        """Serialise self._sample_prompts_data to a .txt file next to the output dir, return path or ''."""
        if not self._sample_prompts_data:
            return ""
        mode = self.training_mode_var.get()
        is_krea2 = mode == "Krea 2"
        lines = []
        for p in self._sample_prompts_data:
            if not p.get("enabled", True):
                continue
            line = p.get("prompt", "")
            if p.get("width"):      line += f" --w {p['width']}"
            if p.get("height"):     line += f" --h {p['height']}"
            if p.get("steps"):      line += f" --s {p['steps']}"
            if p.get("guidance"):   line += f" --g {p['guidance']}"
            if is_krea2:
                if p.get("mu"):     line += f" --mu {p['mu']}"
                if p.get("y1"):     line += f" --y1 {p['y1']}"
                if p.get("y2"):     line += f" --y2 {p['y2']}"
            else:
                if p.get("frames"):     line += f" --f {p['frames']}"
                if p.get("flow_shift"): line += f" --fs {p['flow_shift']}"
                if p.get("cfg_scale"):  line += f" --l {p['cfg_scale']}"
            if p.get("seed"):       line += f" --d {p['seed']}"
            if p.get("neg"):        line += f" --n {p['neg']}"
            if not is_krea2 and p.get("image_path"): line += f" --i {p['image_path']}"
            lines.append(line)
        if not lines:
            return ""
        # Save next to the dataset config (always outside the repo, always exists)
        output_name = self.entries["output_name"].get().strip() or "training"
        dataset_config = self.entries["dataset_config"].get().strip()
        if dataset_config and os.path.isfile(dataset_config):
            base_dir = os.path.dirname(dataset_config)
        else:
            import tempfile
            base_dir = tempfile.gettempdir()
        save_path = os.path.join(base_dir, f"{output_name}_sample_prompts.txt")
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self._temp_prompts_file = save_path
            print(f"[Samples] Wrote prompts file: {save_path}")
            return save_path
        except Exception as e:
            messagebox.showerror("Sample Prompts Error", f"Could not write sample prompts file:\n{save_path}\n\n{e}")
            return ""

    def _open_output_folder(self):
        output_dir = self.entries["output_dir"].get()
        if not output_dir or not os.path.isdir(output_dir):
            messagebox.showinfo("Output Folder", "Output directory is not set or does not exist."); return
        if sys.platform == "win32":
            os.startfile(output_dir)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", output_dir])
        else:
            subprocess.Popen(["xdg-open", output_dir])

    def _get_sample_files(self):
        output_dir = self.entries["output_dir"].get()
        if not output_dir or not os.path.isdir(output_dir):
            return []
        results = []
        for root_dir, _, files in os.walk(output_dir):
            for fname in files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.mp4', '.webp')):
                    fpath = os.path.join(root_dir, fname)
                    try:
                        results.append((os.path.getmtime(fpath), fpath))
                    except OSError:
                        pass
        results.sort(key=lambda x: x[0])
        return results

    def _refresh_sample_list(self):
        files = self._get_sample_files()
        if files == self._last_sample_files:
            return
        self._last_sample_files = files

        for w in self._sample_list_frame.winfo_children():
            w.destroy()

        if not files:
            ttk.Label(self._sample_list_frame, text="No samples yet. Start training to generate samples.",
                      foreground="#777777").pack(padx=10, pady=10)
            return

        for mtime, fpath in reversed(files):
            row = ttk.Frame(self._sample_list_frame); row.pack(fill="x", padx=5, pady=1)
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
            ttk.Label(row, text=ts, width=17, anchor="w").pack(side="left")
            short = Path(fpath).name
            if len(short) > 50:
                short = short[:22] + "..." + short[-22:]
            ttk.Label(row, text=short, anchor="w").pack(side="left", fill="x", expand=True)
            def _open(p=fpath):
                if sys.platform == "win32":
                    os.startfile(p)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", p])
                else:
                    subprocess.Popen(["xdg-open", p])
            ttk.Button(row, text="Open", width=6, command=_open).pack(side="right", padx=(5, 0))

    def _start_sample_watcher(self):
        if self._count_enabled_sample_prompts() == 0:
            return
        self._last_sample_files = []
        self.sample_watcher_active = True

        def _watch():
            while self.sample_watcher_active:
                new_files = self._get_sample_files()
                if new_files != self._last_sample_files:
                    self.root.after(0, self._refresh_sample_list)
                time.sleep(3)

        self._sample_watcher_thread = threading.Thread(target=_watch, daemon=True)
        self._sample_watcher_thread.start()

    def _stop_sample_watcher(self):
        self.sample_watcher_active = False

    def create_run_monitor_tab(self):
        tab_frame = ttk.Frame(self.notebook); self.notebook.add(tab_frame, text="Run & Monitor")
        top_pane = ttk.Frame(tab_frame); top_pane.pack(fill='x', padx=10, pady=10)
        controls_frame = ttk.LabelFrame(top_pane, text="Controls & Caching"); controls_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        self.run_status_var = tk.StringVar(value="⚪ New Training RUN")
        self.run_status_label = ttk.Label(controls_frame, textvariable=self.run_status_var, style='Status.TLabel')
        self.run_status_label.pack(pady=5, padx=10)
        cache_opts_frame = ttk.Frame(controls_frame)
        cache_opts_frame.pack(pady=5, padx=10, fill='x')
        self._add_widget(cache_opts_frame, "recache_latents", "Re-cache Latents Before Training", "If your dataset or VAE changes, check this to force regeneration of the latent cache.", kind='checkbox')
        self._add_widget(cache_opts_frame, "recache_text", "Re-cache Text Encoders Before Training", "If your dataset or T5 model changes, check this to force regeneration of the text encoder cache.", kind='checkbox')
        train_button_frame = ttk.Frame(controls_frame); train_button_frame.pack(pady=10, padx=10, fill='x')
        self.start_btn = ttk.Button(train_button_frame, text="Start Training", command=self.start_training); self.start_btn.pack(side="left", padx=(0, 5), expand=True, fill='x')
        self.stop_btn = ttk.Button(train_button_frame, text="Stop Training", command=self.stop_training, state="disabled"); self.stop_btn.pack(side="left", padx=5, expand=True, fill='x')
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(controls_frame, variable=self.progress_var, style='TProgressbar'); self.progress_bar.pack(pady=(5, 5), padx=10, fill='x')
        self.progress_label_var = tk.StringVar(value="Ready"); ttk.Label(controls_frame, textvariable=self.progress_label_var, anchor='center').pack(fill='x')
        monitor_frame = ttk.LabelFrame(top_pane, text="Live Monitoring"); monitor_frame.pack(side='left', fill='both', expand=True)
        self.vram_label_var = tk.StringVar(value="VRAM: N/A"); ttk.Label(monitor_frame, textvariable=self.vram_label_var).pack(anchor='w', padx=10, pady=5)
        self.peak_vram_label_var = tk.StringVar(value="Peak VRAM: N/A"); ttk.Label(monitor_frame, textvariable=self.peak_vram_label_var).pack(anchor='w', padx=10)
        self.epoch_counter_var = tk.StringVar(value="Epoch: N/A")
        ttk.Label(monitor_frame, textvariable=self.epoch_counter_var).pack(anchor='w', padx=10, pady=(6, 0))
        self.step_counter_var = tk.StringVar(value="Step: N/A")
        ttk.Label(monitor_frame, textvariable=self.step_counter_var).pack(anchor='w', padx=10, pady=(2, 0))
        self.next_epoch_var = tk.StringVar(value="To next epoch: N/A")
        ttk.Label(monitor_frame, textvariable=self.next_epoch_var).pack(anchor='w', padx=10, pady=(2, 0))
        ttk.Button(monitor_frame, text="Generate Command", command=self.show_command).pack(pady=(10,5), padx=10, fill='x')

        bottom_pane_host = ttk.Frame(tab_frame, height=320)
        bottom_pane_host.pack(fill='x', expand=False, padx=10, pady=10)
        bottom_pane_host.pack_propagate(False)

        bottom_pane = ttk.PanedWindow(bottom_pane_host, orient=tk.HORIZONTAL)
        bottom_pane.pack(fill='both', expand=True)
        graph_frame = ttk.LabelFrame(bottom_pane, text="Live Loss"); bottom_pane.add(graph_frame, weight=1)
        if MATPLOTLIB_AVAILABLE:
            self.fig = Figure(figsize=(5, 2.8), dpi=100); self.ax = self.fig.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame); self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.setup_graph_style()
        else: ttk.Label(graph_frame, text="Matplotlib not found.\nInstall with 'pip install matplotlib'", wraplength=200, justify='center').pack(expand=True)
        console_frame = ttk.LabelFrame(bottom_pane, text="Console Output"); bottom_pane.add(console_frame, weight=1)
        self.output_text = tk.Text(console_frame, wrap=tk.WORD, height=14, bg='#3C3F41', fg='#D3D3D3', insertbackground='#D3D3D3', font=('Consolas', 9), relief=tk.FLAT, bd=0)
        output_scrollbar = ttk.Scrollbar(console_frame, orient="vertical", command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=output_scrollbar.set); self.output_text.pack(side="left", fill="both", expand=True); output_scrollbar.pack(side="right", fill="y")

    def create_convert_lora_tab(self):
        tab_frame = ttk.Frame(self.notebook); self.notebook.add(tab_frame, text="Convert LoRA")
        main_frame = ttk.Frame(tab_frame); main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        info_frame = ttk.LabelFrame(main_frame, text="Format Reference"); info_frame.pack(fill='x', pady=(0, 10))
        info_text = tk.Text(info_frame, wrap=tk.WORD, bg='#2B2B2B', fg='#AAAAAA', font=('Consolas', 9),
                            relief=tk.FLAT, bd=0, height=7, state='normal', cursor='arrow')
        info_text.insert(tk.END,
            "musubi-tuner format  (ComfyUI-compatible, trained by this tool)\n"
            "  Keys: lora_unet_...lora_down.weight / lora_up.weight\n"
            "  Alpha: lora_unet_...alpha\n\n"
            "Diffusers format  (HuggingFace Diffusers-based tools)\n"
            "  Keys: diffusion_model....lora_A.weight / lora_B.weight\n"
            "  No alpha key (rank is used instead)"
        )
        info_text.config(state='disabled')
        info_text.pack(fill='x', padx=8, pady=(6, 8))

        settings_frame = ttk.LabelFrame(main_frame, text="Conversion Settings"); settings_frame.pack(fill='x', pady=(0,10))
        self._add_widget(settings_frame, "convert_lora_path", "LoRA to Convert:", "Path to the .safetensors LoRA file you want to convert.", kind='path_entry', options=[("Safetensors", "*.safetensors")], is_path=True)
        self._add_widget(settings_frame, "convert_output_dir", "Output Directory:", "Folder to save the converted LoRA file.", kind='path_entry', is_dir=True)

        dir_frame = ttk.Frame(settings_frame); dir_frame.pack(fill='x', padx=5, pady=(5, 0))
        ttk.Label(dir_frame, text="Conversion Direction:").pack(anchor='w')
        self._convert_target_var = tk.StringVar(value="default")
        dir_inner = ttk.Frame(dir_frame); dir_inner.pack(anchor='w', pady=(3, 0))
        ttk.Radiobutton(dir_inner, text="Diffusers → musubi-tuner  (lora_A/B  →  lora_down/up)",
                        variable=self._convert_target_var, value="default").pack(anchor='w')
        ttk.Radiobutton(dir_inner, text="musubi-tuner → Diffusers  (lora_down/up  →  lora_A/B)",
                        variable=self._convert_target_var, value="other").pack(anchor='w')

        button = ttk.Button(settings_frame, text="Start Conversion", command=self.start_conversion); button.pack(pady=10)

        console_frame = ttk.LabelFrame(main_frame, text="Conversion Output"); console_frame.pack(fill='both', expand=True)
        self.convert_output_text = tk.Text(console_frame, wrap=tk.WORD, bg='#3C3F41', fg='#D3D3D3', insertbackground='#D3D3D3', font=('Consolas', 9), relief=tk.FLAT, bd=0)
        scrollbar = ttk.Scrollbar(console_frame, orient="vertical", command=self.convert_output_text.yview)
        self.convert_output_text.configure(yscrollcommand=scrollbar.set); self.convert_output_text.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")
        
    def create_accelerate_config_tab(self):
        tab_frame = ttk.Frame(self.notebook); self.notebook.add(tab_frame, text="Accelerate Config")
        main_frame = ttk.Frame(tab_frame); main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        info_frame = ttk.LabelFrame(main_frame, text="Setup Instructions"); info_frame.pack(fill='x', pady=(0, 10))
        info_text_content = """This needs to be done only once before your first training run.
Click the button below to open a new terminal where you will configure Accelerate. Answer the questions based on your environment. For a standard single GPU setup, use the following answers:

- In which compute environment are you running?: This machine
- Which type of machine are you using?: No distributed training
- Do you want to run your training on CPU only...?: NO
- Do you wish to optimize your script with torch dynamo?: NO
- Do you want to use DeepSpeed? [yes/NO]: NO
- What GPU(s) (by id) should be used for training...?: all
- Would you like to enable numa efficiency...?: NO
- Do you wish to use mixed precision?: bf16 (or fp16)

Note: If you get a 'ValueError: fp16 mixed precision requires a GPU', try answering '0' to the GPU question to explicitly select your first GPU.
"""
        info_text = tk.Text(info_frame, wrap=tk.WORD, bg='#3C3F41', fg='#D3D3D3', font=('Calibri', 10), relief=tk.FLAT, bd=0, height=15)
        info_text.insert(tk.END, info_text_content); info_text.config(state="disabled")
        info_text.pack(fill='x', expand=True, padx=10, pady=10)

        action_frame = ttk.LabelFrame(main_frame, text="Run Configuration"); action_frame.pack(fill='x')
        button = ttk.Button(action_frame, text="Run Accelerate Config", command=self.run_accelerate_config)
        button.pack(pady=20)

    def on_training_mode_change(self, event=None):
        mode = self.training_mode_var.get()
        is_wan = (mode == "Wan 2.2")

        if is_wan:
            self.title_label.config(text="Musubi Tuner - WAN 2.2 LoRA Training")
            self.root.title("Musubi Tuner GUI - WAN 2.2 LoRA Training")
            self.mode_note_label.config(text="")
            self.hidden_frames['flux2_model_paths'].pack_forget()
            self.hidden_frames['krea2_model_paths'].pack_forget()
            self.hidden_frames['wan_dit'].pack(fill="x", padx=10, pady=10, before=self._vae_frame)
            self.hidden_frames['wan_models'].pack(fill="x", padx=10, pady=10, before=self._vae_frame)
        elif mode == "Krea 2":
            self.title_label.config(text="Musubi Tuner - Krea 2 LoRA Training")
            self.root.title("Musubi Tuner GUI - Krea 2 LoRA Training")
            self.mode_note_label.config(text="Single RAW DiT, Qwen3-VL text encoder, Qwen-Image VAE, image-only training")
            self.hidden_frames['wan_dit'].pack_forget()
            self.hidden_frames['wan_models'].pack_forget()
            self.hidden_frames['flux2_model_paths'].pack_forget()
            self.hidden_frames['krea2_model_paths'].pack(fill="x", padx=10, pady=10, before=self._vae_frame)
            if self.entries["timestep_sampling"].get() in ("", "shift"):
                self.entries["timestep_sampling"].set("krea2_shift")
            if self.entries["network_dim_low"].get().strip() == "":
                self.entries["network_dim_low"].insert(0, "32")
            if self.entries["network_alpha_low"].get().strip() in ("", "16"):
                self.entries["network_alpha_low"].delete(0, tk.END)
                self.entries["network_alpha_low"].insert(0, "32")
        else:
            title_txt = "Musubi Tuner - Flux.2 Klein LoRA Training" if mode == "Flux.2 Klein" else "Musubi Tuner - Flux.2 Dev LoRA Training"
            self.title_label.config(text=title_txt)
            self.root.title(f"Musubi Tuner GUI - {mode} LoRA Training")
            self.mode_note_label.config(text="Single DiT, Qwen3/Mistral3 text encoder" if mode == "Flux.2 Klein" else "Single DiT, Mistral3 text encoder")
            self.hidden_frames['wan_dit'].pack_forget()
            self.hidden_frames['wan_models'].pack_forget()
            self.hidden_frames['krea2_model_paths'].pack_forget()
            self.hidden_frames['flux2_model_paths'].pack(fill="x", padx=10, pady=10, before=self._vae_frame)
            # Update version choices
            ver_combo = self.entries["flux2_model_version"]
            if mode == "Flux.2 Dev":
                ver_combo.config(values=["Dev"])
                ver_combo.set("Dev")
            else:
                klein_versions = [k for k in FLUX2_VERSION_MAP if k != "Dev"]
                ver_combo.config(values=klein_versions)
                if ver_combo.get() == "Dev":
                    ver_combo.set("Klein Base 4B ★")

        try:
            self._rebuild_prompt_list()
        except AttributeError:
            pass
        self.update_button_states()

    def setup_graph_style(self):
        self.fig.patch.set_facecolor('#2B2B2B'); self.ax.set_facecolor('#3C3F41')
        self.ax.tick_params(axis='x', colors='white'); self.ax.tick_params(axis='y', colors='white')
        self.ax.spines['bottom'].set_color('white'); self.ax.spines['top'].set_color('white') 
        self.ax.spines['right'].set_color('white'); self.ax.spines['left'].set_color('white')
        self.ax.yaxis.label.set_color('white'); self.ax.xaxis.label.set_color('white')
        self.ax.title.set_color('white'); self.ax.set_xlabel("Steps"); self.ax.set_ylabel("Loss")
        self.canvas.draw()
    
    def validate_number(self, value):
        if value in ("", ".", "-"): return True
        try: float(value); return True
        except ValueError: return False

    def _create_temp_cache_config(self, original_config_path):
        """Create a temporary dataset config for caching that excludes image_directory"""
        import tempfile
        import shutil

        # Create temp file
        temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False)

        # Read original config and filter out image_directory
        with open(original_config_path, 'r') as f:
            content = f.read()

        # Remove image_directory lines
        filtered_lines = []
        for line in content.split('\n'):
            if not line.strip().startswith('image_directory'):
                filtered_lines.append(line)

        # Write filtered content
        temp_config.write('\n'.join(filtered_lines))
        temp_config.close()

        return temp_config.name

    def update_button_states(self, event=None):
        try:
            self._update_dynamic_widgets()
            if self.entries["resume_path"].get(): self.run_status_var.set("🟢 Resuming Training RUN")
            else: self.run_status_var.set("⚪ New Training RUN")
        except (KeyError, AttributeError): pass

        mode = self.training_mode_var.get()
        is_wan = (mode == "Wan 2.2")
        is_flux2 = mode in ("Flux.2 Klein", "Flux.2 Dev")
        is_krea2 = (mode == "Krea 2")
        all_valid = True
        wants_samples = bool(
            self._sample_prompts_data and (
                str(self.entries["sample_every_n_epochs"].get()).strip() or
                str(self.entries["sample_every_n_steps"].get()).strip() or
                self.entries["sample_at_first"].var.get()
            )
        )

        if is_wan:
            train_high = self.entries["train_high_noise"].var.get(); train_low = self.entries["train_low_noise"].var.get()
            self.entries["dit_high_noise"].is_required = train_high; self.entries["dit_low_noise"].is_required = train_low
            self.entries["network_dim_low"].is_required = train_low; self.entries["network_alpha_low"].is_required = train_low
            self.entries["clip_model"].is_required = self.entries["is_i2v"].var.get()
            self.entries["t5_model"].is_required = True
            self.entries["flux2_dit_model"].is_required = False
            self.entries["flux2_text_encoder"].is_required = False
            self.entries["krea2_dit_model"].is_required = False
            self.entries["krea2_text_encoder"].is_required = False
        else:
            # Single-model modes: set their own required fields and clear the rest
            self.entries["flux2_dit_model"].is_required = is_flux2
            self.entries["flux2_text_encoder"].is_required = is_flux2
            self.entries["krea2_dit_model"].is_required = is_krea2
            self.entries["krea2_text_encoder"].is_required = is_krea2 and (self.entries["recache_text"].var.get() or wants_samples)
            self.entries["t5_model"].is_required = False
            self.entries["dit_high_noise"].is_required = False
            self.entries["dit_low_noise"].is_required = False
            self.entries["clip_model"].is_required = False
            self.entries["network_dim_low"].is_required = True
            self.entries["network_alpha_low"].is_required = True
            train_high = False; train_low = True  # for combined-run logic below

        log_with = self.entries["log_with"].get(); self.entries["logging_dir"].is_required = log_with != "none"

        for widget in self.entries.values():
            if not isinstance(widget, tk.Widget): continue
            is_visible = False
            try:
                if widget.winfo_manager(): is_visible = True
            except tk.TclError: is_visible = False
            if not is_visible:
                if isinstance(widget, ttk.Entry): widget.config(style="Valid.TEntry")
                continue
            if isinstance(widget, ttk.Entry):
                is_valid = True
                if getattr(widget, 'is_required', False):
                    value = widget.get()
                    if not value: is_valid = False
                    elif getattr(widget, 'is_path', False) and not os.path.exists(value): is_valid = False
                style = "Valid.TEntry" if is_valid else "Invalid.TEntry"
                widget.config(style=style)
                if not is_valid: all_valid = False

        if is_wan:
            if not (train_high or train_low): all_valid = False
        self.start_btn.config(state="normal" if all_valid else "disabled")
        try:
            can_cache_latents = all(self.entries[key].get() and os.path.exists(self.entries[key].get()) for key in ["dataset_config", "vae_model"])
            self.entries["recache_latents"].config(state="normal" if can_cache_latents else "disabled")
            if is_wan:
                can_cache_text = all(self.entries[key].get() and os.path.exists(self.entries[key].get()) for key in ["dataset_config", "t5_model"])
            elif is_krea2:
                can_cache_text = all(self.entries[key].get() and os.path.exists(self.entries[key].get()) for key in ["dataset_config", "krea2_text_encoder"])
            else:
                can_cache_text = all(self.entries[key].get() and os.path.exists(self.entries[key].get()) for key in ["dataset_config", "flux2_text_encoder"])
            self.entries["recache_text"].config(state="normal" if can_cache_text else "disabled")
        except (AttributeError, KeyError): pass

    def _update_dynamic_widgets(self):
        mode = self.training_mode_var.get()
        is_wan = (mode == "Wan 2.2")
        is_krea2 = (mode == "Krea 2")

        show_low = self.entries["train_low_noise"].var.get() if is_wan else False
        show_high = self.entries["train_high_noise"].var.get() if is_wan else False
        is_i2v = self.entries["is_i2v"].var.get() if is_wan else False

        net_type = self.entries.get("network_type", None)
        is_lokr = net_type and net_type.get() == "LoKr"
        if is_lokr: self.hidden_frames['lokr_factor'].pack(fill='x', pady=(0, 3))
        else: self.hidden_frames['lokr_factor'].pack_forget()

        # Low/high noise lora params only shown in Wan mode
        if is_wan and show_low: self.hidden_frames['low_noise_lora_params'].pack(fill='x', expand=True, pady=(0, 5))
        else: self.hidden_frames['low_noise_lora_params'].pack_forget()
        if is_wan and show_high: self.hidden_frames['high_noise_lora_params'].pack(fill='x', expand=True, pady=(0, 5))
        else: self.hidden_frames['high_noise_lora_params'].pack_forget()

        # In Flux.2 mode always show low_noise_lora_params as the single network params section
        if not is_wan:
            self.hidden_frames['low_noise_lora_params'].config(text="Network Parameters")
            self.hidden_frames['low_noise_lora_params'].pack(fill='x', expand=True, pady=(0, 5))
        else:
            self.hidden_frames['low_noise_lora_params'].config(text="Low Noise Network Parameters")

        dim_high_val = self.entries["network_dim_high"].get().strip()
        alpha_high_val = self.entries["network_alpha_high"].get().strip()
        is_separate_run = (dim_high_val and dim_high_val != "None") or \
                          (alpha_high_val and alpha_high_val != "None")
        is_combined_run = is_wan and show_low and show_high and not is_separate_run

        if is_combined_run:
            self.hidden_frames['timestep_boundary'].pack(fill='x', expand=True)
            boundary_widget = self.entries["timestep_boundary"]
            current_val = boundary_widget.get()
            default_val = "900" if is_i2v else "875"
            if current_val != default_val: boundary_widget.delete(0, tk.END); boundary_widget.insert(0, default_val)
        else:
            self.hidden_frames['timestep_boundary'].pack_forget()

        offload_widget = self.entries["offload_inactive_dit"]
        blocks_to_swap_widget = self.entries["blocks_to_swap"]
        offload_widget.config(state="normal" if is_wan else "disabled")
        is_offloading = is_wan and offload_widget.var.get()
        blocks_to_swap_widget.config(state="disabled" if is_offloading else "normal")
        if is_offloading and blocks_to_swap_widget.cget('state') == 'normal':
            blocks_to_swap_widget.delete(0, tk.END)

        scheduler = self.entries["lr_scheduler"].get()
        if scheduler == "constant_with_warmup": self.hidden_frames['lr_warmup'].pack(fill='x', expand=True)
        else: self.hidden_frames['lr_warmup'].pack_forget()
        if scheduler == "cosine_with_restarts": self.hidden_frames['lr_restarts'].pack(fill='x', expand=True)
        else: self.hidden_frames['lr_restarts'].pack_forget()

        # fp8_t5 only relevant for Wan mode
        try:
            if is_wan: self.hidden_frames['fp8_t5_frame'].pack(fill='x')
            else: self.hidden_frames['fp8_t5_frame'].pack_forget()
        except KeyError: pass

    def get_settings(self):
        settings = {}
        for key, widget in self.entries.items():
            if isinstance(widget, (tk.BooleanVar, tk.StringVar)): settings[key] = widget.get()
            elif hasattr(widget, 'var'): settings[key] = widget.var.get()
            else: settings[key] = widget.get()
        settings["training_mode"] = self.training_mode_var.get()
        settings["sample_prompts_data"] = self._sample_prompts_data
        return settings

    def set_values(self, settings):
        # Restore training mode first so widget visibility is correct
        if "training_mode" in settings:
            self.training_mode_var.set(settings["training_mode"])
            self.on_training_mode_change()
        if "sample_prompts_data" in settings:
            self._sample_prompts_data = settings["sample_prompts_data"] or []
            for prompt in self._sample_prompts_data:
                if "enabled" not in prompt:
                    prompt["enabled"] = True
            try: self._rebuild_prompt_list()
            except AttributeError: pass  # UI not built yet during early init
        for key, value in settings.items():
            if key in ("training_mode", "sample_prompts_data", "sample_prompts"): continue
            if key in self.entries:
                widget = self.entries[key]
                if isinstance(widget, (tk.BooleanVar, tk.StringVar)):
                    widget.set(value if value is not None else "")
                elif hasattr(widget, 'var'):
                    widget.var.set(value if value is not None else False)
                elif isinstance(widget, ttk.Combobox):
                    widget.set(value if value is not None else "")
                elif isinstance(widget, ttk.Entry):
                    widget.delete(0, tk.END)
                    widget.insert(0, str(value) if value is not None else "")
        self.update_button_states()

    def load_default_settings(self):
        defaults = {
            "dataset_config": "", "dit_high_noise": "", "dit_low_noise": "", "is_i2v": False,
            "train_high_noise": True, "train_low_noise": True,
            "min_timestep_low": "0", "max_timestep_low": "875", "min_timestep_high": "875", "max_timestep_high": "1000",
            "vae_model": "", "clip_model": "", "t5_model": "",
            "flux2_model_version": "Klein Base 4B ★", "flux2_dit_model": "", "flux2_text_encoder": "", "fp8_text_encoder": False,
            "krea2_dit_model": "", "krea2_text_encoder": "", "krea2_turbo_dit": "", "krea2_turbo_dit_cache": False,
            "krea2_projector_diff": "", "krea2_projector_diff_strength": "1.0",
            "output_dir": "", "output_name": "my-lora",
            "learning_rate": "2e-4", "max_train_epochs": "10", "save_every_n_epochs": "1", "save_every_n_steps": "", "seed": "42",
            "network_type": "LoRA", "lokr_factor": "", "network_dim_low": "32", "network_alpha_low": "16", "network_dim_high": "", "network_alpha_high": "",
            "optimizer_type": "adamw8bit", "max_grad_norm": "1.0", "optimizer_args": "", "lr_scheduler": "cosine",
            "lr_warmup_steps": "0", "lr_scheduler_num_cycles": "1",
            "mixed_precision": "fp16", "gradient_accumulation_steps": "1",
            "max_data_loader_n_workers": "2", "blocks_to_swap": "10", "timestep_sampling": "shift",
            "num_timestep_buckets": "", "timestep_boundary": "875", "discrete_flow_shift": "3.0", "preserve_distribution_shape": False,
            "gradient_checkpointing": True, "persistent_data_loader_workers": True, "save_state": True,
            "fp8_base": False, "fp8_scaled": False, "fp8_t5": False, "fp8_llm": False, "force_v2_1_time_embedding": False, "offload_inactive_dit": False,
            "attention_mechanism": "xformers", "resume_path": "", "network_weights": "",
            "log_with": "none", "logging_dir": "", "log_prefix": "",
            "recache_latents": False, "recache_text": False,
            "convert_lora_path": "", "convert_output_dir": "",
            "training_mode": "Wan 2.2",
            "sample_every_n_epochs": "", "sample_every_n_steps": "", "sample_at_first": False,
            "sample_prompts_data": [],
        }
        self.set_values(defaults)
        
    def _save_settings_to_file(self, filepath):
        try:
            with open(filepath, "w") as f: json.dump(self.get_settings(), f, indent=4); return True
        except Exception as e: print(f"Error saving settings to {filepath}: {e}"); return False

    def save_settings(self):
        initial_dir = self.entries["output_dir"].get() if self.entries["output_dir"].get() else os.getcwd()
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")], initialdir=initial_dir)
        if file_path and self._save_settings_to_file(file_path): messagebox.showinfo("Success", "Settings saved successfully!")

    def load_settings(self, filepath=None):
        if filepath is None:
            initial_dir = self.entries["output_dir"].get() if self.entries["output_dir"].get() else os.getcwd()
            filepath = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")], initialdir=initial_dir)
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, "r") as f: settings = json.load(f)
                self.set_values(settings)
                if not filepath.endswith("last_settings.json"): messagebox.showinfo("Success", "Settings loaded successfully!")
            except Exception as e: messagebox.showerror("Error", f"Failed to load settings: {e}")

    def _load_last_settings(self): self.load_settings(filepath="last_settings.json")

    def start_vram_monitor(self):
        if not PYNVML_AVAILABLE: self.vram_label_var.set("VRAM: pynvml not installed"); return
        try:
            pynvml.nvmlInit(); self.monitoring_active = True; self.peak_vram = 0
            self.vram_thread = threading.Thread(target=self.vram_monitor_loop, daemon=True); self.vram_thread.start()
        except pynvml.NVMLError: self.vram_label_var.set(f"VRAM: NVML Error")

    def stop_vram_monitor(self):
        self.monitoring_active = False
        if PYNVML_AVAILABLE:
            try: pynvml.nvmlShutdown()
            except pynvml.NVMLError: pass

    def vram_monitor_loop(self):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            while self.monitoring_active:
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_gb = info.used / (1024**3)
                if used_gb > self.peak_vram: self.peak_vram = used_gb
                self.root.after(0, self.update_vram_display, used_gb, self.peak_vram, info.total / (1024**3))
                time.sleep(1)
        except pynvml.NVMLError: self.root.after(0, lambda: self.vram_label_var.set("VRAM: Monitoring Error"))

    def update_vram_display(self, used, peak, total):
        self.vram_label_var.set(f"VRAM: {used:.2f} GB / {total:.2f} GB")
        self.peak_vram_label_var.set(f"Peak VRAM: {peak:.2f} GB")
        
    def update_loss_graph(self, step=None, loss_value=None):
        if not MATPLOTLIB_AVAILABLE: return
        if step is not None and loss_value is not None: self.loss_data.append((step, loss_value))
        self.ax.clear(); self.setup_graph_style()
        if self.loss_data:
            steps, losses = zip(*self.loss_data)
            self.ax.plot(steps, losses, color='#68bcece8')
        self.canvas.draw()
        
    def update_progress_bar(self, current, total):
        percentage = (current / total) * 100 if total > 0 else 0
        self.progress_var.set(percentage)
        self.progress_label_var.set(f"Epoch {current} of {total}" if total > 0 else "Epochs complete")

    def update_training_counters(self, current_step=None, total_steps=None, current_epoch=None, total_epochs=None):
        if current_step is not None:
            self.current_step = current_step
        if total_steps is not None:
            self.current_total_steps = total_steps
        if current_epoch is not None:
            self.current_epoch_num = current_epoch
        if total_epochs is not None:
            self.current_epoch_total = total_epochs

        if self.current_epoch_num > 0 and self.current_epoch_total > 0:
            self.epoch_counter_var.set(f"Epoch: {self.current_epoch_num} / {self.current_epoch_total}")
        else:
            self.epoch_counter_var.set("Epoch: N/A")

        if self.current_step > 0 and self.current_total_steps > 0:
            self.step_counter_var.set(f"Step: {self.current_step} / {self.current_total_steps}")
        else:
            self.step_counter_var.set("Step: N/A")

        if self.current_step > 0 and self.current_total_steps > 0 and self.current_epoch_num > 0 and self.current_epoch_total > 0:
            steps_per_epoch = max(1, (self.current_total_steps + self.current_epoch_total - 1) // self.current_epoch_total)
            next_epoch_step = min(self.current_total_steps, self.current_epoch_num * steps_per_epoch)
            remaining = max(0, next_epoch_step - self.current_step)
            self.next_epoch_var.set(f"To next epoch: {remaining} steps")
        else:
            self.next_epoch_var.set("To next epoch: N/A")
            
    def run_process(self, command, on_complete=None, output_widget=None):
        if output_widget is None: output_widget = self.output_text
        self.start_btn.config(state="disabled"); self.stop_btn.config(state="normal")
        self.last_line_was_progress = False
        command_display = ' '.join(f'"{part}"' if ' ' in part else part for part in command)
        output_widget.insert(tk.END, f"\n--- Running command ---\n{command_display}\n\n")

        try:
            env = os.environ.copy(); env['PYTHONUNBUFFERED'] = '1'; env['PYTHONUTF8'] = '1'
            project_root = os.getcwd(); src_path = os.path.join(project_root, 'src')
            env['PYTHONPATH'] = f"{src_path}{os.pathsep}{env.get('PYTHONPATH', '')}"

            self.current_process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=project_root,
                encoding='utf-8', errors='replace', bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0, env=env
            )
        except FileNotFoundError as e:
            messagebox.showerror("Error", f"Could not find '{e.filename}'. Is it in your system's PATH or venv?")
            self.stop_all_activity(); return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start process: {e}")
            self.stop_all_activity(); return
        
        threading.Thread(target=self.read_output, args=(on_complete, output_widget), daemon=True).start()
    
    def stop_all_activity(self):
        self.start_btn.config(state="normal"); self.stop_btn.config(state="disabled")
        self.stop_vram_monitor(); self._stop_sample_watcher(); self.current_process = None
        self.current_step = 0
        self.current_total_steps = 0
        self.current_epoch_num = 0
        self.current_epoch_total = 0
        self.update_training_counters()
        self.update_button_states()

    def process_console_output(self, line, output_widget):
        is_progress_line = line.endswith('\r')
        clean_line = line.strip()

        # Check if user is at the bottom of the text widget
        is_at_bottom = output_widget.yview()[1] >= 0.99

        if is_progress_line:
            if self.last_line_was_progress: output_widget.delete("end-2l", "end-1l")
            output_widget.insert(tk.END, clean_line + '\n')
            self.last_line_was_progress = True
        else:
            output_widget.insert(tk.END, line)
            self.last_line_was_progress = False

        # Only auto-scroll if user was already at the bottom
        if is_at_bottom:
            output_widget.see(tk.END)

    def read_output(self, on_complete, output_widget):
        if not self.current_process: 
            if on_complete: self.root.after(0, on_complete, -1); return
        try:
            buffer = ""
            while True:
                char = self.current_process.stdout.read(1)
                if not char and self.current_process.poll() is not None: break
                if not char: continue
                buffer += char
                if char in ('\n', '\r'):
                    self.root.after(0, self.process_console_output, buffer, output_widget)
                    if output_widget == self.output_text:
                        step_match = re.search(r"(\d+)/\d+ \[", buffer)
                        step_total_match = re.search(r"(\d+)/(\d+) \[", buffer)
                        if step_total_match:
                            self.root.after(
                                0,
                                self.update_training_counters,
                                int(step_total_match.group(1)),
                                int(step_total_match.group(2)),
                                None,
                                None,
                            )

                        loss_match = re.search(r"loss=([\d\.]+)", buffer)
                        if loss_match and self.current_step > 0:
                            self.root.after(0, self.update_loss_graph, self.current_step, float(loss_match.group(1)))

                        epoch_match = re.search(r"epoch\s*=?\s*(\d+)\s*/\s*(\d+)", buffer, re.IGNORECASE)
                        if epoch_match:
                            self.root.after(
                                0,
                                self.update_training_counters,
                                None,
                                None,
                                int(epoch_match.group(1)),
                                int(epoch_match.group(2)),
                            )
                            self.root.after(0, self.update_progress_bar, int(epoch_match.group(1)), int(epoch_match.group(2)))
                    buffer = ""
            if buffer: self.root.after(0, self.process_console_output, buffer, output_widget)
        except Exception as e:
            self.root.after(0, output_widget.insert, tk.END, f"\n[Read error] {e}\n")
        finally:
            return_code = self.current_process.wait() if self.current_process else -1
            self.current_process = None
            if on_complete: self.root.after(0, on_complete, return_code)

    def _run_next_command_in_sequence(self, return_code):
        if return_code != 0:
            self.output_text.insert(tk.END, f"\n--- Previous step failed with code {return_code}. Halting sequence. ---\n")
            self.stop_all_activity(); return
        if self.command_sequence:
            self.loss_data.clear()
            self.current_step = 0
            self.update_loss_graph() 
            next_command = self.command_sequence.pop(0)
            self.run_process(next_command, self._run_next_command_in_sequence, self.output_text)
        else:
            self.output_text.insert(tk.END, f"\n--- All steps completed successfully. ---\n")
            self.stop_all_activity()

    def _check_logging_dependencies(self, log_with):
        if log_with in ["wandb", "all"]:
            try: import wandb
            except Exception: messagebox.showerror("Missing Dependency", "Please run: pip install wandb"); return False
        if log_with in ["tensorboard", "all"]:
            try: import tensorboard
            except Exception: messagebox.showerror("Missing Dependency", "Please run: pip install tensorboard"); return False
        return True

    def start_training(self):
        self.update_button_states(); settings = self.get_settings()
        if not self._check_logging_dependencies(settings.get("log_with")): return
        mode = settings.get("training_mode", "Wan 2.2")
        if self.start_btn['state'] == 'disabled':
            messagebox.showerror("Validation Error", "Please fill all required fields before training."); return
        if mode == "Krea 2":
            if settings.get("fp8_base") and not settings.get("fp8_scaled"):
                messagebox.showerror("Validation Error", "Krea 2 requires FP8 Scaled when FP8 Base is enabled.")
                return
            if settings.get("krea2_turbo_dit_cache") and not settings.get("krea2_turbo_dit"):
                messagebox.showerror("Validation Error", "Turbo DiT cache requires a Turbo DiT model path in Krea 2 mode.")
                return
            if settings.get("krea2_turbo_dit") and (settings.get("blocks_to_swap") or "").strip() not in ("", "0"):
                messagebox.showerror("Validation Error", "Krea 2 Turbo DiT sampling is not compatible with Blocks to Swap. Clear one of them.")
                return
        # Warn if sample frequency is set but no prompts were added
        wants_samples = (settings.get("sample_every_n_epochs") or settings.get("sample_every_n_steps") or settings.get("sample_at_first"))
        if wants_samples and self._count_enabled_sample_prompts() == 0:
            messagebox.showwarning("No Active Sample Prompts", "You set a sample frequency but have no enabled prompts in the Samples tab.\nNo samples will be generated.\n\nEnable at least one saved prompt or add a new one.")
        
        self.loss_data.clear(); self.current_step = 0
        self.update_loss_graph(); self.start_vram_monitor(); self._start_sample_watcher()
        self.progress_var.set(0); self.progress_label_var.set("Starting sequence...")
        self.output_text.delete("1.0", tk.END); self.command_sequence = []
        python_executable = sys.executable or "python"
        is_wan = (mode == "Wan 2.2")

        if is_wan:
            cache_cmds = wan_backend.build_cache_commands(
                settings, python_executable, temp_config_fn=self._create_temp_cache_config
            )
        elif mode == "Krea 2":
            cache_cmds = krea2_backend.build_cache_commands(settings, python_executable)
        else:
            cache_cmds = flux2_backend.build_cache_commands(settings, python_executable)
        self.command_sequence.extend(cache_cmds)
        
        training_commands = self.build_training_commands()
        if training_commands: self.command_sequence.extend(training_commands)
        if self.command_sequence: self._run_next_command_in_sequence(0)
        else: messagebox.showwarning("Warning", "No training or caching steps were selected."); self.stop_all_activity()

    def stop_training(self):
        if self.current_process:
            self.output_text.insert(tk.END, "\n⚠️ Terminating process and sequence...\n")
            self.command_sequence = [];
            try: self.current_process.terminate()
            except Exception: pass
            self.stop_all_activity()

    def build_training_commands(self):
        settings = self.get_settings()
        settings["sample_prompts"] = self._build_sample_prompts_txt()
        mode = settings.get("training_mode", "Wan 2.2")
        if mode == "Wan 2.2":
            return wan_backend.build_commands(settings)
        elif mode == "Krea 2":
            return krea2_backend.build_commands(settings)
        else:
            return flux2_backend.build_commands(settings)

    def show_command(self):
        commands = self.build_training_commands()
        if commands:
            full_command_str = ""
            for i, command in enumerate(commands):
                command_str = " ".join(f'"{arg}"' if " " in arg else arg for arg in command)
                full_command_str += f"--- Command {i+1} ---\n{command_str}\n\n"
            dialog = tk.Toplevel(self.root); dialog.title("Generated Command(s)"); dialog.geometry("800x400")
            text = tk.Text(dialog, wrap="word", font=("Consolas", 10)); text.pack(expand=True, fill="both", padx=10, pady=10)
            text.insert("1.0", full_command_str); text.config(state="disabled")
            try: self.root.clipboard_clear(); self.root.clipboard_append(full_command_str)
            except Exception: pass

    def start_conversion(self):
        lora_path = self.entries["convert_lora_path"].get()
        output_dir = self.entries["convert_output_dir"].get()

        if not (lora_path and os.path.exists(lora_path) and output_dir and os.path.isdir(output_dir)):
            messagebox.showerror("Validation Error", "Please provide a valid LoRA file and a valid output directory."); return
        
        # --- MODIFIED --- Auto-generate output path and fix command arguments
        base_name = Path(lora_path).stem
        output_name = f"{base_name}_converted.safetensors"
        final_output_path = Path(output_dir) / output_name

        self.convert_output_text.delete("1.0", tk.END)
        python_executable = sys.executable or "python"
        target = getattr(self, '_convert_target_var', None)
        target = target.get() if target else "default"
        command = [python_executable, "src/musubi_tuner/convert_lora.py",
                   "--input", lora_path, "--output", str(final_output_path), "--target", target]
        
        self.run_process(command, on_complete=self.on_conversion_complete, output_widget=self.convert_output_text)

    def on_conversion_complete(self, return_code):
        if return_code == 0:
            self.convert_output_text.insert(tk.END, "\n--- Conversion completed successfully. ---")
        else:
            self.convert_output_text.insert(tk.END, f"\n--- Conversion failed with code {return_code}. ---")
        self.stop_all_activity()

    def run_accelerate_config(self):
        try:
            python_executable = Path(sys.executable)
            accelerate_path = python_executable.parent / "accelerate"
            if sys.platform == "win32":
                accelerate_path = accelerate_path.with_suffix(".exe")

            if not accelerate_path.exists():
                accelerate_path = "accelerate" 

            command = f'"{accelerate_path}" config'

            if sys.platform == "win32":
                subprocess.Popen(f'start cmd /k {command}', shell=True)
            elif sys.platform == "darwin":
                script = f'tell application "Terminal" to do script "{command}"'
                subprocess.Popen(['osascript', '-e', script])
            else: 
                try:
                    subprocess.Popen(['x-terminal-emulator', '-e', command])
                except FileNotFoundError:
                    messagebox.showerror("Error", "Could not find a default terminal. Please run 'accelerate config' manually in your terminal.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch 'accelerate config': {e}\nPlease run it manually in your activated virtual environment.")

    def on_closing(self):
        self._save_settings_to_file("last_settings.json")
        if self.current_process and messagebox.askokcancel("Quit", "A process is running. Stop it and quit?"):
            self.stop_training()
        self.stop_vram_monitor(); self.root.destroy()
        
if __name__ == "__main__":
    if not PYNVML_AVAILABLE: print("WARNING: pynvml not found. VRAM monitoring disabled. Run 'pip install pynvml'.")
    if not MATPLOTLIB_AVAILABLE: print("WARNING: matplotlib not found. Live graph disabled. Run 'pip install matplotlib'.")
    root = tk.Tk()
    app = MusubiTunerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
