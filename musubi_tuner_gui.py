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
from datetime import datetime
from collections import defaultdict

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

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# --- Helper Class for Tooltips ---
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip, add="+")
        self.widget.bind("<Leave>", self.hide_tooltip, add="+")

    def show_tooltip(self, _event=None):
        self.hide_tooltip()
        x = self.widget.winfo_rootx() + 24
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tooltip, text=self.text, justify="left", background="#E2E8F0",
            foreground="#0F172A", relief="solid", borderwidth=1,
            font=("Segoe UI", 9), wraplength=420, padx=8, pady=6,
        )
        label.pack()

    def hide_tooltip(self, _event=None):
        if self.tooltip:
            self.tooltip.destroy()
        self.tooltip = None


class ScrollableTab(ttk.Frame):
    """Notebook page with a fixed viewport and independently scrolling content."""

    def __init__(self, parent, background="#111827"):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, background=background, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas, style="Page.TFrame")
        self._window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.content.bind("<Configure>", self._sync_scrollregion)
        self.canvas.bind("<Configure>", self._sync_width)

    def _sync_scrollregion(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _sync_width(self, event):
        self.canvas.itemconfigure(self._window, width=event.width)

    def _on_mousewheel(self, event):
        if getattr(event, "num", None) == 4:
            direction = -1
        elif getattr(event, "num", None) == 5:
            direction = 1
        else:
            direction = -int(event.delta / 120) if event.delta else 0
        if direction:
            self.canvas.yview_scroll(direction * 3, "units")
        return "break"

# --- Main Application ---
class MusubiTunerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Musubi Tuner · LoRA Training Studio")
        self.root.minsize(980, 680)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        window_w = min(1180, max(980, screen_w - 180))
        window_h = min(760, max(680, screen_h - 180))
        pos_x = min(40, max(0, screen_w - window_w))
        pos_y = min(40, max(0, screen_h - window_h))
        self.root.geometry(f"{window_w}x{window_h}+{pos_x}+{pos_y}")

        self.entries = {}
        self.field_labels = {}
        self.field_label_text = {}
        self.hidden_frames = {}
        self.training_mode_var = tk.StringVar(value="Wan 2.2")
        self.appearance_mode_var = tk.StringVar(value=self._load_saved_appearance())
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
        self._sample_thumbnail_refs = {}
        self._sample_preview_images = {}
        self._sample_gallery_columns = 3
        self._job_history_path = "job_history_local.json"
        self._job_history = []
        self._jobs_tree = None
        self._jobs_details_text = None
        self._jobs_summary_var = tk.StringVar(value="No jobs recorded yet.")
        self._active_job = None
        self._stop_requested = False
        self._last_loss_value = None

        self.create_interface()
        self.load_default_settings()
        self._load_last_settings()
        self._load_job_history()
        self.update_button_states()

    def _load_saved_appearance(self):
        environment_value = os.environ.get("MUSUBI_GUI_THEME")
        if environment_value in ("Light", "Dark"):
            return environment_value
        try:
            with open("last_settings.json", "r", encoding="utf-8") as settings_file:
                value = json.load(settings_file).get("appearance_mode", "Dark")
                return value if value in ("Light", "Dark") else "Dark"
        except (OSError, ValueError, TypeError):
            return "Dark"

    def setup_styles(self):
        self.current_appearance_mode = self.appearance_mode_var.get()
        if self.appearance_mode_var.get() == "Light":
            self.colors = {
                "bg": "#E8EDF5", "page": "#F6F8FC", "surface": "#FFFFFF",
                "surface_alt": "#E9EEF6", "field": "#FFFFFF", "border": "#CBD5E1",
                "text": "#172033", "muted": "#64748B", "accent": "#2563EB",
                "accent_hover": "#1D4ED8", "success": "#15803D", "warning": "#B45309",
                "danger": "#DC2626", "selection": "#BFDBFE", "disabled": "#94A3B8",
                "danger_bg": "#FEE2E2", "danger_hover": "#FECACA",
            }
        else:
            self.colors = {
                "bg": "#182033", "page": "#202A3C", "surface": "#29364D",
                "surface_alt": "#36445E", "field": "#172033", "border": "#475569",
                "text": "#F1F5F9", "muted": "#B0BED0", "accent": "#60A5FA",
                "accent_hover": "#3B82F6", "success": "#4ADE80", "warning": "#FBBF24",
                "danger": "#F87171", "selection": "#1D4ED8", "disabled": "#718096",
                "danger_bg": "#7F1D1D", "danger_hover": "#991B1B",
            }
        BG_COLOR = self.colors["bg"]; TEXT_COLOR = self.colors["text"]
        FIELD_BG_COLOR = self.colors["field"]; SELECT_BG_COLOR = self.colors["selection"]
        BORDER_COLOR = self.colors["border"]; ERROR_BORDER = self.colors["danger"]

        self.root.configure(bg=BG_COLOR)
        style = ttk.Style()
        try: style.theme_use('clam')
        except Exception: pass

        style.configure('.', background=BG_COLOR, foreground=TEXT_COLOR, font=('Segoe UI', 10))
        style.configure('TFrame', background=self.colors["page"])
        style.configure('Page.TFrame', background=self.colors["page"])
        style.configure('Surface.TFrame', background=self.colors["surface"])
        style.configure('TLabel', background=self.colors["page"], foreground=TEXT_COLOR, font=('Segoe UI', 10))
        style.configure('Page.TLabel', background=self.colors["page"])
        style.configure('Muted.TLabel', background=self.colors["page"], foreground=self.colors["muted"], font=('Segoe UI', 9))
        style.configure('PageTitle.TLabel', background=self.colors["page"], foreground=TEXT_COLOR, font=('Segoe UI Semibold', 16))
        style.configure('PageHelp.TLabel', background=self.colors["page"], foreground=self.colors["muted"], font=('Segoe UI', 9))
        style.configure('Header.TFrame', background=self.colors["surface"])
        style.configure('Header.TLabel', background=self.colors["surface"], foreground=TEXT_COLOR)
        style.configure('TLabelframe', background=self.colors["page"], bordercolor=self.colors["page"], lightcolor=self.colors["page"], darkcolor=self.colors["page"], relief='flat', borderwidth=0)
        style.configure('TLabelframe.Label', background=self.colors["page"], foreground=TEXT_COLOR, font=('Segoe UI Semibold', 11))
        style.configure('TNotebook', background=BG_COLOR, bordercolor=BG_COLOR, lightcolor=BG_COLOR, darkcolor=BG_COLOR, borderwidth=0, tabmargins=[0, 0, 0, 0])
        style.configure('TNotebook.Tab', background=self.colors["surface"], foreground=self.colors["muted"], bordercolor=self.colors["surface"], lightcolor=self.colors["surface"], darkcolor=self.colors["surface"], padding=[16, 10], borderwidth=0, relief='flat', font=('Segoe UI Semibold', 9))
        style.map('TNotebook.Tab', background=[('selected', self.colors["page"]), ('active', self.colors["surface_alt"])], foreground=[('selected', self.colors["accent"]), ('active', TEXT_COLOR)])
        style.layout('Modern.TNotebook', [('Notebook.client', {'sticky': 'nswe'})])
        style.layout('Modern.TNotebook.Tab', [])
        style.configure('Modern.TNotebook', background=self.colors["page"], borderwidth=0)
        style.configure('Nav.TButton', background=self.colors["surface"], foreground=self.colors["muted"], borderwidth=0, relief='flat', padding=[10, 5], font=('Segoe UI Semibold', 9))
        style.map('Nav.TButton', background=[('active', self.colors["surface_alt"])], foreground=[('active', TEXT_COLOR)])
        style.configure('NavActive.TButton', background=self.colors["page"], foreground=self.colors["accent"], borderwidth=0, relief='flat', padding=[10, 5], font=('Segoe UI Semibold', 9))
        style.configure('TButton', background=self.colors["surface_alt"], foreground=TEXT_COLOR, font=('Segoe UI Semibold', 9), bordercolor=self.colors["surface_alt"], lightcolor=self.colors["surface_alt"], darkcolor=self.colors["surface_alt"], borderwidth=0, relief='flat', padding=[11, 7])
        style.map('TButton', background=[('active', self.colors["border"]), ('pressed', self.colors["border"])], foreground=[('disabled', self.colors["disabled"])])
        style.configure('Accent.TButton', background=self.colors["accent_hover"], foreground='#FFFFFF', padding=[14, 8])
        style.map('Accent.TButton', background=[('active', self.colors["accent"]), ('pressed', '#0284C7')])
        style.configure('Danger.TButton', background=self.colors["danger_bg"], foreground=self.colors["danger"])
        style.map('Danger.TButton', background=[('active', self.colors["danger_hover"]), ('pressed', self.colors["danger_hover"])])
        style.configure('TEntry', foreground=TEXT_COLOR, fieldbackground=FIELD_BG_COLOR, insertcolor=TEXT_COLOR, borderwidth=1, relief='solid', bordercolor=BORDER_COLOR, padding=6)
        style.map('TCombobox', fieldbackground=[('readonly', FIELD_BG_COLOR)], foreground=[('readonly', TEXT_COLOR)], selectbackground=[('readonly', SELECT_BG_COLOR)])
        self.root.option_add('*TCombobox*Listbox.background', FIELD_BG_COLOR); self.root.option_add('*TCombobox*Listbox.foreground', TEXT_COLOR)
        self.root.option_add('*TCombobox*Listbox.selectBackground', SELECT_BG_COLOR); self.root.option_add('*TCombobox*Listbox.selectForeground', TEXT_COLOR)
        style.configure('TCheckbutton', font=('Segoe UI', 10)); style.configure('Title.TLabel', background=self.colors["surface"], font=('Segoe UI Semibold', 18))
        style.configure('Subtitle.TLabel', background=self.colors["surface"], foreground=self.colors["muted"], font=('Segoe UI', 9))
        style.configure('Status.TLabel', font=('Segoe UI Semibold', 11)); style.configure('TProgressbar', thickness=12, background=self.colors["accent"], troughcolor=FIELD_BG_COLOR)
        style.configure('Invalid.TEntry', fieldbackground=FIELD_BG_COLOR, bordercolor=ERROR_BORDER, foreground=TEXT_COLOR, relief='solid', borderwidth=1)
        style.configure('Valid.TEntry', fieldbackground=FIELD_BG_COLOR, bordercolor=BORDER_COLOR, foreground=TEXT_COLOR, relief='solid', borderwidth=1)
        style.configure(
            'Treeview',
            background=FIELD_BG_COLOR,
            fieldbackground=FIELD_BG_COLOR,
            foreground=TEXT_COLOR,
            bordercolor=BORDER_COLOR,
            lightcolor=BORDER_COLOR,
            darkcolor=BORDER_COLOR,
            rowheight=28,
            relief='flat',
        )
        style.map(
            'Treeview',
            background=[('selected', self.colors["selection"])],
            foreground=[('selected', TEXT_COLOR)],
        )
        style.configure(
            'Treeview.Heading',
            background=self.colors["surface"],
            foreground=TEXT_COLOR,
            bordercolor=BORDER_COLOR,
            lightcolor=BORDER_COLOR,
            darkcolor=BORDER_COLOR,
            font=('Segoe UI Semibold', 9),
            relief='flat',
            padding=(8, 6),
        )
        style.map(
            'Treeview.Heading',
            background=[('active', self.colors["surface_alt"])],
            foreground=[('active', TEXT_COLOR)],
        )

    def create_interface(self):
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, style="Header.TFrame", padding=(18, 12))
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        brand = ttk.Frame(header, style="Header.TFrame")
        brand.grid(row=0, column=0, sticky="w")
        self.title_label = ttk.Label(brand, text="Musubi Tuner", style='Title.TLabel')
        self.title_label.pack(anchor='w')
        self.subtitle_label = ttk.Label(brand, text="LoRA training studio", style="Subtitle.TLabel")
        self.subtitle_label.pack(anchor='w')

        toolbar = ttk.Frame(header, style="Header.TFrame")
        toolbar.grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Label(toolbar, text="Training mode", style="Header.TLabel").pack(side="left", padx=(0, 8))
        self.mode_combo = ttk.Combobox(toolbar, textvariable=self.training_mode_var,
                                      values=["Wan 2.2", "Flux.2 Klein", "Flux.2 Dev", "Krea 2"],
                                      state="readonly", width=18)
        self.mode_combo.pack(side="left", padx=(0, 12)); self.mode_combo.bind("<MouseWheel>", lambda e: "break")
        self.mode_combo.bind("<<ComboboxSelected>>", self.on_training_mode_change)
        ttk.Label(toolbar, text="Theme", style="Header.TLabel").pack(side="left", padx=(0, 7))
        self.appearance_combo = ttk.Combobox(
            toolbar, textvariable=self.appearance_mode_var,
            values=["Light", "Dark"], state="readonly", width=7,
        )
        self.appearance_combo.pack(side="left", padx=(0, 12))
        self.appearance_combo.bind("<MouseWheel>", lambda _e: "break")
        self.appearance_combo.bind("<<ComboboxSelected>>", self._request_appearance_change)
        self.create_settings_buttons(toolbar)

        body = ttk.Frame(self.root)
        body.grid(row=1, column=0, sticky="nsew")
        self.nav_bar = ttk.Frame(body, style="Header.TFrame", width=170, padding=(10, 12))
        self.nav_bar.pack(side="left", fill="y", padx=(12, 0), pady=(10, 8))
        self.nav_bar.pack_propagate(False)
        self.notebook = ttk.Notebook(body, style="Modern.TNotebook")
        self.notebook.pack(side="right", fill="both", expand=True, padx=12, pady=(10, 8))

        self.create_model_paths_tab()
        self.create_training_params_tab()
        self.create_advanced_tab()
        self.create_samples_tab()
        self.create_run_monitor_tab()
        self.create_jobs_tab()
        self.create_convert_lora_tab()
        self.create_accelerate_config_tab()
        self._build_navigation()

        footer = ttk.Frame(self.root, style="Header.TFrame", padding=(14, 5))
        footer.grid(row=2, column=0, sticky="ew")
        self.mode_note_label = ttk.Label(footer, text="", style="Subtitle.TLabel")
        self.mode_note_label.pack(side="left", fill="x", expand=True)
        self.validation_status_var = tk.StringVar(value="Checking configuration…")
        self.validation_status_label = ttk.Label(footer, textvariable=self.validation_status_var, style="Subtitle.TLabel")
        self.validation_status_label.pack(side="right")

        self.root.bind("<Control-s>", lambda _e: self.save_settings())
        self.root.bind("<Control-o>", lambda _e: self.load_settings())
        self.root.bind("<Control-Return>", lambda _e: self.start_training())
        self.root.bind_all("<MouseWheel>", self._route_tab_mousewheel)
        self.root.bind_all("<Button-4>", self._route_tab_mousewheel)
        self.root.bind_all("<Button-5>", self._route_tab_mousewheel)

    def create_settings_buttons(self, parent):
        button_frame = ttk.Frame(parent, style="Header.TFrame")
        button_frame.pack(side="left")
        ttk.Button(button_frame, text="Load", command=self.load_settings).pack(side="left", padx=(0, 4))
        ttk.Button(button_frame, text="Save", command=self.save_settings).pack(side="left", padx=(0, 4))
        reset_btn = ttk.Button(button_frame, text="Reset", command=self._confirm_reset_settings)
        reset_btn.pack(side="left")
        ToolTip(reset_btn, "Restore every setting to its default value.")

    def _create_scrollable_tab(self, title):
        tab = ScrollableTab(self.notebook, background=self.colors["page"])
        self.notebook.add(tab, text=title)
        return tab.content

    def _build_navigation(self):
        labels = ("Models", "Training", "Advanced", "Samples", "Monitor", "Jobs", "Convert", "Setup")
        self._nav_buttons = []
        ttk.Label(self.nav_bar, text="WORKSPACE", style="Subtitle.TLabel").pack(anchor="w", pady=(0, 6))
        for index, label in enumerate(labels):
            button = ttk.Button(
                self.nav_bar, text=label, style="Nav.TButton",
                command=lambda page=index: self._select_page(page),
            )
            button.pack(fill="x", pady=(0, 3))
            self._nav_buttons.append(button)
        self.notebook.bind("<<NotebookTabChanged>>", self._sync_navigation)
        self._sync_navigation()

    def _select_page(self, index):
        self.notebook.select(index)

    def _sync_navigation(self, _event=None):
        try:
            selected = self.notebook.index(self.notebook.select())
        except tk.TclError:
            selected = 0
        for index, button in enumerate(self._nav_buttons):
            button.configure(style="NavActive.TButton" if index == selected else "Nav.TButton")

    def _add_page_intro(self, parent, title, description):
        intro = ttk.Frame(parent, style="Page.TFrame")
        intro.pack(fill="x", padx=12, pady=(14, 5))
        ttk.Label(intro, text=title, style="PageTitle.TLabel").pack(anchor="w")
        ttk.Label(intro, text=description, style="PageHelp.TLabel", wraplength=920, justify="left").pack(anchor="w", pady=(3, 0))

    def _confirm_reset_settings(self):
        if messagebox.askyesno("Reset settings", "Restore every field to its default value?\n\nThis does not delete saved settings files or training outputs."):
            self.load_default_settings()

    def _request_appearance_change(self, _event=None):
        requested = self.appearance_mode_var.get()
        if requested == self.current_appearance_mode:
            return
        if self.current_process:
            self.appearance_mode_var.set(self.current_appearance_mode)
            messagebox.showinfo("Training active", "Appearance can be changed after the current process finishes.")
            return
        self.root.after_idle(self._apply_appearance_mode, requested)

    def _apply_appearance_mode(self, requested, settings=None):
        if requested not in ("Light", "Dark") or requested == self.current_appearance_mode:
            return
        settings = dict(settings or self.get_settings())
        settings["appearance_mode"] = requested
        self.appearance_mode_var.set(requested)

        for child in self.root.winfo_children():
            child.destroy()
        self.entries = {}
        self.field_labels = {}
        self.field_label_text = {}
        self.hidden_frames = {}
        self._sample_list_frame = None
        self._sample_thumbnail_refs = {}
        self._sample_preview_images = {}
        self._jobs_tree = None
        self._jobs_details_text = None

        self.setup_styles()
        self.create_interface()
        self.set_values(settings)
        self.update_button_states()
        self.update_loss_graph()

    def _route_tab_mousewheel(self, event):
        """Scroll the active page unless a nested widget handles the wheel itself."""
        try:
            selected = self.notebook.select()
            page = self.root.nametowidget(selected)
            if isinstance(page, ScrollableTab):
                return page._on_mousewheel(event)
        except (tk.TclError, AttributeError):
            pass
        return None

    def _add_widget(self, parent, key, label, tooltip, kind='entry', options=None, is_required=False, validate_num=False, is_path=False, is_dir=False, default_val=False, command=None):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=10, pady=5)
        frame.grid_columnconfigure(1, weight=1)
        label_widget = None
        if kind != 'checkbox':
            label_text = label.rstrip(":") + ("  *" if is_required else "")
            label_widget = ttk.Label(frame, text=label_text, width=27, anchor="w")
            label_widget.grid(row=0, column=0, sticky="w", padx=(0, 14))

        widget = None
        if kind == 'path_entry':
            path_frame = ttk.Frame(frame)
            path_frame.grid(row=0, column=1, sticky="ew")
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
            widget.grid(row=0, column=1, sticky="ew"); widget.bind("<MouseWheel>", lambda e: "break")
            if command: widget.bind("<<ComboboxSelected>>", command)
        elif kind == 'checkbox':
            var = tk.BooleanVar(value=default_val)
            def chained_command(event=None):
                if command and callable(command): command()
                self.update_button_states()
            widget = ttk.Checkbutton(frame, text=label, variable=var, command=chained_command)
            widget.var = var; widget.grid(row=0, column=0, columnspan=2, sticky="w")
        else:
            vcmd = (self.root.register(self.validate_number), '%P') if validate_num else None
            widget = ttk.Entry(frame, validate="key", validatecommand=vcmd)
            widget.grid(row=0, column=1, sticky="ew")

        if tooltip:
            ToolTip(widget, tooltip)
            if label_widget is not None:
                ToolTip(label_widget, tooltip)
        self.entries[key] = widget
        if label_widget is not None:
            self.field_labels[key] = label_widget
            self.field_label_text[key] = label.rstrip(":")
        widget.is_required = is_required; widget.is_path = is_path
        if isinstance(widget, ttk.Entry):
            widget.bind("<FocusOut>", self.update_button_states); widget.bind("<KeyRelease>", self.update_button_states)
        return widget

    def create_model_paths_tab(self):
        frame = self._create_scrollable_tab("1  Models")
        self._add_page_intro(frame, "Models & dataset", "Choose the dataset, model components, and output destination for the selected training mode. Required fields are marked with an asterisk.")

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
                                           foreground=self.colors["warning"], font=("Segoe UI", 9, "italic"),
                                           wraplength=940, justify="left")
        self.flux2_note_label.pack(anchor="w", padx=8, pady=(0, 4))

        self._add_widget(self.hidden_frames['flux2_model_paths'], "flux2_dit_model", "DiT Model:", "Path to the Flux.2 DiT model (.safetensors).", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_required=True, is_path=True)
        self._add_widget(self.hidden_frames['flux2_model_paths'], "flux2_text_encoder", "Text Encoder (Qwen3 or Mistral3):", "Path to the Qwen3 or Mistral3 text encoder directory or safetensors file.", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_required=True, is_path=True)
        self._add_widget(self.hidden_frames['flux2_model_paths'], "fp8_text_encoder", "FP8 Text Encoder", "Load the text encoder in FP8 precision to reduce VRAM.", kind='checkbox')

        # ---- Krea 2 model paths section ----
        self.hidden_frames['krea2_model_paths'] = ttk.LabelFrame(frame, text="Krea 2 Model Paths")

        self.krea2_note_label = ttk.Label(
            self.hidden_frames['krea2_model_paths'],
            text="Train on RAW DiT. Qwen-Image VAE is required. Qwen3-VL text encoder is only required for text re-caching and sample generation. Upstream starting point: bf16, rank 32, alpha 32, timestep_sampling=krea2_shift.",
            foreground=self.colors["warning"], font=("Segoe UI", 9, "italic"),
            wraplength=940, justify="left",
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
        frame = self._create_scrollable_tab("2  Training")
        self._add_page_intro(frame, "Training recipe", "Set learning duration, network capacity, optimizer behavior, and the learning-rate schedule.")
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
        frame = self._create_scrollable_tab("3  Advanced")
        self._add_page_intro(frame, "Advanced controls", "Tune memory usage, timestep sampling, attention, logging, precision, and resume behavior.")
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
        tab_frame = self._create_scrollable_tab("4  Samples")
        self._add_page_intro(tab_frame, "Sample previews", "Choose when previews run, manage reusable prompts, and inspect generated samples without deleting inactive prompts.")

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
                  foreground=self.colors["muted"], font=("Segoe UI", 9, "italic")).pack(side="left", padx=(12, 0))

        plist_container = ttk.Frame(prompts_frame); plist_container.pack(fill="x", padx=5, pady=(0, 6))
        plist_canvas = tk.Canvas(plist_container, bg=self.colors["page"], highlightthickness=0, height=170)
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
        canvas = tk.Canvas(list_container, bg=self.colors["page"], highlightthickness=0, height=220)
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=canvas.yview)
        self._sample_list_frame = ttk.Frame(canvas)
        self._sample_list_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._sample_list_frame, anchor="nw", tags="sframe")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig("sframe", width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")
        self._sample_canvas = canvas

        ttk.Label(self._sample_list_frame, text="No samples yet. Start training to generate samples.",
                  foreground=self.colors["muted"]).pack(padx=10, pady=10)

    # ---------- Prompt list helpers ----------

    def _rebuild_prompt_list(self):
        for w in self._prompt_list_inner.winfo_children():
            w.destroy()
        if not self._sample_prompts_data:
            ttk.Label(self._prompt_list_inner, text="No prompts added yet.",
                      foreground=self.colors["muted"]).pack(padx=10, pady=8)
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
                      foreground=self.colors["text"] if p.get("enabled", True) else self.colors["disabled"]).pack(side="left", fill="x", expand=True)
            preview_btn = ttk.Button(row, text="Preview", width=7,
                       command=lambda i=idx: self._test_sample_prompt(i))
            preview_btn.pack(side="right", padx=(3, 0))
            if self.training_mode_var.get() != "Krea 2":
                preview_btn.configure(state="disabled")
                ToolTip(preview_btn, "Standalone preview generation is currently available for Krea 2.")
            ttk.Button(row, text="Duplicate", width=9,
                       command=lambda i=idx: self._duplicate_sample_prompt(i)).pack(side="right", padx=(3, 0))
            ttk.Button(row, text="Edit", width=5,
                       command=lambda i=idx: self._edit_sample_prompt_dialog(i)).pack(side="right", padx=(3, 0))
            ttk.Button(row, text="Delete", width=7, style="Danger.TButton",
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
        prompt_summary = prompt_data.get("prompt", "")[:120]
        self._begin_job("sample_test", "Krea 2 test sample", settings=settings, note=prompt_summary)
        self.run_process(command, on_complete=self._on_test_sample_complete, output_widget=self.output_text, job_context={"attach_to_active": True})

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
        self._finalize_active_job("completed" if return_code == 0 else ("stopped" if self._stop_requested else "failed"), return_code)
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
        dlg.geometry("720x650" if is_krea2 else "680x650")
        dlg.minsize(620, 600)
        dlg.configure(bg=self.colors["page"])
        dlg.resizable(True, False)
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
        e_prompt = tk.Text(dlg, height=5, wrap=tk.WORD, bg=self.colors["field"], fg=self.colors["text"],
                           insertbackground=self.colors["text"], selectbackground=self.colors["selection"],
                           relief=tk.FLAT, padx=8, pady=6, font=("Segoe UI", 10))
        e_prompt.insert("1.0", existing.get("prompt", ""))
        e_prompt.pack(fill="both", expand=True, padx=10, pady=(0, 2))

        lbl(dlg, "Negative prompt  (optional)")
        e_neg = tk.Text(dlg, height=3, wrap=tk.WORD, bg=self.colors["field"], fg=self.colors["text"],
                        insertbackground=self.colors["text"], selectbackground=self.colors["selection"],
                        relief=tk.FLAT, padx=8, pady=6, font=("Segoe UI", 10))
        e_neg.insert("1.0", existing.get("neg", ""))
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
                foreground=self.colors["muted"],
                font=("Segoe UI", 9, "italic"),
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
            prompt_text = " ".join(line.strip() for line in e_prompt.get("1.0", "end-1c").splitlines() if line.strip())
            if not prompt_text:
                messagebox.showerror("Validation", "Prompt text cannot be empty.", parent=dlg); return
            data = {"prompt": prompt_text}
            data["neg"]        = " ".join(line.strip() for line in e_neg.get("1.0", "end-1c").splitlines() if line.strip())
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
        ttk.Button(btn_row, text="Save Prompt", style="Accent.TButton", command=_save).pack(side="left", padx=6)
        ttk.Button(btn_row, text="Cancel", command=dlg.destroy).pack(side="left", padx=6)
        dlg.bind("<Control-Return>", lambda _e: _save())
        dlg.bind("<Escape>", lambda _e: dlg.destroy())
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
        self._open_path(output_dir)

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
        self._sample_thumbnail_refs = {}

        for w in self._sample_list_frame.winfo_children():
            w.destroy()

        if not files:
            ttk.Label(self._sample_list_frame, text="No samples yet. Start training to generate samples.",
                      foreground=self.colors["muted"]).pack(padx=10, pady=10)
            return

        columns = self._sample_gallery_columns
        for col in range(columns):
            self._sample_list_frame.grid_columnconfigure(col, weight=1, uniform="samplegrid")

        for idx, (mtime, fpath) in enumerate(reversed(files)):
            card = ttk.Frame(self._sample_list_frame, style="Surface.TFrame", padding=8)
            row = idx // columns
            col = idx % columns
            card.grid(row=row, column=col, sticky="nsew", padx=6, pady=6)
            self._build_sample_card(card, mtime, fpath)

    def _build_sample_card(self, parent, mtime, fpath):
        ext = Path(fpath).suffix.lower()
        is_image = ext in (".png", ".jpg", ".jpeg", ".webp")
        thumb_frame = tk.Frame(parent, bg=self.colors["surface_alt"], highlightthickness=0, bd=0, width=180, height=140)
        thumb_frame.pack(fill="x")
        thumb_frame.pack_propagate(False)

        if is_image and PIL_AVAILABLE:
            thumbnail = self._load_thumbnail_image(fpath)
            if thumbnail is not None:
                thumb_label = tk.Label(
                    thumb_frame,
                    image=thumbnail,
                    bg=self.colors["surface_alt"],
                    cursor="hand2",
                    relief=tk.FLAT,
                    bd=0,
                )
                thumb_label.image = thumbnail
                thumb_label.pack(expand=True)
                thumb_label.bind("<Button-1>", lambda _e, p=fpath: self._open_sample_preview(p))
            else:
                self._build_sample_placeholder(thumb_frame, "Preview unavailable")
        elif is_image:
            self._build_sample_placeholder(thumb_frame, "PIL not installed")
        else:
            self._build_sample_placeholder(thumb_frame, ext.upper().lstrip(".") or "FILE")

        name = Path(fpath).name
        short_name = name if len(name) <= 38 else name[:18] + "..." + name[-16:]
        ttk.Label(parent, text=short_name, anchor="w").pack(fill="x", pady=(8, 0))
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
        ttk.Label(parent, text=ts, style="Muted.TLabel").pack(anchor="w", pady=(2, 0))

        actions = ttk.Frame(parent)
        actions.pack(fill="x", pady=(8, 0))
        ttk.Button(
            actions,
            text="Preview" if is_image else "Open",
            command=lambda p=fpath, image=is_image: self._open_sample_preview(p) if image else self._open_path(p),
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(actions, text="Folder", command=lambda p=fpath: self._open_path(os.path.dirname(p))).pack(side="left", padx=(6, 0))

    def _build_sample_placeholder(self, parent, text):
        label = tk.Label(
            parent,
            text=text,
            bg=self.colors["surface_alt"],
            fg=self.colors["muted"],
            font=("Segoe UI Semibold", 11),
        )
        label.pack(expand=True)

    def _load_thumbnail_image(self, fpath):
        if not PIL_AVAILABLE:
            return None
        try:
            with Image.open(fpath) as image:
                preview = image.copy()
            preview.thumbnail((180, 140))
            photo = ImageTk.PhotoImage(preview)
            self._sample_thumbnail_refs[fpath] = photo
            return photo
        except Exception:
            return None

    def _open_sample_preview(self, fpath):
        ext = Path(fpath).suffix.lower()
        if ext not in (".png", ".jpg", ".jpeg", ".webp") or not PIL_AVAILABLE:
            self._open_path(fpath)
            return

        dialog = tk.Toplevel(self.root)
        dialog.title(Path(fpath).name)
        dialog.geometry("960x760")
        dialog.minsize(420, 320)
        dialog.configure(bg=self.colors["page"])

        toolbar = ttk.Frame(dialog)
        toolbar.pack(fill="x", padx=10, pady=(10, 6))
        ttk.Button(toolbar, text="Open Externally", command=lambda: self._open_path(fpath)).pack(side="right")
        info = ttk.Label(toolbar, text=fpath, style="PageHelp.TLabel", wraplength=760, justify="left")
        info.pack(side="left", fill="x", expand=True)

        viewport = tk.Canvas(dialog, bg=self.colors["field"], highlightthickness=0, bd=0)
        viewport.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        def render_preview(_event=None):
            try:
                with Image.open(fpath) as source:
                    preview = source.copy()
                max_w = max(120, viewport.winfo_width() - 20)
                max_h = max(120, viewport.winfo_height() - 20)
                preview.thumbnail((max_w, max_h))
                photo = ImageTk.PhotoImage(preview)
                self._sample_preview_images[fpath] = photo
                viewport.delete("all")
                viewport.create_image(viewport.winfo_width() // 2, viewport.winfo_height() // 2, image=photo, anchor="center")
            except Exception as exc:
                viewport.delete("all")
                viewport.create_text(20, 20, anchor="nw", text=f"Could not load preview:\n{exc}", fill=self.colors["muted"])

        viewport.bind("<Configure>", render_preview)
        render_preview()

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
        tab_frame = ttk.Frame(self.notebook); self.notebook.add(tab_frame, text="5  Monitor")
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
        self.start_btn = ttk.Button(train_button_frame, text="Start Training", style="Accent.TButton", command=self.start_training); self.start_btn.pack(side="left", padx=(0, 5), expand=True, fill='x')
        self.stop_btn = ttk.Button(train_button_frame, text="Stop Training", style="Danger.TButton", command=self.stop_training, state="disabled"); self.stop_btn.pack(side="left", padx=5, expand=True, fill='x')
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

        bottom_pane_host = ttk.Frame(tab_frame)
        bottom_pane_host.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        bottom_pane = ttk.PanedWindow(bottom_pane_host, orient=tk.HORIZONTAL)
        bottom_pane.pack(fill='both', expand=True)
        graph_frame = ttk.LabelFrame(bottom_pane, text="Live Loss"); bottom_pane.add(graph_frame, weight=1)
        if MATPLOTLIB_AVAILABLE:
            self.fig = Figure(figsize=(5, 2.8), dpi=100); self.ax = self.fig.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame); self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.setup_graph_style()
        else: ttk.Label(graph_frame, text="Matplotlib not found.\nInstall with 'pip install matplotlib'", wraplength=200, justify='center').pack(expand=True)
        console_frame = ttk.LabelFrame(bottom_pane, text="Console Output"); bottom_pane.add(console_frame, weight=1)
        console_toolbar = ttk.Frame(console_frame)
        console_toolbar.pack(fill="x", padx=5, pady=(4, 2))
        ttk.Button(console_toolbar, text="Copy", command=self._copy_console_output).pack(side="right", padx=(4, 0))
        ttk.Button(console_toolbar, text="Clear", command=lambda: self.output_text.delete("1.0", tk.END)).pack(side="right")
        self.output_text = tk.Text(console_frame, wrap=tk.WORD, height=14, bg=self.colors["field"], fg=self.colors["text"], insertbackground=self.colors["text"], selectbackground=self.colors["selection"], font=('Consolas', 9), relief=tk.FLAT, bd=0, padx=8, pady=6)
        output_scrollbar = ttk.Scrollbar(console_frame, orient="vertical", command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=output_scrollbar.set); self.output_text.pack(side="left", fill="both", expand=True); output_scrollbar.pack(side="right", fill="y")

    def _copy_console_output(self):
        text = self.output_text.get("1.0", tk.END).strip()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def create_jobs_tab(self):
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="6  Jobs")
        tab_frame.grid_columnconfigure(0, weight=1)
        tab_frame.grid_rowconfigure(2, weight=1)

        intro = ttk.Frame(tab_frame, style="Page.TFrame")
        intro.grid(row=0, column=0, sticky="ew", padx=12, pady=(14, 5))
        ttk.Label(intro, text="Recent jobs", style="PageTitle.TLabel").pack(anchor="w")
        ttk.Label(
            intro,
            text="Track completed, failed, and stopped runs locally. Historical imports read saved settings, model folders, and log files when available.",
            style="PageHelp.TLabel",
            wraplength=920,
            justify="left",
        ).pack(anchor="w", pady=(3, 0))

        summary_row = ttk.Frame(tab_frame)
        summary_row.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        ttk.Label(summary_row, textvariable=self._jobs_summary_var, style="PageHelp.TLabel").pack(side="left", fill="x", expand=True)
        ttk.Button(summary_row, text="Refresh", command=self._refresh_job_history_view).pack(side="right", padx=(6, 0))
        ttk.Button(summary_row, text="Import Found Jobs", command=self._import_historical_jobs).pack(side="right", padx=(6, 0))
        ttk.Button(summary_row, text="Clear History", style="Danger.TButton", command=self._clear_job_history).pack(side="right")

        split = ttk.PanedWindow(tab_frame, orient=tk.HORIZONTAL)
        split.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))

        list_frame = ttk.LabelFrame(split, text="Recorded Jobs")
        split.add(list_frame, weight=3)
        columns = ("status", "mode", "started", "progress", "title")
        self._jobs_tree = ttk.Treeview(
            list_frame,
            columns=columns,
            show="headings",
            selectmode="browse",
            height=16,
        )
        headings = {
            "status": "Status",
            "mode": "Mode",
            "started": "Started",
            "progress": "Progress",
            "title": "Title",
        }
        widths = {"status": 96, "mode": 92, "started": 132, "progress": 128, "title": 340}
        anchors = {"status": "center", "mode": "center", "started": "center", "progress": "center", "title": "w"}
        for key in columns:
            self._jobs_tree.heading(key, text=headings[key])
            self._jobs_tree.column(key, width=widths[key], minwidth=70, anchor=anchors[key], stretch=(key == "title"))
        list_scroll_y = ttk.Scrollbar(list_frame, orient="vertical", command=self._jobs_tree.yview)
        list_scroll_x = ttk.Scrollbar(list_frame, orient="horizontal", command=self._jobs_tree.xview)
        self._jobs_tree.configure(yscrollcommand=list_scroll_y.set, xscrollcommand=list_scroll_x.set)
        self._jobs_tree.pack(side="top", fill="both", expand=True, padx=6, pady=(6, 0))
        list_scroll_y.pack(side="right", fill="y", pady=(6, 6), padx=(0, 6))
        list_scroll_x.pack(side="bottom", fill="x", padx=6, pady=(0, 6))
        self._jobs_tree.bind("<<TreeviewSelect>>", lambda _e: self._show_selected_job_details())
        self._jobs_tree.bind("<Double-1>", lambda _e: self._open_selected_job_output())

        details_frame = ttk.LabelFrame(split, text="Job Details")
        split.add(details_frame, weight=4)
        details_toolbar = ttk.Frame(details_frame)
        details_toolbar.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Button(details_toolbar, text="Open Output", command=self._open_selected_job_output).pack(side="left")
        ttk.Button(details_toolbar, text="Open Logs", command=self._open_selected_job_logs).pack(side="left", padx=(6, 0))
        ttk.Button(details_toolbar, text="Copy Command", command=self._copy_selected_job_command).pack(side="left", padx=(6, 0))
        self._jobs_details_text = tk.Text(
            details_frame,
            wrap=tk.WORD,
            bg=self.colors["field"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            selectbackground=self.colors["selection"],
            font=("Consolas", 9),
            relief=tk.FLAT,
            bd=0,
            padx=8,
            pady=6,
            state="disabled",
        )
        details_scroll = ttk.Scrollbar(details_frame, orient="vertical", command=self._jobs_details_text.yview)
        self._jobs_details_text.configure(yscrollcommand=details_scroll.set)
        self._jobs_details_text.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        details_scroll.pack(side="right", fill="y", pady=6, padx=(0, 6))
        self._refresh_job_history_view()

    def _load_job_history(self):
        try:
            with open(self._job_history_path, "r", encoding="utf-8") as history_file:
                data = json.load(history_file)
            self._job_history = data if isinstance(data, list) else []
        except (OSError, ValueError, TypeError):
            self._job_history = []
        self._refresh_job_history_view()

    def _save_job_history(self):
        try:
            with open(self._job_history_path, "w", encoding="utf-8") as history_file:
                json.dump(self._job_history[:200], history_file, indent=2)
        except OSError:
            pass

    def _refresh_job_history_view(self):
        if self._jobs_tree is None:
            return
        for item in self._jobs_tree.get_children():
            self._jobs_tree.delete(item)
        completed = failed = stopped = 0
        for job in self._job_history:
            status = job.get("status", "unknown")
            if status == "completed":
                completed += 1
            elif status == "failed":
                failed += 1
            elif status == "stopped":
                stopped += 1
        total = len(self._job_history)
        self._jobs_summary_var.set(
            f"{total} jobs saved locally  |  completed {completed}  |  failed {failed}  |  stopped {stopped}"
            if total else "No jobs recorded yet."
        )
        for index, job in enumerate(self._job_history):
            started = job.get("started_at", "")
            timestamp = started.replace("T", " ")[:16] if started else ""
            step_now = job.get("current_step") or 0
            step_total = job.get("total_steps") or 0
            epoch_now = job.get("current_epoch") or 0
            epoch_total = job.get("total_epochs") or 0
            if step_total:
                progress = f"{step_now}/{step_total}"
            elif epoch_total:
                progress = f"e{epoch_now}/{epoch_total}"
            else:
                progress = "-"
            self._jobs_tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    job.get("status", "unknown"),
                    job.get("mode", ""),
                    timestamp,
                    progress,
                    job.get("title", "Job"),
                ),
            )
        if self._job_history:
            first = self._jobs_tree.get_children()[0]
            self._jobs_tree.selection_set(first)
            self._jobs_tree.focus(first)
            self._show_selected_job_details()
        else:
            self._set_job_details_text("No recorded jobs yet.")

    def _selected_job(self):
        if self._jobs_tree is None:
            return None
        selection = self._jobs_tree.selection()
        if not selection:
            return None
        index = int(selection[0])
        if index >= len(self._job_history):
            return None
        return self._job_history[index]

    def _set_job_details_text(self, text):
        if self._jobs_details_text is None:
            return
        self._jobs_details_text.config(state="normal")
        self._jobs_details_text.delete("1.0", tk.END)
        self._jobs_details_text.insert("1.0", text)
        self._jobs_details_text.config(state="disabled")

    def _show_selected_job_details(self):
        job = self._selected_job()
        if not job:
            self._set_job_details_text("No recorded jobs yet.")
            return
        lines = [
            f"Title: {job.get('title', '')}",
            f"Kind: {job.get('kind', '')}",
            f"Status: {job.get('status', '')}",
            f"Mode: {job.get('mode', '')}",
            f"Started: {job.get('started_at', '')}",
            f"Finished: {job.get('finished_at', '')}",
            f"Duration: {job.get('duration_seconds', 0):.1f}s" if job.get("duration_seconds") is not None else "Duration: N/A",
            f"Output Name: {job.get('output_name', '')}",
            f"Output Dir: {job.get('output_dir', '')}",
            f"Resume: {job.get('resume_path', '')}",
            f"Logging Dir: {job.get('logging_dir', '')}",
            f"Peak VRAM: {job.get('peak_vram_gb', 'N/A')}",
            f"Last Loss: {job.get('last_loss', 'N/A')}",
            f"Step: {job.get('current_step', 'N/A')} / {job.get('total_steps', 'N/A')}",
            f"Epoch: {job.get('current_epoch', 'N/A')} / {job.get('total_epochs', 'N/A')}",
            "",
            "Prompt / note:",
            job.get("note", "") or "(none)",
            "",
            "Commands:",
        ]
        commands = job.get("commands") or []
        if commands:
            lines.extend(commands)
        else:
            lines.append("(none)")
        self._set_job_details_text("\n".join(lines))

    def _open_path(self, path):
        if not path:
            return
        if os.path.isfile(path):
            target = path
        elif os.path.isdir(path):
            target = path
        else:
            return
        if sys.platform == "win32":
            os.startfile(target)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", target])
        else:
            subprocess.Popen(["xdg-open", target])

    def _open_selected_job_output(self):
        job = self._selected_job()
        if not job:
            return
        output_dir = job.get("output_dir", "")
        output_name = job.get("output_name", "")
        preferred = os.path.join(output_dir, output_name) if output_dir and output_name else output_dir
        if os.path.exists(preferred):
            self._open_path(preferred)
        elif output_dir and os.path.exists(output_dir):
            self._open_path(output_dir)

    def _open_selected_job_logs(self):
        job = self._selected_job()
        if not job:
            return
        logging_dir = job.get("logging_dir", "")
        if logging_dir and os.path.exists(logging_dir):
            self._open_path(logging_dir)

    def _copy_selected_job_command(self):
        job = self._selected_job()
        if not job:
            return
        commands = job.get("commands") or []
        if not commands:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append("\n\n".join(commands))

    def _clear_job_history(self):
        if not self._job_history:
            return
        if not messagebox.askyesno("Clear job history", "Delete all locally saved job history entries?"):
            return
        self._job_history = []
        self._save_job_history()
        self._refresh_job_history_view()

    def _infer_mode_from_path(self, value):
        lowered = str(value or "").lower()
        if "krea" in lowered:
            return "Krea 2"
        if "flux" in lowered or "klein" in lowered:
            return "Flux.2"
        if "wan" in lowered:
            return "Wan 2.2"
        return "Unknown"

    def _history_key(self, job):
        return (
            job.get("kind", ""),
            job.get("output_dir", ""),
            job.get("output_name", ""),
            job.get("started_at", ""),
            job.get("title", ""),
        )

    def _extract_metrics_from_output_log(self, text):
        step_now = step_total = epoch_now = epoch_total = 0
        loss_value = None

        step_matches = re.findall(r"^steps:\s+\d+%\|.*?\|\s*(\d+)/(\d+)\s+\[", text, re.MULTILINE)
        if step_matches:
            step_now, step_total = map(int, step_matches[-1])

        epoch_matches = re.findall(r"epoch\s+(\d+)/(\d+)", text, re.IGNORECASE)
        if epoch_matches:
            epoch_now, epoch_total = map(int, epoch_matches[-1])

        loss_matches = re.findall(r"avr_loss=([\d.]+)", text)
        if loss_matches:
            try:
                loss_value = float(loss_matches[-1])
            except ValueError:
                loss_value = None

        return step_now, step_total, epoch_now, epoch_total, loss_value

    def _extract_command_context_from_wandb_config(self, text):
        context = {
            "commands": [],
            "output_dir": "",
            "output_name": "",
            "resume_path": "",
            "started_at": "",
        }

        started_match = re.search(r"startedAt:\s*\"([^\"]+)\"", text)
        if started_match:
            context["started_at"] = started_match.group(1).replace("Z", "")

        key_map = {"--output_dir": "output_dir", "--output_name": "output_name", "--resume": "resume_path"}
        current_key = None
        command_parts = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("- "):
                value = line[2:].strip().strip('"')
                command_parts.append(value)
                if value in key_map:
                    current_key = key_map[value]
                elif current_key:
                    context[current_key] = value
                    current_key = None
            elif current_key:
                current_key = None
        if command_parts:
            context["commands"] = [" ".join(f'"{part}"' if " " in part else part for part in command_parts)]
        return context

    def _normalise_imported_job(self, job):
        normalised = {
            "kind": job.get("kind", "training"),
            "title": job.get("title", "Imported job"),
            "mode": job.get("mode", "Unknown"),
            "status": job.get("status", "completed"),
            "started_at": job.get("started_at", ""),
            "finished_at": job.get("finished_at", ""),
            "duration_seconds": job.get("duration_seconds"),
            "return_code": job.get("return_code"),
            "output_dir": job.get("output_dir", ""),
            "output_name": job.get("output_name", ""),
            "resume_path": job.get("resume_path", ""),
            "logging_dir": job.get("logging_dir", ""),
            "note": job.get("note", ""),
            "commands": job.get("commands", []),
            "peak_vram_gb": job.get("peak_vram_gb"),
            "last_loss": job.get("last_loss"),
            "current_step": job.get("current_step", 0),
            "total_steps": job.get("total_steps", 0),
            "current_epoch": job.get("current_epoch", 0),
            "total_epochs": job.get("total_epochs", 0),
        }
        return normalised

    def _import_historical_jobs(self):
        settings = self.get_settings()
        roots = []
        output_dir = settings.get("output_dir", "")
        logging_dir = settings.get("logging_dir", "")
        if output_dir:
            roots.append(Path(output_dir).expanduser())
            parent = Path(output_dir).expanduser().parent
            if parent not in roots:
                roots.append(parent)
        if logging_dir:
            roots.append(Path(logging_dir).expanduser())
        roots = [root for root in roots if root.exists()]
        if not roots:
            messagebox.showinfo("Import Jobs", "Set a valid output or logging directory first.")
            return

        imported = []
        imported.extend(self._scan_settings_json_jobs(roots))
        imported.extend(self._scan_model_directory_jobs(roots))
        imported.extend(self._scan_log_directory_jobs(roots))

        existing = {self._history_key(job): index for index, job in enumerate(self._job_history)}
        added = 0
        updated = 0
        for job in imported:
            normalised = self._normalise_imported_job(job)
            key = self._history_key(normalised)
            if key in existing:
                current = self._job_history[existing[key]]
                current.update({k: v for k, v in normalised.items() if v not in ("", [], None, 0) or k in ("status", "mode", "title")})
                updated += 1
            else:
                self._job_history.append(normalised)
                existing[key] = len(self._job_history) - 1
                added += 1

        self._job_history.sort(key=lambda job: job.get("started_at", ""), reverse=True)
        self._job_history = self._job_history[:200]
        self._save_job_history()
        self._refresh_job_history_view()
        messagebox.showinfo("Import Jobs", f"Imported {added} historical job(s), updated {updated}.")

    def _scan_settings_json_jobs(self, roots):
        jobs = []
        for root in roots:
            for json_path in root.glob("*.json"):
                try:
                    with open(json_path, "r", encoding="utf-8") as handle:
                        data = json.load(handle)
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue
                if "output_dir" not in data or "output_name" not in data:
                    continue
                started = datetime.fromtimestamp(json_path.stat().st_mtime).isoformat(timespec="seconds")
                jobs.append(
                    {
                        "kind": "training",
                        "title": data.get("output_name") or json_path.stem,
                        "mode": self._infer_mode_from_path(
                            data.get("LoRA_type") or data.get("training_mode") or data.get("pretrained_model_name_or_path") or json_path.name
                        ),
                        "status": "completed",
                        "started_at": started,
                        "finished_at": started,
                        "output_dir": data.get("output_dir", ""),
                        "output_name": data.get("output_name", ""),
                        "logging_dir": data.get("logging_dir", ""),
                        "resume_path": data.get("resume", ""),
                        "note": f"Imported from settings file {json_path.name}",
                        "commands": [],
                        "current_epoch": data.get("epoch", 0) or data.get("max_train_epochs", 0) or 0,
                        "total_epochs": data.get("max_train_epochs", 0) or 0,
                    }
                )
        return jobs

    def _scan_model_directory_jobs(self, roots):
        jobs = []
        seen_dirs = set()
        for root in roots:
            search_root = root if root.is_dir() else root.parent
            for state_dir in search_root.rglob("*-state"):
                if not state_dir.is_dir():
                    continue
                run_dir = state_dir.parent
                if run_dir in seen_dirs:
                    continue
                seen_dirs.add(run_dir)
                mode = self._infer_mode_from_path(run_dir.as_posix())
                states = sorted(p for p in run_dir.glob("*-state") if p.is_dir())
                sample_dir = run_dir / "sample"
                sample_test_dir = run_dir / "sample_test"
                started_ts = min((p.stat().st_mtime for p in states), default=run_dir.stat().st_mtime)
                finished_ts = max(
                    [p.stat().st_mtime for p in states] +
                    ([sample_dir.stat().st_mtime] if sample_dir.exists() else []) +
                    ([sample_test_dir.stat().st_mtime] if sample_test_dir.exists() else [])
                )
                jobs.append(
                    {
                        "kind": "training",
                        "title": run_dir.name,
                        "mode": mode,
                        "status": "completed",
                        "started_at": datetime.fromtimestamp(started_ts).isoformat(timespec="seconds"),
                        "finished_at": datetime.fromtimestamp(finished_ts).isoformat(timespec="seconds"),
                        "output_dir": str(run_dir.parent),
                        "output_name": run_dir.name,
                        "logging_dir": "",
                        "resume_path": "",
                        "note": "Imported from model/state folders",
                        "commands": [],
                        "current_epoch": len(states),
                        "total_epochs": len(states),
                    }
                )
                if sample_test_dir.exists():
                    for sample_file in sorted(sample_test_dir.glob("*")):
                        if sample_file.is_file():
                            ts = datetime.fromtimestamp(sample_file.stat().st_mtime).isoformat(timespec="seconds")
                            jobs.append(
                                {
                                    "kind": "sample_test",
                                    "title": f"{run_dir.name} sample test",
                                    "mode": mode,
                                    "status": "completed",
                                    "started_at": ts,
                                    "finished_at": ts,
                                    "output_dir": str(run_dir.parent),
                                    "output_name": run_dir.name,
                                    "logging_dir": "",
                                    "resume_path": "",
                                    "note": sample_file.name,
                                    "commands": [],
                                }
                            )
        return jobs

    def _scan_log_directory_jobs(self, roots):
        jobs = []
        for root in roots:
            if root.name != "log":
                candidates = [p for p in root.rglob("log") if p.is_dir()]
            else:
                candidates = [root]
            for log_root in candidates:
                for run_dir in sorted(p for p in log_root.iterdir() if p.is_dir()):
                    mode = self._infer_mode_from_path(run_dir.name)
                    started = datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(timespec="seconds")
                    note = "Imported from log folder"
                    last_loss = None
                    current_step = total_steps = current_epoch = total_epochs = 0
                    output_dir = ""
                    output_name = run_dir.name
                    resume_path = ""
                    commands = []
                    summary_json = next(run_dir.rglob("wandb-summary.json"), None)
                    if summary_json:
                        try:
                            with open(summary_json, "r", encoding="utf-8") as handle:
                                summary = json.load(handle)
                            last_loss = summary.get("loss") or summary.get("avr_loss") or summary.get("loss/current") or summary.get("loss/average")
                            current_step = summary.get("global_step") or summary.get("train/global_step") or summary.get("_step") or 0
                            current_epoch = summary.get("epoch") or 0
                        except Exception:
                            pass
                    output_log = next(run_dir.rglob("output.log"), None)
                    if output_log:
                        try:
                            text = output_log.read_text(encoding="utf-8", errors="replace")
                            parsed_step, parsed_total, parsed_epoch, parsed_epoch_total, parsed_loss = self._extract_metrics_from_output_log(text)
                            current_step = current_step or parsed_step
                            total_steps = total_steps or parsed_total
                            current_epoch = current_epoch or parsed_epoch
                            total_epochs = total_epochs or parsed_epoch_total
                            last_loss = last_loss if last_loss is not None else parsed_loss
                        except Exception:
                            pass
                    config_yaml = next(run_dir.rglob("config.yaml"), None)
                    if config_yaml:
                        try:
                            context = self._extract_command_context_from_wandb_config(
                                config_yaml.read_text(encoding="utf-8", errors="replace")
                            )
                            commands = context["commands"]
                            output_dir = context["output_dir"] or output_dir
                            output_name = context["output_name"] or output_name
                            resume_path = context["resume_path"] or resume_path
                            started = context["started_at"] or started
                        except Exception:
                            pass
                    jobs.append(
                        {
                            "kind": "training",
                            "title": output_name,
                            "mode": mode,
                            "status": "completed",
                            "started_at": started,
                            "finished_at": started,
                            "output_dir": output_dir,
                            "output_name": output_name,
                            "logging_dir": str(run_dir),
                            "resume_path": resume_path,
                            "note": note,
                            "commands": commands,
                            "last_loss": last_loss,
                            "current_step": current_step,
                            "total_steps": total_steps,
                            "current_epoch": current_epoch,
                            "total_epochs": total_epochs,
                        }
                    )
        return jobs

    def _begin_job(self, kind, title, settings=None, note=""):
        settings = settings or self.get_settings()
        self._stop_requested = False
        self._last_loss_value = None
        self._active_job = {
            "kind": kind,
            "title": title,
            "mode": settings.get("training_mode", self.training_mode_var.get()),
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": settings.get("output_dir", ""),
            "output_name": settings.get("output_name", ""),
            "resume_path": settings.get("resume_path", ""),
            "logging_dir": settings.get("logging_dir", ""),
            "note": note,
            "commands": [],
            "peak_vram_gb": None,
            "last_loss": None,
            "current_step": 0,
            "total_steps": 0,
            "current_epoch": 0,
            "total_epochs": 0,
        }

    def _record_job_command(self, command):
        if not self._active_job:
            return
        command_display = " ".join(f'"{part}"' if " " in part else part for part in command)
        self._active_job.setdefault("commands", []).append(command_display)

    def _finalize_active_job(self, status, return_code=None):
        if not self._active_job or self._active_job.get("status") != "running":
            return
        finished_at = datetime.now()
        started_raw = self._active_job.get("started_at")
        try:
            started_at = datetime.fromisoformat(started_raw)
            duration_seconds = max(0.0, (finished_at - started_at).total_seconds())
        except Exception:
            duration_seconds = None
        self._active_job.update(
            {
                "status": status,
                "finished_at": finished_at.isoformat(timespec="seconds"),
                "duration_seconds": duration_seconds,
                "return_code": return_code,
                "peak_vram_gb": round(self.peak_vram, 2) if self.peak_vram else 0.0,
                "last_loss": self._last_loss_value,
                "current_step": self.current_step,
                "total_steps": self.current_total_steps,
                "current_epoch": self.current_epoch_num,
                "total_epochs": self.current_epoch_total,
            }
        )
        self._job_history.insert(0, self._active_job)
        self._job_history = self._job_history[:200]
        self._active_job = None
        self._save_job_history()
        self._refresh_job_history_view()

    def create_convert_lora_tab(self):
        tab_frame = self._create_scrollable_tab("Convert")
        main_frame = ttk.Frame(tab_frame); main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        info_frame = ttk.LabelFrame(main_frame, text="Format Reference"); info_frame.pack(fill='x', pady=(0, 10))
        info_text = tk.Text(info_frame, wrap=tk.WORD, bg=self.colors["field"], fg=self.colors["muted"], font=('Consolas', 9),
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
        self.convert_output_text = tk.Text(console_frame, wrap=tk.WORD, bg=self.colors["field"], fg=self.colors["text"], insertbackground=self.colors["text"], font=('Consolas', 9), relief=tk.FLAT, bd=0)
        scrollbar = ttk.Scrollbar(console_frame, orient="vertical", command=self.convert_output_text.yview)
        self.convert_output_text.configure(yscrollcommand=scrollbar.set); self.convert_output_text.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")

    def create_accelerate_config_tab(self):
        tab_frame = self._create_scrollable_tab("Setup")
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
        info_text = tk.Text(info_frame, wrap=tk.WORD, bg=self.colors["field"], fg=self.colors["text"], font=('Segoe UI', 10), relief=tk.FLAT, bd=0, height=15)
        info_text.insert(tk.END, info_text_content); info_text.config(state="disabled")
        info_text.pack(fill='x', expand=True, padx=10, pady=10)

        action_frame = ttk.LabelFrame(main_frame, text="Run Configuration"); action_frame.pack(fill='x')
        button = ttk.Button(action_frame, text="Run Accelerate Config", command=self.run_accelerate_config)
        button.pack(pady=20)

    def on_training_mode_change(self, event=None):
        mode = self.training_mode_var.get()
        is_wan = (mode == "Wan 2.2")
        self.title_label.config(text="Musubi Tuner")
        self.subtitle_label.config(text=f"{mode} · LoRA training studio")
        self.root.title(f"Musubi Tuner · {mode}")

        if is_wan:
            self.mode_note_label.config(text="Dual-stage Wan training · T2V and I2V workflows")
            self.hidden_frames['flux2_model_paths'].pack_forget()
            self.hidden_frames['krea2_model_paths'].pack_forget()
            self.hidden_frames['wan_dit'].pack(fill="x", padx=10, pady=10, before=self._vae_frame)
            self.hidden_frames['wan_models'].pack(fill="x", padx=10, pady=10, before=self._vae_frame)
        elif mode == "Krea 2":
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
        self.fig.patch.set_facecolor(self.colors["page"]); self.ax.set_facecolor(self.colors["field"])
        self.ax.tick_params(axis='x', colors=self.colors["muted"]); self.ax.tick_params(axis='y', colors=self.colors["muted"])
        for spine in self.ax.spines.values(): spine.set_color(self.colors["border"])
        self.ax.yaxis.label.set_color(self.colors["muted"]); self.ax.xaxis.label.set_color(self.colors["muted"])
        self.ax.title.set_color(self.colors["text"]); self.ax.set_xlabel("Steps"); self.ax.set_ylabel("Loss")
        self.ax.grid(color=self.colors["border"], alpha=0.25, linewidth=0.6)
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
        invalid_fields = []
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

        for key, widget in self.entries.items():
            if not isinstance(widget, tk.Widget): continue
            if key in self.field_labels:
                required = bool(getattr(widget, "is_required", False))
                self.field_labels[key].configure(text=self.field_label_text[key] + ("  *" if required else ""))
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
                if not is_valid:
                    all_valid = False
                    invalid_fields.append(key)

        if is_wan:
            if not (train_high or train_low): all_valid = False
        self.start_btn.config(state="normal" if all_valid else "disabled")
        try:
            if self.current_process:
                self.validation_status_var.set("Training process active")
                self.validation_status_label.configure(foreground=self.colors["accent"])
            elif all_valid:
                self.validation_status_var.set("Configuration ready · Ctrl+Enter to start")
                self.validation_status_label.configure(foreground=self.colors["success"])
            else:
                count = len(invalid_fields)
                noun = "field" if count == 1 else "fields"
                self.validation_status_var.set(f"{count} required {noun} need attention")
                self.validation_status_label.configure(foreground=self.colors["warning"])
        except AttributeError:
            pass
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
        settings["appearance_mode"] = self.appearance_mode_var.get()
        settings["sample_prompts_data"] = self._sample_prompts_data
        return settings

    def set_values(self, settings):
        requested_appearance = settings.get("appearance_mode")
        if requested_appearance in ("Light", "Dark") and requested_appearance != self.current_appearance_mode:
            if not self.current_process:
                self._apply_appearance_mode(requested_appearance, settings=settings)
            return
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
            if key in ("training_mode", "appearance_mode", "sample_prompts_data", "sample_prompts"): continue
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

    def run_process(self, command, on_complete=None, output_widget=None, job_context=None):
        if output_widget is None: output_widget = self.output_text
        self.start_btn.config(state="disabled"); self.stop_btn.config(state="normal")
        self.last_line_was_progress = False
        command_display = ' '.join(f'"{part}"' if ' ' in part else part for part in command)
        output_widget.insert(tk.END, f"\n--- Running command ---\n{command_display}\n\n")
        if job_context and job_context.get("attach_to_active"):
            self._record_job_command(command)

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
            if job_context and job_context.get("attach_to_active"):
                self._finalize_active_job("failed", -1)
            self.stop_all_activity(); return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start process: {e}")
            if job_context and job_context.get("attach_to_active"):
                self._finalize_active_job("failed", -1)
            self.stop_all_activity(); return

        threading.Thread(target=self.read_output, args=(on_complete, output_widget), daemon=True).start()

    def stop_all_activity(self):
        self.start_btn.config(state="normal"); self.stop_btn.config(state="disabled")
        self.stop_vram_monitor(); self._stop_sample_watcher(); self.current_process = None
        self.current_step = 0
        self.current_total_steps = 0
        self.current_epoch_num = 0
        self.current_epoch_total = 0
        self._stop_requested = False
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

                        loss_match = re.search(r"(?:avr_loss|loss)=([\d\.]+)", buffer)
                        if loss_match and self.current_step > 0:
                            loss_value = float(loss_match.group(1))
                            self._last_loss_value = loss_value
                            self.root.after(0, self.update_loss_graph, self.current_step, loss_value)

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
            self._finalize_active_job("stopped" if self._stop_requested else "failed", return_code)
            self.output_text.insert(tk.END, f"\n--- Previous step failed with code {return_code}. Halting sequence. ---\n")
            self.stop_all_activity(); return
        if self.command_sequence:
            self.loss_data.clear()
            self.current_step = 0
            self.update_loss_graph()
            next_command = self.command_sequence.pop(0)
            self.run_process(next_command, self._run_next_command_in_sequence, self.output_text, job_context={"attach_to_active": True})
        else:
            self._finalize_active_job("completed", 0)
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
        self._begin_job(
            "training",
            "Resumed training" if self.entries["resume_path"].get().strip() else "Training run",
            settings=settings,
            note=f"Output: {settings.get('output_name', '')}",
        )
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
        if self.command_sequence:
            self._run_next_command_in_sequence(0)
        else:
            self._finalize_active_job("failed", -1)
            messagebox.showwarning("Warning", "No training or caching steps were selected.")
            self.stop_all_activity()

    def stop_training(self):
        if self.current_process:
            self.output_text.insert(tk.END, "\n⚠️ Terminating process and sequence...\n")
            self._stop_requested = True
            self.command_sequence = []
            self.run_status_var.set("🛑 Stopping Run")
            self.progress_label_var.set("Stopping current process...")
            try:
                self.current_process.terminate()
            except Exception:
                self._finalize_active_job("stopped", -1)
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

        conversion_settings = dict(self.get_settings())
        conversion_settings["output_dir"] = output_dir
        conversion_settings["output_name"] = output_name
        self._begin_job("conversion", "Convert LoRA", settings=conversion_settings, note=base_name)
        self.run_process(command, on_complete=self.on_conversion_complete, output_widget=self.convert_output_text, job_context={"attach_to_active": True})

    def on_conversion_complete(self, return_code):
        self._finalize_active_job("completed" if return_code == 0 else ("stopped" if self._stop_requested else "failed"), return_code)
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
