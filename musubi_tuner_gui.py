import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import io
import threading
import copy
import json
import os
import re
import signal
import struct
import time
import sys
import uuid
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from backends import wan as wan_backend, flux2 as flux2_backend, krea2 as krea2_backend, krea2_face as krea2_face_backend, krea2_face_eval as krea2_face_eval_backend
from backends.flux2 import FLUX2_VERSION_MAP
from dataset_config_builder import DatasetConfigBuilder
from prompt_library import PromptLibraryDialog, PromptLibraryStore, prompt_identity

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
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False

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
        self._vram_gpu_index = None
        self._vram_previous_gpu_index = None
        self._vram_baseline = []
        self.loss_data = []
        self.peak_vram = 0
        self.command_sequence = []
        self._staged_run = None
        self._staged_training_config = []
        self._staged_recache_latents = True
        self._face_refinement_config = {}
        self._face_eval_context = None
        self.last_line_was_progress = False
        self.last_progress_milestone = 0
        self.current_step = 0
        self.current_total_steps = 0
        self.current_epoch_num = 0
        self.current_epoch_total = 0
        self.current_prior_steps = 0
        self.current_prior_epochs = 0
        self._pending_continuation = None
        self._last_loss_step = 0
        self.sample_watcher_active = False
        self._sample_watcher_thread = None
        self._last_sample_files = []
        self._sample_list_frame = None
        self._sample_prompts_data = []  # list of dicts
        self.prompt_library = PromptLibraryStore()
        self._prompt_library_dialog = None
        self._sample_test_context = None
        self._temp_prompts_file = None  # path to auto-written temp .txt
        self._sample_thumbnail_refs = {}
        self._sample_preview_images = {}
        self._sample_gallery_columns = 3
        self._lora_shape_cache = {}
        self._job_history_path = "job_history_local.json"
        self._job_history = []
        self._jobs_tree = None
        self._jobs_details_text = None
        self._jobs_context_menu = None
        self._jobs_summary_var = tk.StringVar(value="No jobs recorded yet.")
        self._active_job = None
        self._monitor_top_collapsed = False
        self._monitor_top_sash_position = 270
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
        toggle_indicator_bg = FIELD_BG_COLOR
        toggle_indicator_fg = self.colors["accent"]
        toggle_indicator_border = BORDER_COLOR
        style.configure(
            'TCheckbutton',
            font=('Segoe UI', 10),
            background=self.colors["page"],
            foreground=TEXT_COLOR,
            indicatorbackground=toggle_indicator_bg,
            indicatorforeground=toggle_indicator_fg,
            upperbordercolor=toggle_indicator_border,
            lowerbordercolor=toggle_indicator_border,
            focuscolor=self.colors["page"],
        )
        style.map(
            'TCheckbutton',
            background=[('disabled', self.colors["page"]), ('active', self.colors["page"])],
            foreground=[('disabled', self.colors["disabled"]), ('active', TEXT_COLOR)],
            indicatorbackground=[('disabled', self.colors["surface_alt"]), ('active', toggle_indicator_bg)],
            indicatorforeground=[('disabled', self.colors["disabled"]), ('selected', toggle_indicator_fg)],
        )
        style.configure(
            'TRadiobutton',
            font=('Segoe UI', 10),
            background=self.colors["page"],
            foreground=TEXT_COLOR,
            indicatorbackground=toggle_indicator_bg,
            indicatorforeground=toggle_indicator_fg,
            upperbordercolor=toggle_indicator_border,
            lowerbordercolor=toggle_indicator_border,
            focuscolor=self.colors["page"],
        )
        style.map(
            'TRadiobutton',
            background=[('disabled', self.colors["page"]), ('active', self.colors["page"])],
            foreground=[('disabled', self.colors["disabled"]), ('active', TEXT_COLOR)],
            indicatorbackground=[('disabled', self.colors["surface_alt"]), ('active', toggle_indicator_bg)],
            indicatorforeground=[('disabled', self.colors["disabled"]), ('selected', toggle_indicator_fg)],
        )
        style.configure('Title.TLabel', background=self.colors["surface"], font=('Segoe UI Semibold', 18))
        style.configure('Subtitle.TLabel', background=self.colors["surface"], foreground=self.colors["muted"], font=('Segoe UI', 9))
        style.configure('Status.TLabel', font=('Segoe UI Semibold', 11)); style.configure('TProgressbar', thickness=12, background=self.colors["accent"], troughcolor=FIELD_BG_COLOR)
        style.configure(
            "Console.Vertical.TScrollbar",
            background=self.colors["surface_alt"],
            troughcolor=FIELD_BG_COLOR,
            bordercolor=BORDER_COLOR,
            lightcolor=self.colors["surface_alt"],
            darkcolor=self.colors["surface_alt"],
            arrowcolor=TEXT_COLOR,
            relief="flat",
            borderwidth=1,
            arrowsize=15,
        )
        style.map(
            "Console.Vertical.TScrollbar",
            background=[("active", self.colors["accent"]), ("pressed", self.colors["accent_hover"])],
        )
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
        self.create_face_refinement_tab()
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
        self.root.bind("<Control-Return>", lambda _e: self.start_selected_run())
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
        labels = ("Models", "Training", "Advanced", "Face Refinement", "Samples", "Monitor", "Jobs", "Convert", "Setup")
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
            if event.widget.winfo_toplevel() is not self.root:
                return "break"
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
                if path:
                    widget.delete(0, tk.END); widget.insert(0, path)
                    if command and callable(command): command()
                    self.update_button_states()
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

    def _themed_text(self, parent, **kwargs):
        """Create a multiline editor using the same palette as themed entry fields."""
        options = {
            "bg": self.colors["field"],
            "fg": self.colors["text"],
            "insertbackground": self.colors["text"],
            "selectbackground": self.colors["selection"],
            "selectforeground": self.colors["text"],
            "highlightbackground": self.colors["border"],
            "highlightcolor": self.colors["accent"],
            "highlightthickness": 1,
            "relief": tk.FLAT,
            "bd": 0,
            "padx": 7,
            "pady": 6,
            "font": ("Segoe UI", 10),
        }
        options.update(kwargs)
        return tk.Text(parent, **options)

    def _open_dataset_config_builder(self):
        current_path = self.entries["dataset_config"].get().strip()

        def use_config(path):
            entry = self.entries["dataset_config"]
            entry.delete(0, tk.END)
            entry.insert(0, path)
            self.update_button_states()

        DatasetConfigBuilder(
            self.root,
            initial_path=current_path,
            on_use=use_config,
            colors=self.colors,
        )

    def create_model_paths_tab(self):
        frame = self._create_scrollable_tab("1  Models")
        self._add_page_intro(frame, "Models & dataset", "Choose the dataset, model components, and output destination for the selected training mode. Required fields are marked with an asterisk.")

        dataset_frame = ttk.LabelFrame(frame, text="Dataset Configuration"); dataset_frame.pack(fill="x", padx=10, pady=10)
        dataset_entry = self._add_widget(dataset_frame, "dataset_config", "Dataset Config (TOML):", "Path to .toml dataset configuration file.", kind='path_entry', options=[("TOML files", "*.toml")], is_required=True, is_path=True)
        builder_button = ttk.Button(dataset_entry.master, text="Create / Edit", command=self._open_dataset_config_builder)
        builder_button.pack(side="right", padx=(5, 0))
        ToolTip(builder_button, "Opens a visual dataset-config builder. You can also edit and validate the raw TOML.")

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
        self._add_widget(output_frame, "project_root", "Concept Workspace:", "Optional main folder for this concept. Selecting it creates and fills the models and log subfolders.", kind='path_entry', is_dir=True, is_path=True, command=self._apply_project_root_paths)
        workspace_actions = ttk.Frame(output_frame); workspace_actions.pack(fill="x", padx=10, pady=(0, 5))
        ttk.Button(workspace_actions, text="Apply Workspace Layout", command=self._apply_project_root_paths).pack(side="right")
        ttk.Label(workspace_actions, text="Creates models/ and log/ and fills the related paths.", style="PageHelp.TLabel").pack(side="right", padx=(0, 10))
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
        self._add_widget(self.hidden_frames['low_noise_lora_params'], "network_dim_low", "Network Dimension (Rank):", "Controls network capacity. LoRA: 32-128 typical. LoHa: use lower values (4-32) because its paired decomposition is more expressive. LoKr: start around 16-32; sufficiently large values switch its larger Kronecker factor to a full matrix, so size and capacity stop increasing like ordinary LoRA rank.", is_required=True, validate_num=True)
        self._add_widget(self.hidden_frames['low_noise_lora_params'], "network_alpha_low", "Network Alpha:", "Scaling factor for network weights. Often set to half of Network Dimension, or equal to it for LoHa/LoKr.", is_required=True, validate_num=True)

        self.hidden_frames['high_noise_lora_params'] = ttk.LabelFrame(network_container, text="High Noise Network Parameters")
        self._add_widget(self.hidden_frames['high_noise_lora_params'], "network_dim_high", "Network Dimension (Rank):", "Leave blank to use the same as the Low Noise model. If different, a separate training run will be executed.", is_required=False, validate_num=True)
        self._add_widget(self.hidden_frames['high_noise_lora_params'], "network_alpha_high", "Network Alpha:", "Leave blank to use the same as the Low Noise model.", is_required=False, validate_num=True)

        size_frame = ttk.LabelFrame(frame, text="Estimated Adapter Size")
        size_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.lora_size_estimate_var = tk.StringVar(value="Select a DiT model and enter a rank to estimate the final LoRA size.")
        size_label = ttk.Label(
            size_frame,
            textvariable=self.lora_size_estimate_var,
            style="PageHelp.TLabel",
            wraplength=900,
            justify="left",
        )
        size_label.pack(fill="x", padx=10, pady=8)
        ToolTip(
            size_label,
            "Estimates the final LoRA .safetensors file by reading model tensor shapes without loading model weights. Rank and network type change the size; network alpha only changes weight scaling and has no effect on file size. Optimizer and training-state files are not included.",
        )

        notes_frame = ttk.LabelFrame(frame, text="Training Notes")
        notes_frame.pack(fill="x", padx=10, pady=(0, 10))
        notes_help = ttk.Label(
            notes_frame,
            text="Saved with this GUI session, shown in Jobs, and embedded in LoRA metadata.",
            style="PageHelp.TLabel",
        )
        notes_help.pack(anchor="w", padx=10, pady=(8, 3))
        self.training_comment_text = tk.Text(
            notes_frame,
            height=4,
            wrap=tk.WORD,
            bg=self.colors["field"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            selectbackground=self.colors["selection"],
            relief=tk.FLAT,
            bd=0,
            padx=8,
            pady=6,
            font=("Segoe UI", 10),
        )
        self.training_comment_text.pack(fill="x", padx=10, pady=(0, 10))
        self.entries["training_comment"] = self.training_comment_text
        notes_tooltip = (
            "Write reminders about the dataset, intended strength, trigger words, experiments, or results. "
            "The complete text is saved in settings and local job history and is embedded in each output "
            "safetensors file as ss_training_comment. It does not affect training."
        )
        ToolTip(notes_help, notes_tooltip)
        ToolTip(self.training_comment_text, notes_tooltip)

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

    def _apply_project_root_paths(self):
        root_value = self.entries["project_root"].get().strip()
        if not root_value:
            messagebox.showwarning("Workspace", "Choose a concept workspace folder first.")
            return
        root_path = Path(root_value).expanduser()
        try:
            models_path = root_path / "models"
            logs_path = root_path / "log"
            models_path.mkdir(parents=True, exist_ok=True)
            logs_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("Workspace", f"Could not create the workspace folders:\n{exc}")
            return

        for key, value in (("output_dir", models_path), ("logging_dir", logs_path), ("convert_output_dir", models_path)):
            widget = self.entries.get(key)
            if isinstance(widget, ttk.Entry):
                widget.delete(0, tk.END)
                widget.insert(0, str(value))
        self.update_button_states()

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
        self._add_widget(
            memory_frame,
            "compile",
            "Enable Torch Compile",
            "Compiles the repeatedly used DiT blocks into optimized GPU kernels. This can speed up longer training runs after a slower first-step compilation. It requires a working Triton installation and can use additional RAM and VRAM. With Blocks to Swap, swapped linear layers are excluded from compilation, so the speed benefit may be smaller.",
            kind="checkbox",
            command=self.update_button_states,
        )
        self.hidden_frames["compile_options"] = ttk.Frame(memory_frame)
        self._add_widget(
            self.hidden_frames["compile_options"],
            "compile_backend",
            "Compile Backend:",
            "Selects how PyTorch compiles the model. 'inductor' generates optimized Triton GPU kernels and is the normal choice for speed. 'aot_eager' and 'eager' are mainly troubleshooting options and usually provide little or no speedup.",
            kind="combobox",
            options=["inductor", "aot_eager", "eager"],
        )
        self._add_widget(
            self.hidden_frames["compile_options"],
            "compile_mode",
            "Compile Mode:",
            "'default' balances compilation time and runtime speed. 'reduce-overhead' tries to reduce Python and launch overhead but may use more memory. Autotune modes spend substantially longer compiling to search for faster kernels.",
            kind="combobox",
            options=["default", "reduce-overhead", "max-autotune-no-cudagraphs", "max-autotune"],
        )
        self._add_widget(
            self.hidden_frames["compile_options"],
            "compile_dynamic",
            "Dynamic Shapes:",
            "'auto' lets PyTorch choose. 'true' makes compiled kernels accept changing bucket shapes and can reduce recompilation, but may produce slower kernels and requires MSVC C++ tools on Windows. 'false' specializes each shape for speed but recompiles when shapes change.",
            kind="combobox",
            options=["auto", "true", "false"],
        )
        self._add_widget(
            self.hidden_frames["compile_options"],
            "compile_fullgraph",
            "Compile Full Graph",
            "Forces each DiT block to compile as one uninterrupted graph. This can expose more optimization, but training fails instead of falling back when PyTorch encounters an unsupported graph break. Leave it off unless the normal compile mode is already proven stable.",
            kind="checkbox",
        )
        self._add_widget(
            self.hidden_frames["compile_options"],
            "compile_cache_size_limit",
            "Graph Cache Limit:",
            "Controls how many compiled graph variants PyTorch keeps. A value of 32 reduces repeated compilation when a dataset uses several bucket shapes, at the cost of additional system RAM and compile-cache storage.",
            kind="combobox",
            options=["32", "16", "8", "64"],
        )

        flow_frame = ttk.LabelFrame(frame, text="Flow Matching Parameters"); flow_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(flow_frame, "timestep_sampling", "Timestep Sampling:", "Method for selecting timesteps during training. 'shift' is recommended for Wan/Flux. 'krea2_shift' matches Krea 2's resolution-aware schedule.", kind='combobox', options=["uniform", "shift", "sigma", "logsnr", "qinglong_flux", "krea2_shift"])
        self._add_widget(flow_frame, "num_timestep_buckets", "Timestep Buckets:", "Enables stratified sampling by dividing timesteps into buckets. Can improve training stability, especially with small datasets. (e.g., 10)", validate_num=True)
        self.hidden_frames['timestep_boundary'] = ttk.Frame(flow_frame)
        self._add_widget(self.hidden_frames['timestep_boundary'], "timestep_boundary", "Timestep Boundary:", "The integer timestep where the model switches from low to high noise (e.g., 875). Only for combined runs.", validate_num=True)
        self._add_widget(flow_frame, "discrete_flow_shift", "Discrete Flow Shift:", "Shift value for 'shift' sampling. The documentation recommends 3.0.", validate_num=True)
        self._add_widget(flow_frame, "preserve_distribution_shape", "Preserve Distribution Shape", "Prevents distortion of the timestep distribution. Recommended when training only one model (e.g., only low noise).", kind='checkbox')

        self.hidden_frames['krea2_regularization'] = ttk.LabelFrame(frame, text="Krea 2 · Generalization (Experimental)")
        ttk.Label(
            self.hidden_frames['krea2_regularization'],
            text="Optional tools for small datasets. Use a preset for a controlled first comparison; all existing behavior stays unchanged when Off.",
            wraplength=850,
        ).pack(anchor="w", padx=8, pady=(8, 4))
        inspiration_label = ttk.Label(
            self.hidden_frames['krea2_regularization'],
            text="Inspired by BuffaloBuffaloBuffaloBuffalo's Perceptual LoRA Toolkit (independent Krea 2 adaptation).",
            wraplength=850,
        )
        inspiration_label.pack(anchor="w", padx=8, pady=(0, 4))
        ToolTip(
            inspiration_label,
            "Reference project: github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual. "
            "It documented practical LoRA experiments with weight noise and frozen depth models. "
            "This Krea 2 implementation was written independently for Musubi and is not endorsed by that project.",
        )
        preset_row = ttk.Frame(self.hidden_frames['krea2_regularization']); preset_row.pack(fill="x", padx=8, pady=(0, 6))
        preset_widget = self._add_widget(
            preset_row,
            "krea2_generalization_preset",
            "Starting Preset:",
            "Off is the exact baseline. Weight Noise Only is the safer first experiment. Balanced Experimental combines weight noise with a conservative Krea depth anchor.",
            kind='combobox',
            options=["Off (Baseline)", "Weight Noise Only", "Balanced Experimental"],
        )
        ttk.Button(preset_widget.master, text="Apply Preset", command=self._apply_krea2_generalization_preset).grid(
            row=0, column=2, padx=(6, 0)
        )
        self._add_widget(
            self.hidden_frames['krea2_regularization'],
            "krea2_weight_noise_sigma",
            "Weight Noise Strength:",
            "Adds a tiny random perturbation to LoRA/LoKr weights after each optimizer update. 0 disables it. Suggested first experiment: 0.0125; useful range reported by the reference project is roughly 0.01–0.017, but Krea 2 still needs validation.",
            kind='combobox',
            options=["0", "0.01", "0.0125", "0.015", "0.017"],
        )
        self._add_widget(
            self.hidden_frames['krea2_regularization'],
            "krea2_weight_noise_mode",
            "Noise Scaling:",
            "Relative adapts to each adapter tensor's size and is the recommended default. Absolute applies the same raw magnitude everywhere and requires careful calibration.",
            kind='combobox',
            options=["relative", "absolute"],
        )
        self._add_widget(
            self.hidden_frames['krea2_regularization'],
            "krea2_weight_noise_bound_norm",
            "Prevent Weight-Norm Drift",
            "Keeps each adapter tensor at its pre-noise norm. Recommended for long runs; usually unnecessary for short comparisons.",
            kind='checkbox',
        )
        ttk.Separator(self.hidden_frames['krea2_regularization']).pack(fill="x", padx=8, pady=6)
        self._add_widget(
            self.hidden_frames['krea2_regularization'],
            "krea2_depth_anchor_weight",
            "Depth Anchor Strength:",
            "Automatically makes a depth map for each training image, then checks whether the LoRA's predicted image has a similar 3D shape. You do not need to create depth maps yourself. This can help preserve faces, bodies, and object shapes without copying the exact colors, lighting, or background. It is slower and uses more VRAM. 0 turns it off; start with 0.01.",
            kind='combobox',
            options=["0", "0.005", "0.01", "0.025", "0.05"],
        )
        self._add_widget(
            self.hidden_frames['krea2_regularization'],
            "krea2_depth_anchor_model",
            "Depth Model:",
            "The helper model that creates the automatic depth maps. Small is recommended and downloads automatically the first time you enable depth anchoring. Base may see more detail but needs more memory and has not been calibrated for Krea 2.",
            kind='combobox',
            options=["depth-anything/Depth-Anything-V2-Small-hf", "depth-anything/Depth-Anything-V2-Base-hf"],
        )
        self._add_widget(
            self.hidden_frames['krea2_regularization'],
            "krea2_depth_anchor_input_size",
            "Depth Resolution:",
            "How much detail the automatic depth checker sees. 518 is the recommended starting point. Larger values may capture finer shapes but make training slower and use more VRAM.",
            kind='combobox',
            options=["518", "714", "980"],
        )
        self._add_widget(
            self.hidden_frames['krea2_regularization'],
            "krea2_depth_anchor_gradient_weight",
            "Edge/Shape Emphasis:",
            "How much the trainer cares about clear shape boundaries, such as a face outline, limbs, or the edge of an object. Leave this at 0.5 unless you are deliberately comparing test runs.",
            kind='combobox',
            options=["0", "0.25", "0.5", "1.0"],
        )
        self._add_widget(
            self.hidden_frames['krea2_regularization'],
            "krea2_depth_anchor_grad_checkpoint",
            "Checkpoint Depth Model",
            "Saves GPU memory while using the depth checker, at the cost of some extra processing time. Keep this enabled unless you have plenty of VRAM.",
            kind='checkbox',
            default_val=True,
        )
        face_action = ttk.Frame(self.hidden_frames['krea2_regularization']); face_action.pack(fill="x", padx=10, pady=(8, 10))
        ttk.Label(
            face_action,
            text="Face Refinement now has its own workspace for references, pose goals, Turbo evaluation, and staged-run setup.",
            wraplength=700,
        ).pack(side="left", fill="x", expand=True)
        open_face_button = ttk.Button(face_action, text="Open Face Refinement", command=lambda: self._select_page(3))
        open_face_button.pack(side="right")
        ToolTip(open_face_button, "Opens the dedicated Face Refinement workspace. Nothing starts automatically.")

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
        self._add_widget(
            other_frame,
            "rename_final_artifacts_to_epoch",
            "Rename Final Save to Epoch Number",
            "After training finishes, rename the final local LoRA file and final state folder from the upstream default names to the last epoch format, for example '-000002'. GUI-only; upstream trainer code stays unchanged.",
            kind='checkbox',
            default_val=True,
        )

        resume_frame = ttk.LabelFrame(frame, text="Resume Training"); resume_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(resume_frame, "resume_path", "Resume from State:", "Path to a saved state folder to continue a previous training run.", kind='path_entry', is_dir=True, is_path=True)
        self._add_widget(resume_frame, "network_weights", "Network Weights:", "Load pre-trained LoRA weights to continue training from them (fine-tuning a LoRA).", kind='path_entry', options=[("Weight files", "*.safetensors")], is_path=True)

    @staticmethod
    def _face_refinement_workspace_state(config):
        config = config or {}
        input_mode = config.get("input_mode", "previous_stage")
        input_lora = str(config.get("input_lora") or "").strip()
        if input_mode == "existing_lora":
            source = f"Existing LoRA: {Path(input_lora).name}" if input_lora else "Existing LoRA not selected"
        else:
            source = "LoRA from the previous staged-training step"

        report = config.get("preflight_report") or {}
        if report:
            valid = int(report.get("valid_faces") or 0)
            scanned = int(report.get("images_scanned") or 0)
            flagged = sum(1 for item in report.get("scored_images", []) if item.get("outlier"))
            references = f"Analyzed: {valid}/{scanned} usable faces; {flagged} flagged for review"
        elif str(config.get("reference_dir") or "").strip():
            references = "Reference folder selected, but analysis still needs to run"
        else:
            references = "Reference folder not selected"

        plan = config.get("pose_plan") or {}
        pose_enabled = bool(config.get("pose_aware") and plan.get("enabled"))
        enabled_poses = sum(1 for item in (plan.get("buckets") or {}).values() if item.get("enabled"))
        poses = f"Pose-aware plan active for {enabled_poses} angle group(s)" if pose_enabled else "Simple all-angle identity matching"

        baseline = str(config.get("evaluation_baseline_result") or "").strip()
        evaluation = f"Turbo baseline ready: {Path(baseline).parent.name}" if baseline and Path(baseline).is_file() else "Turbo baseline not created yet"
        configured = bool(report and str(config.get("trigger_word") or "").strip())
        return {
            "source": source,
            "references": references,
            "poses": poses,
            "evaluation": evaluation,
            "configured": configured,
        }

    def create_face_refinement_tab(self):
        frame = self._create_scrollable_tab("4  Face Refinement")
        self._add_page_intro(
            frame,
            "Face Refinement",
            "Evaluate an existing Krea 2 LoRA, analyze identity references and viewing angles, build a focused refinement plan, then run it as a staged-training step.",
        )
        self.face_workspace_mode_var = tk.StringVar()
        mode_label = ttk.Label(frame, textvariable=self.face_workspace_mode_var, style="PageHelp.TLabel", wraplength=920)
        mode_label.pack(fill="x", padx=12, pady=(2, 8))

        self.face_workspace_vars = {
            key: tk.StringVar(value="Not configured")
            for key in ("source", "references", "poses", "evaluation")
        }
        cards = (
            ("1 · Starting LoRA and identity references", "source", "Choose the LoRA, trigger word, reference folder, and face-analysis models.", "Configure Setup…", self._configure_face_refinement_from_workspace,
             "Opens all required setup fields. Saving does not start training."),
            ("2 · Analyze and review references", "references", "Face and pose analysis is run from Setup. Reopen it to rescan changed images or review low-confidence results.", "Analyze / Review…", self._configure_face_refinement_from_workspace,
             "Opens Setup, where Analyze Faces & Poses can be run again and results can be reviewed without changing source images."),
            ("3 · Pose training plan", "poses", "Optionally give weak profiles or other angles their own prompts, step shares, targets, and stopping rules.", "Configure Pose Plan…", self._open_pose_plan_from_workspace,
             "Opens the pose planner directly. Simple all-angle refinement remains available when no pose plan is enabled."),
            ("4 · Turbo evaluation", "evaluation", "Create a read-only baseline with the Turbo model before refinement, then compare the refined LoRA using identical prompts and seeds.", "Evaluate LoRA…", self._open_face_evaluation_dialog,
             "Generates and scores test images. It does not update LoRA weights."),
        )
        self._face_workspace_buttons = []
        for title, key, help_text, button_text, command, tooltip in cards:
            card = ttk.LabelFrame(frame, text=title); card.pack(fill="x", padx=10, pady=6)
            content = ttk.Frame(card); content.pack(fill="x", padx=10, pady=9)
            ttk.Label(content, textvariable=self.face_workspace_vars[key], style="Status.TLabel", wraplength=690).pack(anchor="w")
            ttk.Label(content, text=help_text, style="PageHelp.TLabel", wraplength=690).pack(anchor="w", pady=(3, 0))
            button = ttk.Button(content, text=button_text, command=command)
            button.pack(side="right", pady=(4, 0)); ToolTip(button, tooltip)
            self._face_workspace_buttons.append(button)

        results_card = ttk.LabelFrame(frame, text="Latest Turbo evaluation report")
        results_card.pack(fill="x", padx=10, pady=6)
        self.face_results_summary_var = tk.StringVar(value="Run a Turbo evaluation to see identity and pose results here.")
        ttk.Label(results_card, textvariable=self.face_results_summary_var, style="PageHelp.TLabel", wraplength=880).pack(anchor="w", padx=10, pady=(9, 6))
        ttk.Label(results_card, text="Double-click a pose row to inspect all generated images for that angle.", style="PageHelp.TLabel").pack(anchor="w", padx=10, pady=(0, 6))
        columns = ("pose", "samples", "identity", "pose_identity", "pose_success", "detection", "delta")
        self.face_results_tree = ttk.Treeview(results_card, columns=columns, show="headings", height=6)
        labels = {"pose": "Requested pose", "samples": "Samples", "identity": "Overall identity", "pose_identity": "Matching-pose identity", "pose_success": "Pose success", "detection": "Face detected", "delta": "Identity change"}
        for key in columns:
            self.face_results_tree.heading(key, text=labels[key])
            self.face_results_tree.column(key, width=125, stretch=key == "pose")
        self.face_results_tree.pack(fill="x", padx=10, pady=(0, 6))
        self.face_results_tree.bind("<Double-1>", self._open_selected_face_evaluation_pose_gallery)
        result_actions = ttk.Frame(results_card); result_actions.pack(fill="x", padx=10, pady=(0, 10))
        self.face_open_results_button = ttk.Button(result_actions, text="Open Generated Images", command=self._open_latest_face_evaluation_folder, state="disabled")
        self.face_open_results_button.pack(side="left")
        ToolTip(self.face_open_results_button, "Opens the local evaluation folder so you can visually inspect the generated Turbo images. Numbers should support visual judgment, not replace it.")
        self.face_build_plan_button = ttk.Button(result_actions, text="Build Plan from Weak Poses", command=self._build_plan_from_latest_face_evaluation, state="disabled")
        self.face_build_plan_button.pack(side="left", padx=(6, 0))
        ToolTip(self.face_build_plan_button, "Uses below-target pose scores to create an editable pose plan. It does not start training.")
        self._face_latest_evaluation_path = ""
        self._face_latest_evaluation_payload = None

        run_card = ttk.LabelFrame(frame, text="5 · Add to a run"); run_card.pack(fill="x", padx=10, pady=(6, 14))
        ttk.Label(
            run_card,
            text="Face refinement runs through Staged Progression so it can safely receive a LoRA from an earlier training stage or start from an existing LoRA.",
            style="PageHelp.TLabel", wraplength=880,
        ).pack(anchor="w", padx=10, pady=(9, 5))
        run_actions = ttk.Frame(run_card); run_actions.pack(fill="x", padx=10, pady=(0, 10))
        add_button = ttk.Button(run_actions, text="Add as Final Stage", style="Accent.TButton", command=self._add_face_refinement_final_stage)
        add_button.pack(side="left"); ToolTip(add_button, "Adds or updates one Face Refinement step at the end of the staged plan and enables Staged Progression. It does not start the run.")
        stages_button = ttk.Button(run_actions, text="Review Staged Plan…", command=self._open_staged_training_dialog)
        stages_button.pack(side="left", padx=(6, 0)); ToolTip(stages_button, "Opens the full stage editor so you can review ordering, resolutions, datasets, and step limits.")
        monitor_button = ttk.Button(run_actions, text="Open Monitor", command=lambda: self._select_page(5))
        monitor_button.pack(side="right"); ToolTip(monitor_button, "Opens the Monitor, where the staged run is started and refinement progress is shown.")
        self._face_workspace_buttons.extend((add_button, stages_button))
        self._refresh_face_refinement_workspace()

    def _refresh_face_refinement_workspace(self):
        if not hasattr(self, "face_workspace_vars"):
            return
        state = self._face_refinement_workspace_state(self._face_refinement_config)
        for key, variable in self.face_workspace_vars.items():
            variable.set(state[key])
        is_krea = self.training_mode_var.get() == "Krea 2"
        self.face_workspace_mode_var.set(
            "Ready for Krea 2. Configure and evaluate here; start the actual staged run from Monitor."
            if is_krea else "Face Refinement currently supports Krea 2 only. Switch Training mode to Krea 2 to configure or run it."
        )
        for button in getattr(self, "_face_workspace_buttons", []):
            button.configure(state="normal" if is_krea else "disabled")
        latest = str(
            (self._face_refinement_config or {}).get("evaluation_last_result")
            or (self._face_refinement_config or {}).get("evaluation_baseline_result")
            or ""
        )
        if latest and Path(latest).is_file() and latest != getattr(self, "_face_latest_evaluation_path", ""):
            self._display_face_evaluation_result(latest)

    def _display_face_evaluation_result(self, result_path):
        if not hasattr(self, "face_results_tree"):
            return
        payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
        self.face_results_tree.delete(*self.face_results_tree.get_children())
        fmt = lambda value: "—" if value is None else f"{value:.3f}"
        for pose, metrics in payload.get("poses", {}).items():
            delta = payload.get("deltas", {}).get(pose, {}).get("overall_similarity")
            self.face_results_tree.insert("", "end", iid=f"pose::{pose}", values=(
                pose.replace("_", " ").title(), metrics.get("samples", 0),
                fmt(metrics.get("overall_similarity")), fmt(metrics.get("pose_similarity")),
                f"{metrics.get('pose_success_rate', 0.0):.0%}", f"{metrics.get('detection_rate', 0.0):.0%}", fmt(delta),
            ))
        mode_text = "Comparison complete" if payload.get("baseline") else "Starting baseline created"
        self.face_results_summary_var.set(
            f"{mode_text}. Identity says who the face resembles; pose success says whether Turbo followed the requested angle. Review the generated images before deciding what to refine."
        )
        self._face_latest_evaluation_path = str(result_path)
        self._face_latest_evaluation_payload = payload
        self.face_open_results_button.configure(state="normal")
        self.face_build_plan_button.configure(state="normal" if payload.get("poses") else "disabled")

    def _open_latest_face_evaluation_folder(self):
        if self._face_latest_evaluation_path:
            self._open_path(str(Path(self._face_latest_evaluation_path).parent))

    def _build_plan_from_latest_face_evaluation(self):
        if self._face_latest_evaluation_payload:
            self._apply_turbo_evaluation_to_pose_plan(self._face_latest_evaluation_payload)

    def _open_selected_face_evaluation_pose_gallery(self, _event=None):
        if _event is not None:
            row = self.face_results_tree.identify_row(_event.y)
            if row:
                self.face_results_tree.selection_set(row)
        selected = self.face_results_tree.selection()
        if not selected or not self._face_latest_evaluation_payload:
            return
        pose = selected[0].removeprefix("pose::")
        cases = [case for case in self._face_latest_evaluation_payload.get("cases", []) if case.get("pose") == pose]
        images = [case for case in cases if case.get("image") and Path(case["image"]).is_file()]
        if not images:
            messagebox.showinfo("Evaluation images", "No generated image files were found for this pose.", parent=self.root)
            return

        dialog = tk.Toplevel(self.root)
        dialog.title(f"Evaluation Images · {pose.replace('_', ' ').title()}")
        dialog.geometry("1040x760"); dialog.minsize(720, 520)
        dialog.configure(background=self.colors["page"])
        header = ttk.Frame(dialog, padding=(14, 12)); header.pack(fill="x")
        ttk.Label(header, text=pose.replace("_", " ").title(), style="PageTitle.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text=f"{len(images)} generated image(s). Click an image for a larger preview. Identity is resemblance; detected pose is what the analyzer saw.",
            style="PageHelp.TLabel", wraplength=940,
        ).pack(anchor="w", pady=(3, 0))

        body = ttk.Frame(dialog); body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        canvas = tk.Canvas(body, bg=self.colors["page"], highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        grid = ttk.Frame(canvas, style="Page.TFrame")
        window = canvas.create_window((0, 0), window=grid, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")
        grid.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        dialog._evaluation_photos = []
        columns = 3
        for column in range(columns):
            grid.grid_columnconfigure(column, weight=1, uniform="evaluation-gallery")
        for index, case in enumerate(images):
            card = ttk.Frame(grid, style="Surface.TFrame", padding=8)
            card.grid(row=index // columns, column=index % columns, sticky="nsew", padx=6, pady=6)
            image_path = str(case["image"])
            try:
                with Image.open(image_path) as source:
                    preview = source.copy()
                preview.thumbnail((280, 250))
                photo = ImageTk.PhotoImage(preview)
                dialog._evaluation_photos.append(photo)
                image_label = tk.Label(card, image=photo, bg=self.colors["surface_alt"], cursor="hand2", bd=0)
                image_label.pack(fill="x")
                image_label.bind("<Button-1>", lambda _e, path=image_path: self._open_sample_preview(path))
            except Exception as exc:
                tk.Label(card, text=f"Preview unavailable\n{exc}", bg=self.colors["surface_alt"], fg=self.colors["muted"], height=8).pack(fill="x")
            identity = case.get("overall_similarity")
            identity_text = "Not detected" if identity is None else f"Identity: {identity:.3f}"
            detected_pose = str(case.get("actual_pose") or "not detected").replace("_", " ").title()
            ttk.Label(card, text=f"{identity_text} · Detected: {detected_pose}", style="Muted.TLabel", wraplength=280).pack(anchor="w", pady=(7, 2))
            ttk.Label(card, text=str(case.get("prompt") or ""), wraplength=280, justify="left").pack(anchor="w")

        canvas.bind("<MouseWheel>", lambda event: canvas.yview_scroll(-int(event.delta / 120) * 3, "units"))

    def _configure_face_refinement_from_workspace(self):
        self._open_face_refinement_dialog(on_save=lambda _config: self._refresh_face_refinement_workspace())

    def _open_pose_plan_from_workspace(self):
        config = self._default_face_refinement_config()
        config.update(copy.deepcopy(self._face_refinement_config or {}))

        def save(plan):
            config["pose_plan"] = plan
            config["pose_aware"] = bool(plan.get("enabled"))
            config["pose_reward_weight"] = round(min(0.35, 1.0 - float(plan.get("overall_anchor_weight", 0.80))), 3)
            self._face_refinement_config = config
            self._refresh_face_refinement_workspace()

        self._open_pose_training_plan_dialog(config, config.get("trigger_word", ""), save, self.root)

    def _add_face_refinement_final_stage(self):
        if self.training_mode_var.get() != "Krea 2":
            messagebox.showerror("Face Refinement", "Switch Training mode to Krea 2 first.")
            return
        config = self._face_refinement_config or {}
        if not config.get("preflight_report"):
            messagebox.showerror("Face Refinement", "Configure the reference folder and complete Analyze Faces & Poses first.")
            return
        stages = [copy.deepcopy(item) for item in self._staged_training_config if item.get("type") != "face_refinement"]
        stages.append({
            "label": "face-refinement", "enabled": True, "type": "face_refinement",
            "dataset_config": "", "epochs": "", "steps": str(config.get("steps") or 30),
        })
        self._staged_training_config = stages
        self.entries["use_staged_training"].var.set(True)
        self._update_staged_summary()
        self._update_run_mode_controls()
        messagebox.showinfo("Face Refinement", "Face Refinement is now the final staged step. Review the staged plan or open Monitor when you are ready; nothing has started yet.")

    def create_samples_tab(self):
        tab_frame = self._create_scrollable_tab("5  Samples")
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
        ttk.Button(prompt_btn_row, text="Preview Enabled", command=self._test_enabled_sample_prompts).pack(side="left", padx=(6, 0))
        prompt_library_button = ttk.Button(prompt_btn_row, text="Prompt Library", command=self._open_prompt_library)
        prompt_library_button.pack(side="left", padx=(6, 0))
        ToolTip(
            prompt_library_button,
            "Opens the global searchable prompt gallery. Library prompts are copied into runs; successful standalone tests add model-badged thumbnails automatically.",
        )
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

    def _merge_library_prompts(self, prompts):
        existing = {prompt_identity(prompt) for prompt in self._sample_prompts_data if isinstance(prompt, dict)}
        added = duplicates = 0
        for prompt in prompts:
            identity = prompt_identity(prompt)
            if identity in existing:
                duplicates += 1
                continue
            self._sample_prompts_data.append(copy.deepcopy(prompt))
            existing.add(identity)
            added += 1
        if added:
            self._rebuild_prompt_list()
            self.update_button_states()
        return added, duplicates

    def _open_prompt_library(self):
        if self._prompt_library_dialog is not None:
            try:
                if self._prompt_library_dialog.window.winfo_exists():
                    self._prompt_library_dialog.refresh()
                    self._prompt_library_dialog.window.deiconify()
                    self._prompt_library_dialog.window.lift()
                    return
            except tk.TclError:
                pass
        if not self.prompt_library.prompts:
            self.prompt_library.import_prompts(
                self._sample_prompts_data,
                source={"type": "initial_current_settings"},
            )
            self.prompt_library.import_jobs(self._job_history)
        self._prompt_library_dialog = PromptLibraryDialog(
            self.root,
            store=self.prompt_library,
            colors=self.colors,
            current_prompts=lambda: copy.deepcopy(self._sample_prompts_data),
            jobs=lambda: self._job_history,
            on_use=self._merge_library_prompts,
        )

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
        prompt_data = self._sample_prompts_data[idx]
        prompt_summary = prompt_data.get("prompt", "")[:120]
        self._test_sample_prompts([prompt_data], "Krea 2 Test Sample", prompt_summary)

    def _test_enabled_sample_prompts(self):
        enabled_prompts = [prompt for prompt in self._sample_prompts_data if prompt.get("enabled", True)]
        if not enabled_prompts:
            messagebox.showinfo("No Enabled Prompts", "Enable at least one sample prompt before previewing.")
            return

        note = f"{len(enabled_prompts)} enabled prompt" + ("s" if len(enabled_prompts) != 1 else "")
        self._test_sample_prompts(enabled_prompts, "Krea 2 Batch Sample Preview", note)

    def _test_sample_prompts(self, prompt_items, job_title, job_note):
        if self.current_process:
            messagebox.showwarning("Process Running", "Stop the current process before launching a test sample.")
            return

        settings = self.get_settings()
        mode = settings.get("training_mode", "Wan 2.2")
        if mode != "Krea 2":
            messagebox.showinfo("Not Available", "Sample test generation is currently implemented for Krea 2 only.")
            return

        try:
            command = self._build_krea2_test_sample_command(settings, prompt_items)
        except ValueError as e:
            messagebox.showerror("Krea 2 Test Sample", str(e))
            return

        try:
            save_path = Path(command[command.index("--save_path") + 1])
        except (ValueError, IndexError):
            save_path = None
        existing_outputs = {
            str(path.resolve()) for path in (save_path.glob("*") if save_path else [])
            if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        }
        self._sample_test_context = {
            "prompts": copy.deepcopy(prompt_items),
            "save_path": str(save_path),
            "existing_outputs": existing_outputs,
            "mode": "Krea 2 Turbo" if "--turbo" in command else "Krea 2",
            "output_name": settings.get("output_name", ""),
            "network_weights": self._resolve_krea2_preview_lora(settings),
        } if save_path else None

        self.output_text.delete("1.0", tk.END)
        self.run_status_var.set("🧪 " + job_title)
        self.progress_label_var.set("Running test generation...")
        self._begin_job("sample_test", job_title, settings=settings, note=job_note)
        self.run_process(command, on_complete=self._on_test_sample_complete, output_widget=self.output_text, job_context={"attach_to_active": True})

    def _build_krea2_test_sample_command(self, settings, prompt_items):
        required = {
            "DiT model": settings.get("krea2_dit_model"),
            "VAE model": settings.get("vae_model"),
            "Text encoder": settings.get("krea2_text_encoder"),
        }
        missing = [name for name, path in required.items() if not path or not os.path.exists(path)]
        if missing:
            raise ValueError("Missing required Krea 2 paths for test sampling:\n- " + "\n- ".join(missing))

        python_executable = sys.executable or "python"
        is_turbo = bool(settings.get("krea2_turbo_dit"))
        dit_path = settings.get("krea2_turbo_dit") if is_turbo else settings.get("krea2_dit_model")
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
            "--dit", dit_path,
            "--vae", settings["vae_model"],
            "--text_encoder", settings["krea2_text_encoder"],
            "--save_path", str(save_path),
            "--attn_mode", attn_mode,
        ]
        if is_turbo:
            command.append("--turbo")

        if not prompt_items:
            raise ValueError("No prompt data was provided for test sampling.")

        if len(prompt_items) == 1:
            prompt_data = prompt_items[0]
            command.insert(2, prompt_data.get("prompt", ""))
        else:
            prompts_file = self._write_sample_prompts_txt(
                prompt_items,
                output_name_override=f"{output_name}_sample_preview",
            )
            if not prompts_file:
                raise ValueError("Could not create the temporary prompt file for batch preview.")
            command.extend(["--from_file", prompts_file])

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

        network_weights = self._resolve_krea2_preview_lora(settings)
        if network_weights:
            command.extend(["--lora_weight", network_weights])

        if len(prompt_items) == 1:
            prompt_data = prompt_items[0]
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

    @staticmethod
    def _resolve_krea2_preview_lora(settings):
        """Prefer an explicit LoRA, otherwise use the newest checkpoint from this output run."""
        explicit = Path(str(settings.get("network_weights") or "").strip()).expanduser()
        if str(explicit) not in ("", ".") and explicit.is_file():
            return str(explicit)

        output_root = Path(str(settings.get("output_dir") or "").strip()).expanduser()
        output_name = str(settings.get("output_name") or "").strip()
        if not output_name or not output_root.is_dir():
            return ""

        candidates = []
        exact_run = output_root / output_name
        search_roots = [exact_run] if exact_run.is_dir() else []
        if not search_roots:
            search_roots = [path for path in output_root.glob(f"{output_name}*") if path.is_dir()]
        for run_dir in search_roots:
            candidates.extend(
                path for path in run_dir.glob("*.safetensors")
                if path.is_file() and "optimizer" not in path.name.lower()
            )
        if not candidates:
            return ""
        return str(max(candidates, key=lambda path: path.stat().st_mtime))

    def _on_test_sample_complete(self, return_code):
        self._finalize_active_job("completed" if return_code == 0 else ("stopped" if self._stop_requested else "failed"), return_code)
        if return_code == 0:
            self.output_text.insert(tk.END, "\n--- Test sample completed successfully. ---\n")
            captured = self._capture_sample_test_thumbnails()
            if captured:
                self.output_text.insert(
                    tk.END,
                    f"--- Added {captured} tested prompt thumbnail(s) to the global Prompt Library. ---\n",
                )
            self._refresh_sample_list()
        else:
            self.output_text.insert(tk.END, f"\n--- Test sample failed with code {return_code}. ---\n")
            self._sample_test_context = None
        self.stop_all_activity()

    def _capture_sample_test_thumbnails(self):
        context = self._sample_test_context
        self._sample_test_context = None
        if not context:
            return 0
        save_path = Path(context["save_path"])
        before = set(context.get("existing_outputs") or [])
        new_images = [
            path for path in save_path.glob("*")
            if path.is_file()
            and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            and str(path.resolve()) not in before
        ]
        new_images.sort(key=lambda path: (path.stat().st_mtime, path.name))
        captured = 0
        for prompt, image_path in zip(context.get("prompts") or [], new_images):
            metadata = {
                "test_type": "standalone_zero_step",
                "output_name": context.get("output_name", ""),
                "network_weights": context.get("network_weights", ""),
                "seed": prompt.get("seed", ""),
                "width": prompt.get("width", ""),
                "height": prompt.get("height", ""),
            }
            try:
                entry, _created = self.prompt_library.capture_thumbnail(
                    prompt,
                    image_path,
                    context.get("mode", "Krea 2"),
                    metadata=metadata,
                )
                if entry:
                    captured += 1
            except OSError as exc:
                self.output_text.insert(tk.END, f"--- Prompt Library thumbnail could not be saved: {exc} ---\n")
        if captured and self._prompt_library_dialog is not None:
            try:
                if self._prompt_library_dialog.window.winfo_exists():
                    self._prompt_library_dialog.refresh()
            except tk.TclError:
                pass
        return captured

    def _add_sample_prompt_dialog(self):
        self._open_prompt_dialog(None)

    def _edit_sample_prompt_dialog(self, idx):
        self._open_prompt_dialog(idx)

    def _open_prompt_dialog(self, idx):
        """Open a modal dialog to add/edit a sample prompt."""
        existing = self._sample_prompts_data[idx] if idx is not None else {}
        mode = self.training_mode_var.get()
        is_krea2 = mode == "Krea 2"
        is_krea2_turbo = is_krea2 and bool(self.entries.get("krea2_turbo_dit") and self.entries["krea2_turbo_dit"].get().strip())
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
            ("Steps", "steps", ("8" if is_krea2_turbo else "28") if is_krea2 else "20", 5),
            ("Guidance", "guidance", ("1.0" if is_krea2_turbo else "5.5") if is_krea2 else "5.0", 6),
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
                ("Mu", "mu", existing.get("mu", existing.get("flow_shift", "1.15" if is_krea2_turbo else "")), 6),
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

    def _serialize_sample_prompt_line(self, prompt_data, is_krea2):
        line = prompt_data.get("prompt", "")
        if prompt_data.get("width"):
            line += f" --w {prompt_data['width']}"
        if prompt_data.get("height"):
            line += f" --h {prompt_data['height']}"
        if prompt_data.get("steps"):
            line += f" --s {prompt_data['steps']}"
        if prompt_data.get("guidance"):
            # Krea 2 uses ordinary CFG; musubi's training sampler reads it from --l.
            # The standalone Krea generator accepts --g and --l equivalently.
            line += f" --{'l' if is_krea2 else 'g'} {prompt_data['guidance']}"
        if is_krea2:
            if prompt_data.get("mu"):
                line += f" --mu {prompt_data['mu']}"
            if prompt_data.get("y1"):
                line += f" --y1 {prompt_data['y1']}"
            if prompt_data.get("y2"):
                line += f" --y2 {prompt_data['y2']}"
        else:
            if prompt_data.get("frames"):
                line += f" --f {prompt_data['frames']}"
            if prompt_data.get("flow_shift"):
                line += f" --fs {prompt_data['flow_shift']}"
            if prompt_data.get("cfg_scale"):
                line += f" --l {prompt_data['cfg_scale']}"
        if prompt_data.get("seed"):
            line += f" --d {prompt_data['seed']}"
        if prompt_data.get("neg"):
            line += f" --n {prompt_data['neg']}"
        if not is_krea2 and prompt_data.get("image_path"):
            line += f" --i {prompt_data['image_path']}"
        return line

    def _write_sample_prompts_txt(self, prompts_data, output_name_override=None):
        """Serialise prompt dicts to a .txt file next to the dataset config, return path or ''."""
        if not prompts_data:
            return ""
        mode = self.training_mode_var.get()
        is_krea2 = mode == "Krea 2"
        lines = []
        for prompt_data in prompts_data:
            if not prompt_data.get("enabled", True):
                continue
            lines.append(self._serialize_sample_prompt_line(prompt_data, is_krea2))
        if not lines:
            return ""
        # Save next to the dataset config (always outside the repo, always exists)
        output_name = output_name_override or self.entries["output_name"].get().strip() or "training"
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

    def _build_sample_prompts_txt(self):
        """Serialise enabled sample prompts for training to a .txt file, return path or ''."""
        return self._write_sample_prompts_txt(self._sample_prompts_data)

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
        tab_frame = ttk.Frame(self.notebook); self.notebook.add(tab_frame, text="6  Monitor")
        layout_toolbar = ttk.Frame(tab_frame)
        layout_toolbar.pack(fill="x", padx=10, pady=(8, 2))
        ttk.Label(
            layout_toolbar,
            text="Drag the horizontal divider to resize controls and output.",
            style="PageHelp.TLabel",
        ).pack(side="left", fill="x", expand=True)
        self.monitor_top_toggle_btn = ttk.Button(
            layout_toolbar,
            text="Hide Controls",
            command=self._toggle_monitor_top_pane,
        )
        self.monitor_top_toggle_btn.pack(side="right")
        ToolTip(
            self.monitor_top_toggle_btn,
            "Hides or restores the controls and live metrics so the loss graph and console can use more screen space.",
        )

        self.monitor_vertical_pane = ttk.PanedWindow(tab_frame, orient=tk.VERTICAL)
        self.monitor_vertical_pane.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.monitor_top_pane = ttk.Frame(self.monitor_vertical_pane)
        self.monitor_bottom_pane = ttk.Frame(self.monitor_vertical_pane)
        self.monitor_vertical_pane.add(self.monitor_top_pane, weight=0)
        self.monitor_vertical_pane.add(self.monitor_bottom_pane, weight=1)

        top_pane = ttk.Frame(self.monitor_top_pane)
        top_pane.pack(fill='both', expand=True, pady=(6, 4))
        controls_frame = ttk.LabelFrame(top_pane, text="Controls & Caching"); controls_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        self.run_status_var = tk.StringVar(value="⚪ New Training RUN")
        self.run_status_label = ttk.Label(controls_frame, textvariable=self.run_status_var, style='Status.TLabel')
        self.run_status_label.pack(pady=5, padx=10)
        cache_opts_frame = ttk.Frame(controls_frame)
        cache_opts_frame.pack(pady=5, padx=10, fill='x')
        self._add_widget(cache_opts_frame, "recache_latents", "Re-cache Latents Before Training", "If your dataset or VAE changes, check this to force regeneration of the latent cache.", kind='checkbox')
        self._add_widget(cache_opts_frame, "recache_text", "Re-cache Text Encoders Before Training", "If your dataset or T5 model changes, check this to force regeneration of the text encoder cache.", kind='checkbox')

        staged_frame = ttk.LabelFrame(controls_frame, text="Run Mode")
        staged_frame.pack(fill="x", padx=10, pady=(4, 8))
        self._add_widget(
            staged_frame,
            "use_staged_training",
            "Use Staged Progression",
            "When enabled, the Run button executes each configured stage in order and resumes the complete training state between stages. When disabled, the Run button performs one normal training run using the main form settings.",
            kind="checkbox",
            command=self._update_run_mode_controls,
        )
        staged_plan_frame = ttk.Frame(staged_frame)
        staged_plan_frame.pack(fill="x", padx=10, pady=(0, 8))
        self.staged_summary_var = tk.StringVar(value="No staged run configured")
        ttk.Label(staged_plan_frame, textvariable=self.staged_summary_var, style="PageHelp.TLabel").pack(side="left", fill="x", expand=True)
        self.staged_config_btn = ttk.Button(staged_plan_frame, text="Configure Stages…", command=self._open_staged_training_dialog)
        self.staged_config_btn.pack(side="right", padx=(8, 0))
        ToolTip(
            self.staged_config_btn,
            "Opens the stage plan editor. Standard stages select a dataset TOML; Krea Face Refinement stages use their separate saved reference/prompt settings. Configuring a plan does not run it until Staged Progression is enabled and Run is pressed.",
        )

        train_button_frame = ttk.Frame(controls_frame); train_button_frame.pack(pady=10, padx=10, fill='x')
        self.start_btn = ttk.Button(train_button_frame, text="Run Training", style="Accent.TButton", command=self.start_selected_run); self.start_btn.pack(side="left", padx=(0, 5), expand=True, fill='x')
        ToolTip(self.start_btn, "Starts either one normal training run or the configured staged progression, according to the Run Mode selection above.")
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
        self.depth_anchor_status_var = tk.StringVar(value="Depth anchor: Off")
        self.depth_anchor_status_label = ttk.Label(monitor_frame, textvariable=self.depth_anchor_status_var, style="PageHelp.TLabel")
        self.depth_anchor_status_label.pack(anchor="w", padx=10, pady=(4, 0))
        ToolTip(self.depth_anchor_status_label, "When enabled, this shows the raw depth-consistency loss and its weighted contribution to the training objective. A changing finite value confirms the depth checker is participating in training.")
        ttk.Button(monitor_frame, text="Generate Command", command=self.show_command).pack(pady=(10,5), padx=10, fill='x')

        bottom_pane_host = ttk.Frame(self.monitor_bottom_pane)
        bottom_pane_host.pack(fill='both', expand=True, pady=(4, 0))

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
        console_output_frame = ttk.Frame(console_frame)
        console_output_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        console_output_frame.grid_rowconfigure(0, weight=1)
        console_output_frame.grid_columnconfigure(0, weight=1)
        console_output_frame.grid_columnconfigure(1, minsize=18)
        self.output_text = tk.Text(console_output_frame, wrap=tk.WORD, height=14, bg=self.colors["field"], fg=self.colors["text"], insertbackground=self.colors["text"], selectbackground=self.colors["selection"], font=('Consolas', 9), relief=tk.FLAT, bd=0, padx=8, pady=6)
        output_scrollbar = ttk.Scrollbar(
            console_output_frame,
            orient="vertical",
            command=self.output_text.yview,
            style="Console.Vertical.TScrollbar",
        )
        self.output_text.configure(yscrollcommand=output_scrollbar.set)
        self.output_text.grid(row=0, column=0, sticky="nsew")
        output_scrollbar.grid(row=0, column=1, sticky="ns", padx=(3, 0))
        self.root.after_idle(self._restore_monitor_splitter)

    def _restore_monitor_splitter(self):
        try:
            if not self._monitor_top_collapsed and len(self.monitor_vertical_pane.panes()) > 1:
                available = max(160, self.monitor_vertical_pane.winfo_height() - 180)
                self.monitor_vertical_pane.sashpos(
                    0,
                    min(max(140, self._monitor_top_sash_position), available),
                )
        except (AttributeError, tk.TclError):
            pass

    def _toggle_monitor_top_pane(self):
        try:
            panes = self.monitor_vertical_pane.panes()
            top_path = str(self.monitor_top_pane)
            if top_path in panes:
                if len(panes) > 1:
                    self._monitor_top_sash_position = self.monitor_vertical_pane.sashpos(0)
                self.monitor_vertical_pane.forget(self.monitor_top_pane)
                self._monitor_top_collapsed = True
                self.monitor_top_toggle_btn.configure(text="Show Controls")
            else:
                self.monitor_vertical_pane.insert(0, self.monitor_top_pane, weight=0)
                self._monitor_top_collapsed = False
                self.monitor_top_toggle_btn.configure(text="Hide Controls")
                self.root.after_idle(self._restore_monitor_splitter)
        except (AttributeError, tk.TclError):
            pass

    def _update_run_mode_controls(self):
        try:
            staged = self.entries["use_staged_training"].var.get()
            self.start_btn.configure(text="Run Staged Training" if staged else "Run Training")
            control_state = "disabled" if self.current_process else "normal"
            self.entries["use_staged_training"].configure(state=control_state)
            self.staged_config_btn.configure(state=control_state)
        except (AttributeError, KeyError):
            pass

    def start_selected_run(self):
        if self.current_process:
            return
        if self.entries["use_staged_training"].var.get():
            self.start_staged_training()
        else:
            self.start_training()

    @staticmethod
    def _default_face_refinement_config():
        from musubi_tuner.face_refinement.face_models import default_model_dir
        from musubi_tuner.face_refinement.pose_plan import default_pose_plan

        return {
            "input_mode": "previous_stage", "input_lora": "", "trigger_word": "", "excluded_reference_images": [],
            "reference_dir": "", "face_model_dir": str(default_model_dir()),
            "prompts": [
                "portrait photo of {trigger}, natural expression, soft daylight",
                "close-up portrait of {trigger} smiling",
                "photo of {trigger} looking to the side",
                "photo of {trigger} outdoors, candid expression",
                "studio portrait of {trigger}, neutral background",
                "low-angle photo of {trigger}",
                "photo of {trigger} laughing",
                "cinematic portrait of {trigger} in dramatic lighting",
            ],
            "steps": 30, "resolution": 512, "denoise_steps": 12, "draft_k": 1,
            "cfg_scale": 5.5, "learning_rate": 1e-4, "target_similarity": 0.45,
            "stop_similarity": 0.55, "early_stop_patience": 5,
            "min_detection_rate": 0.25, "anti_copy_weight": 0.02,
            "preview_every": 5, "save_every": 10, "qkvo_only": True,
            "checkpoint_vae": True, "license_acknowledged": False,
            "pose_aware": False, "pose_reward_weight": 0.20, "pose_min_references": 2,
            "pose_plan": default_pose_plan(),
            "evaluation_prompts_per_pose": 1, "evaluation_seeds_per_prompt": 2,
            "evaluation_seed": 42000, "evaluation_resolution": 512, "evaluation_steps": 8,
            "evaluation_lora_strength": 1.0,
            "evaluation_baseline_result": "", "evaluation_last_result": "",
            "blocks_to_swap": 10,
            "gpu_id": "auto",
        }

    def _open_face_refinement_dialog(self, on_save=None):
        config = self._default_face_refinement_config()
        config.update(self._face_refinement_config or {})
        dialog = tk.Toplevel(self.root)
        dialog.title("Krea 2 · Face Refinement (Experimental)")
        dialog.transient(self.root); dialog.grab_set(); dialog.minsize(760, 720)
        canvas = tk.Canvas(dialog, highlightthickness=0)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        host = ttk.Frame(canvas, padding=16)
        host.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=host, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")

        ttk.Label(host, text="Face identity refinement", style="PageTitle.TLabel").pack(anchor="w")
        ttk.Label(
            host,
            text="A final-stage polish: Krea generates temporary images from prompts and the LoRA is rewarded when their faces resemble your references. It does not use a dataset TOML.",
            style="PageHelp.TLabel", wraplength=700,
        ).pack(anchor="w", pady=(3, 12))

        source = ttk.LabelFrame(host, text="1 · Starting LoRA"); source.pack(fill="x", pady=6)
        input_mode_var = tk.StringVar(value=config.get("input_mode", "previous_stage"))
        input_lora_var = tk.StringVar(value=config.get("input_lora", ""))
        trigger_var = tk.StringVar(value=config.get("trigger_word", ""))
        ttk.Radiobutton(source, text="Use the LoRA produced by the previous stage", variable=input_mode_var, value="previous_stage").pack(anchor="w", padx=8, pady=(7, 2))
        ttk.Radiobutton(source, text="Refine an existing Krea 2 LoRA", variable=input_mode_var, value="existing_lora").pack(anchor="w", padx=8, pady=2)
        lora_row = ttk.Frame(source); lora_row.pack(fill="x", padx=8, pady=5)
        ttk.Label(lora_row, text="Existing LoRA", width=22).pack(side="left")
        input_lora_entry = ttk.Entry(lora_row, textvariable=input_lora_var); input_lora_entry.pack(side="left", fill="x", expand=True)
        input_lora_button = ttk.Button(lora_row, text="Browse", command=lambda: input_lora_var.set(filedialog.askopenfilename(parent=dialog, filetypes=[("LoRA weights", "*.safetensors")]) or input_lora_var.get()))
        input_lora_button.pack(side="right", padx=(6, 0))
        trigger_row = ttk.Frame(source); trigger_row.pack(fill="x", padx=8, pady=(2, 7))
        ttk.Label(trigger_row, text="Trigger word", width=22).pack(side="left")
        ttk.Entry(trigger_row, textvariable=trigger_var).pack(side="left", fill="x", expand=True)
        ttk.Label(source, text="The trigger is inserted into {trigger} prompts and prefixed to prompts that omit it.", style="PageHelp.TLabel").pack(anchor="w", padx=8, pady=(0, 7))
        def sync_input_mode(*_args):
            state = "normal" if input_mode_var.get() == "existing_lora" else "disabled"
            input_lora_entry.configure(state=state); input_lora_button.configure(state=state)
        input_mode_var.trace_add("write", sync_input_mode); sync_input_mode()

        identity = ttk.LabelFrame(host, text="2 · Reference identity"); identity.pack(fill="x", pady=6)
        reference_var = tk.StringVar(value=config["reference_dir"])
        model_var = tk.StringVar(value=config["face_model_dir"])
        status_var = tk.StringVar(value="Analyze Faces & Poses before adding this stage.")
        def path_row(parent, label, variable, directory=True):
            row = ttk.Frame(parent); row.pack(fill="x", padx=8, pady=5)
            ttk.Label(row, text=label, width=22).pack(side="left")
            ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)
            ttk.Button(row, text="Browse", command=lambda: variable.set(filedialog.askdirectory(parent=dialog) or variable.get())).pack(side="right", padx=(6, 0))
        path_row(identity, "Reference image folder", reference_var)
        path_row(identity, "AntelopeV2 folder", model_var)
        ttk.Label(identity, textvariable=status_var, wraplength=680).pack(anchor="w", padx=8, pady=(3, 6))

        license_var = tk.BooleanVar(value=bool(config.get("license_acknowledged")))
        ttk.Checkbutton(
            identity, variable=license_var,
            text="I understand AntelopeV2 model files are third-party artifacts with separate terms and are not bundled with this GUI.",
        ).pack(anchor="w", padx=8, pady=4)

        def run_background(action, success):
            def worker():
                try:
                    result = action()
                    self.root.after(0, lambda: success(result))
                except Exception as exc:
                    self.root.after(0, lambda exc=exc: messagebox.showerror("Face Refinement", str(exc), parent=dialog))
            threading.Thread(target=worker, daemon=True).start()

        identity_actions = ttk.Frame(identity); identity_actions.pack(fill="x", padx=8, pady=(4, 8))
        def download_models():
            if not license_var.get():
                messagebox.showwarning("Face-model terms", "Acknowledge the third-party model notice before downloading.", parent=dialog); return
            if not messagebox.askokcancel("Download AntelopeV2", "Download approximately 220 MB of third-party face-analysis model files from Hugging Face?", parent=dialog): return
            status_var.set("Downloading face models…")
            from musubi_tuner.face_refinement.face_models import ensure_models
            run_background(lambda: ensure_models(model_var.get()), lambda _r: status_var.set("Face models are ready. Click Analyze Faces & Poses."))
        def face_check():
            if not self._check_face_refinement_dependencies(parent=dialog):
                return
            status_var.set("Checking reference faces…")
            from musubi_tuner.face_refinement.preflight import scan_reference_faces
            def show(report):
                warnings = " ".join(report["warnings"]) or "No obvious reference-set problems found."
                outliers = sum(1 for item in report.get("scored_images", []) if item.get("outlier"))
                status_var.set(f"Usable faces: {report['valid_faces']}/{report['images_scanned']}. Mean identity consistency: {report['similarity_mean']:.2f}. Flagged for review: {outliers}. {warnings}")
                config["preflight_report"] = report
                review_button.configure(state="normal")
            run_background(lambda: scan_reference_faces(reference_var.get(), model_var.get()), show)
        def review_references():
            report = config.get("preflight_report") or {}
            if not report:
                messagebox.showinfo("Reference review", "Click Analyze Faces & Poses first.", parent=dialog); return
            review = tk.Toplevel(dialog)
            review.title("Review Face References")
            review.transient(dialog)
            review.minsize(900, 560)
            review.configure(background=self.colors["page"])
            ttk.Label(review, text="Detected identity outliers and unusable face references", style="PageTitle.TLabel").pack(anchor="w", padx=12, pady=(12, 2))
            ttk.Label(
                review,
                text="Low scores can indicate another person, a bad detection, or an extreme angle. ‘No usable face’ means the method could not create a reliable full-face identity embedding—often because the image is a mouth/nose crop, an extreme profile, too small, blurred, or occluded. It is not used here, but may still be useful for normal LoRA training.",
                style="PageHelp.TLabel", wraplength=850,
            ).pack(anchor="w", padx=12, pady=(0, 8))
            split = ttk.Panedwindow(review, orient="horizontal"); split.pack(fill="both", expand=True, padx=12)
            table_host = ttk.Frame(split)
            preview_host = tk.Frame(split, bg=self.colors["page"], bd=0, highlightthickness=0)
            split.add(table_host, weight=4); split.add(preview_host, weight=2)
            tree = ttk.Treeview(table_host, columns=("use", "kind", "score", "pose", "angles", "confidence", "file"), show="headings", selectmode="browse")
            for key, label, width in (("use", "Use", 48), ("kind", "Result", 90), ("score", "Identity", 70), ("pose", "Pose bucket", 135), ("angles", "Yaw / Pitch", 95), ("confidence", "Confidence", 75), ("file", "File", 260)):
                tree.heading(key, text=label); tree.column(key, width=width, stretch=key == "file")
            scroll = ttk.Scrollbar(table_host, orient="vertical", command=tree.yview); tree.configure(yscrollcommand=scroll.set)
            tree.pack(side="left", fill="both", expand=True); scroll.pack(side="right", fill="y")
            excluded = set(config.get("excluded_reference_images") or [])
            records = {}
            for index, item in enumerate(report.get("scored_images", [])):
                path = item["path"]; flagged = bool(item.get("outlier")); use = path not in excluded
                iid = f"face-{index}"; records[iid] = {"path": path, "scored": True, "item": item}
                tree.insert("", "end", iid=iid, values=("Yes" if use else "No", "Review" if flagged else "Detected", f"{item['similarity']:.3f}", item.get("bucket", "uncertain"), f"{item.get('yaw', 0):.0f}° / {item.get('pitch', 0):.0f}°", f"{item.get('confidence', 0):.2f}", Path(path).name), tags=("outlier",) if flagged else ())
            for index, item in enumerate(report.get("skipped_images", [])):
                path = item["path"]; iid = f"skip-{index}"; records[iid] = {"path": path, "scored": False}
                tree.insert("", "end", iid=iid, values=("No", "No usable face", "—", "no_face", "—", "—", Path(path).name))
            tree.tag_configure("outlier", foreground="#d97706")
            preview_label = tk.Label(
                preview_host,
                text="Select an image",
                anchor="center",
                bg=self.colors["page"],
                fg=self.colors["muted"],
                bd=0,
                highlightthickness=0,
                relief="flat",
            )
            preview_label.pack(fill="both", expand=True, padx=10, pady=10)
            preview_host._photo = None
            def selected_record():
                selected = tree.selection(); return records.get(selected[0]) if selected else None
            def show_preview(_event=None):
                record = selected_record()
                if not record: return
                try:
                    from PIL import Image, ImageTk
                    image = Image.open(record["path"]); image.thumbnail((320, 400))
                    preview_host._photo = ImageTk.PhotoImage(image)
                    preview_label.configure(image=preview_host._photo, text="")
                except Exception as exc:
                    preview_label.configure(image="", text=f"Preview unavailable:\n{exc}")
            def toggle_use(_event=None):
                selected = tree.selection()
                if not selected: return
                iid = selected[0]; record = records[iid]
                if not record["scored"]: return
                values = list(tree.item(iid, "values")); path = record["path"]
                if values[0] == "Yes": values[0] = "No"; excluded.add(path)
                else: values[0] = "Yes"; excluded.discard(path)
                tree.item(iid, values=values)
                config["excluded_reference_images"] = sorted(excluded)
            def change_pose():
                selected = tree.selection()
                if not selected or not records[selected[0]]["scored"]: return
                iid = selected[0]; record = records[iid]
                from musubi_tuner.face_refinement.pose import POSE_BUCKETS, POSE_LABELS
                picker = tk.Toplevel(review)
                picker.title("Correct Viewing Angle")
                picker.transient(review); picker.grab_set(); picker.resizable(False, False)
                picker.configure(background=self.colors["page"])
                panel = ttk.Frame(picker, padding=16); panel.pack(fill="both", expand=True)
                ttk.Label(panel, text="Correct viewing angle", style="PageTitle.TLabel").pack(anchor="w")
                ttk.Label(
                    panel,
                    text="Choose the angle you see. This changes only the analysis label; it does not edit or move the image.",
                    style="PageHelp.TLabel", wraplength=430, justify="left",
                ).pack(anchor="w", pady=(3, 10))
                ttk.Label(panel, text=Path(record["path"]).name, style="Muted.TLabel", wraplength=430).pack(anchor="w", pady=(0, 8))
                display_to_pose = {POSE_LABELS.get(pose, pose.replace("_", " ").title()): pose for pose in POSE_BUCKETS}
                pose_to_display = {pose: label for label, pose in display_to_pose.items()}
                pose_display_var = tk.StringVar(value=pose_to_display.get(record["item"].get("bucket", "uncertain"), "Uncertain"))
                pose_picker = ttk.Combobox(panel, textvariable=pose_display_var, values=list(display_to_pose), state="readonly", width=34)
                pose_picker.pack(fill="x", pady=(0, 12))
                ToolTip(pose_picker, "Choose the viewing angle you see in the image. Use Uncertain when the face is ambiguous, mirrored, heavily tilted, or hard to classify.")
                def apply_pose():
                    pose = display_to_pose[pose_display_var.get()]
                    record["item"]["bucket"] = pose; record["item"]["pose_manual"] = True
                    values = list(tree.item(iid, "values")); values[3] = POSE_LABELS.get(pose, pose); tree.item(iid, values=values)
                    picker.destroy()
                actions = ttk.Frame(panel); actions.pack(fill="x")
                ttk.Button(actions, text="Cancel", command=picker.destroy).pack(side="right", padx=(6, 0))
                ttk.Button(actions, text="Apply Angle", style="Accent.TButton", command=apply_pose).pack(side="right")
            tree.bind("<<TreeviewSelect>>", show_preview); tree.bind("<Double-1>", toggle_use)
            buttons = ttk.Frame(review); buttons.pack(fill="x", padx=12, pady=10)
            use_button = ttk.Button(buttons, text="Use / Exclude Selected", command=toggle_use); use_button.pack(side="left"); ToolTip(use_button, "Switch whether this detected face is used during refinement. Excluding it does not delete, move, or alter the image.")
            correct_button = ttk.Button(buttons, text="Correct Pose…", command=change_pose); correct_button.pack(side="left", padx=(6, 0)); ToolTip(correct_button, "Fix the automatic angle group when it looks wrong. This changes only the virtual training manifest, not the source folder.")
            ttk.Button(buttons, text="Open Image", command=lambda: self._open_path(selected_record()["path"]) if selected_record() else None).pack(side="left", padx=6)
            ttk.Button(buttons, text="Open Folder", command=lambda: self._open_path(str(Path(selected_record()["path"]).parent)) if selected_record() else None).pack(side="left")
            ttk.Button(buttons, text="Close", command=review.destroy).pack(side="right")
        download_button = ttk.Button(identity_actions, text="Download Face Models…", command=download_models); download_button.pack(side="left")
        ToolTip(download_button, "Downloads the optional face detector and identity model after confirmation. This is needed once; the files remain in the selected model folder.")
        analyze_button = ttk.Button(identity_actions, text="Analyze Faces & Poses", command=face_check); analyze_button.pack(side="left", padx=6)
        ToolTip(analyze_button, "Safely reads the selected folder again and recalculates usable faces, identity scores, head angles, confidence, and pose buckets. It does not train, move, rename, or edit any image. Run it again after changing the folder or its images.")
        review_button = ttk.Button(identity_actions, text="Review Results…", command=review_references, state="normal" if config.get("preflight_report") else "disabled")
        review_button.pack(side="left")
        ToolTip(review_button, "Review low identity matches, skipped detail crops, automatic pose groups, and confidence. You can exclude images or correct a pose group without changing files on disk.")

        prompts_frame = ttk.LabelFrame(host, text="3 · Generation prompts"); prompts_frame.pack(fill="both", expand=True, pady=6)
        ttk.Label(prompts_frame, text="One prompt per line. Use {trigger} for the subject. Optional pose tags: [auto], [frontal], [three_quarter_left/right], [profile_left/right], [looking_up/down].", wraplength=680).pack(anchor="w", padx=8, pady=(7, 3))
        prompts_text = self._themed_text(prompts_frame, height=9, wrap="word")
        prompts_text.pack(fill="both", expand=True, padx=8, pady=(0, 8)); prompts_text.insert("1.0", "\n".join(config["prompts"]))
        prompt_source_var = tk.StringVar(value="These prompts are used by simple face refinement.")
        prompt_source_label = ttk.Label(prompts_frame, textvariable=prompt_source_var, style="PageHelp.TLabel", wraplength=680)
        prompt_source_label.pack(anchor="w", padx=8, pady=(0, 8))

        settings_frame = ttk.LabelFrame(host, text="4 · Safe starting settings"); settings_frame.pack(fill="x", pady=6)
        variables = {}
        fields = [
            ("steps", "Refinement steps", int), ("resolution", "Generation resolution", int),
            ("denoise_steps", "Denoising steps", int), ("draft_k", "Differentiable final steps", int),
            ("learning_rate", "Learning rate", float), ("target_similarity", "Reward saturation", float),
            ("stop_similarity", "Early-stop similarity", float), ("early_stop_patience", "Early-stop patience", int),
            ("min_detection_rate", "Minimum face detection rate", float), ("preview_every", "Preview every N steps", int),
            ("save_every", "Save LoRA every N steps", int),
            ("blocks_to_swap", "DiT blocks moved to CPU", int),
            ("gpu_id", "GPU index (auto recommended)", str),
            ("pose_reward_weight", "Matching-pose influence", float),
            ("pose_min_references", "Minimum refs per pose", int),
        ]
        field_help = {
            "pose_reward_weight": "How much of each update may focus on the matching viewing angle. 0.20 is a cautious default. The Pose Training Plan's identity anchor may reduce it further for safety.",
            "pose_min_references": "Minimum number of usable photos required before an angle gets its own identity target. Groups below this number safely fall back or are disabled.",
            "save_every": "Saves an intermediate LoRA after this many refinement steps. For example, 10 saves at steps 10, 20, and 30. Use 0 to disable intermediate checkpoints. The final LoRA is always saved.",
        }
        grid = ttk.Frame(settings_frame); grid.pack(fill="x", padx=8, pady=8)
        for index, (key, label, _kind) in enumerate(fields):
            variable = tk.StringVar(value=str(config[key])); variables[key] = variable
            row, column = divmod(index, 2)
            cell = ttk.Frame(grid); cell.grid(row=row, column=column, sticky="ew", padx=5, pady=3)
            field_label = ttk.Label(cell, text=label, width=26); field_label.pack(side="left"); field_entry = ttk.Entry(cell, textvariable=variable, width=12); field_entry.pack(side="right")
            if key in field_help: ToolTip(field_label, field_help[key]); ToolTip(field_entry, field_help[key])
        grid.columnconfigure(0, weight=1); grid.columnconfigure(1, weight=1)
        qkvo_var = tk.BooleanVar(value=config["qkvo_only"]); checkpoint_var = tk.BooleanVar(value=config["checkpoint_vae"]); pose_aware_var = tk.BooleanVar(value=config.get("pose_aware", False))
        ttk.Checkbutton(settings_frame, text="Train attention Q/K/V/O adapters only (recommended anti-overfit safeguard)", variable=qkvo_var).pack(anchor="w", padx=12)
        ttk.Checkbutton(settings_frame, text="Checkpoint VAE decode to save VRAM", variable=checkpoint_var).pack(anchor="w", padx=12, pady=(0, 8))
        pose_checkbox = ttk.Checkbutton(settings_frame, text="Use pose-aware identity matching (experimental; optional)", variable=pose_aware_var); pose_checkbox.pack(anchor="w", padx=12, pady=(0, 8))
        ToolTip(pose_checkbox, "When enabled, profile prompts can compare against profile references and other angles against their matching groups. Leave it off for the original all-angles identity method.")
        def save_pose_plan(plan):
            config["pose_plan"] = plan; pose_aware_var.set(bool(plan.get("enabled")))
            variables["pose_reward_weight"].set(str(round(min(0.35, 1.0 - float(plan.get("overall_anchor_weight", 0.80))), 3)))
        plan_button = ttk.Button(settings_frame, text="Configure Pose Training Plan…", command=lambda: self._open_pose_training_plan_dialog(config, trigger_var.get(), save_pose_plan, dialog)); plan_button.pack(anchor="w", padx=12, pady=(0, 8))
        ToolTip(plan_button, "Opens the guided pose planner: choose an improvement goal, decide how often each angle is practiced, set finish targets, and edit pose-specific prompts. It does not start training.")
        def sync_prompt_source(*_args):
            plan_active = bool(pose_aware_var.get() and (config.get("pose_plan") or {}).get("enabled"))
            prompts_text.configure(
                state="disabled" if plan_active else "normal",
                foreground=self.colors["muted"] if plan_active else self.colors["text"],
            )
            if plan_active:
                prompt_source_var.set("Pose Training Plan is active. This main prompt list is preserved but not used. Open Configure Pose Training Plan to edit the prompts used for this run.")
                prompt_source_label.configure(foreground=self.colors["warning"])
            else:
                prompt_source_var.set("This main prompt list is active. Enable and save a Pose Training Plan only when you want separate prompts and targets for each viewing angle.")
                prompt_source_label.configure(foreground=self.colors["muted"])
        pose_aware_var.trace_add("write", sync_prompt_source); sync_prompt_source()
        ToolTip(prompt_source_label, "Only one prompt source is used per run. An enabled Pose Training Plan uses its grouped pose tabs; otherwise this main list is used.")

        actions = ttk.Frame(host); actions.pack(fill="x", pady=(12, 0))
        def save():
            try:
                updated = dict(config)
                updated["input_mode"] = input_mode_var.get(); updated["input_lora"] = input_lora_var.get().strip(); updated["trigger_word"] = trigger_var.get().strip()
                updated["reference_dir"] = reference_var.get().strip(); updated["face_model_dir"] = model_var.get().strip()
                updated["prompts"] = [line.strip() for line in prompts_text.get("1.0", "end").splitlines() if line.strip()]
                for key, _label, kind in fields: updated[key] = kind(variables[key].get())
                updated["qkvo_only"] = qkvo_var.get(); updated["checkpoint_vae"] = checkpoint_var.get(); updated["pose_aware"] = pose_aware_var.get(); updated["license_acknowledged"] = license_var.get()
                updated["pose_plan"] = copy.deepcopy(config.get("pose_plan") or self._default_face_refinement_config()["pose_plan"])
                updated["pose_plan"]["enabled"] = updated["pose_aware"]
                if updated["pose_aware"]:
                    from musubi_tuner.face_refinement.pose import parse_pose_prompt
                    for prompt in updated["prompts"]: parse_pose_prompt(prompt)
                if not updated["prompts"] or not os.path.isdir(updated["reference_dir"]): raise ValueError("Choose a reference folder and provide at least one prompt.")
                if not updated["license_acknowledged"]: raise ValueError("Acknowledge the third-party face-model notice.")
                if updated["input_mode"] == "existing_lora":
                    from musubi_tuner.face_refinement.lora_validation import validate_krea2_lora
                    updated["input_lora_report"] = validate_krea2_lora(updated["input_lora"])
                if not updated.get("preflight_report") or updated["preflight_report"].get("reference_dir") != str(Path(updated["reference_dir"]).resolve()):
                    raise ValueError("Run Analyze Faces & Poses successfully for this reference folder before saving.")
                if updated["steps"] < 1 or updated["resolution"] % 16 or not 1 <= updated["draft_k"] <= updated["denoise_steps"] or not 0 <= updated["blocks_to_swap"] <= 26: raise ValueError("Invalid step count, resolution, differentiable-step value, or blocks-to-swap value.")
                if updated["save_every"] < 0: raise ValueError("Save LoRA every N steps must be 0 or greater.")
                if not 0 <= updated["pose_reward_weight"] <= 0.35 or updated["pose_min_references"] < 2: raise ValueError("Pose influence must be 0–0.35 and each pose bucket must require at least 2 references.")
                if updated["gpu_id"] != "auto" and (not updated["gpu_id"].isdigit() or int(updated["gpu_id"]) < 0): raise ValueError("GPU index must be 'auto' or a non-negative number.")
            except Exception as exc:
                messagebox.showerror("Face Refinement", str(exc), parent=dialog); return
            self._face_refinement_config = updated
            if on_save: on_save(updated)
            dialog.destroy()
        ttk.Button(actions, text="Cancel", command=dialog.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(actions, text="Save Settings", style="Accent.TButton", command=save).pack(side="right")

    def _open_face_evaluation_dialog(self):
        if self.current_process:
            messagebox.showwarning("Turbo Evaluation", "Stop the active process before starting an evaluation."); return
        config = self._default_face_refinement_config(); config.update(copy.deepcopy(self._face_refinement_config or {}))
        dialog = tk.Toplevel(self.root); dialog.title("Evaluate Starting LoRA with Krea Turbo"); dialog.transient(self.root); dialog.grab_set(); dialog.minsize(720, 480)
        host = ttk.Frame(dialog, padding=16); host.pack(fill="both", expand=True)
        ttk.Label(host, text="Turbo face and pose baseline", style="PageTitle.TLabel").pack(anchor="w")
        ttk.Label(host, text="This generates images with the Turbo model users actually render with. It never trains or changes the LoRA. Fixed prompts and seeds make later comparisons fair.", style="PageHelp.TLabel", wraplength=680).pack(anchor="w", pady=(3, 12))
        mode_var = tk.StringVar(value="compare" if config.get("evaluation_baseline_result") and Path(config["evaluation_baseline_result"]).is_file() else "baseline")
        create_radio = ttk.Radiobutton(host, text="Create or replace the starting baseline", variable=mode_var, value="baseline"); create_radio.pack(anchor="w")
        compare_radio = ttk.Radiobutton(host, text="Compare this LoRA with the saved baseline", variable=mode_var, value="compare"); compare_radio.pack(anchor="w", pady=(2, 8))
        ToolTip(create_radio, "Use this before refinement. The GUI automatically saves the baseline results inside the local evaluation folder and remembers that file for later.")
        ToolTip(compare_radio, "Use this after refinement or for an intermediate checkpoint. It repeats the baseline's exact prompts, seeds, and Turbo settings and shows the changes.")
        lora_var = tk.StringVar(value=config.get("input_lora", "")); baseline_var = tk.StringVar(value=config.get("evaluation_baseline_result", ""))
        def path_row(label, variable, browse_command):
            row = ttk.Frame(host); row.pack(fill="x", pady=4); label_widget = ttk.Label(row, text=label, width=22); label_widget.pack(side="left"); entry = ttk.Entry(row, textvariable=variable); entry.pack(side="left", fill="x", expand=True); button = ttk.Button(row, text="Browse", command=browse_command); button.pack(side="right", padx=(6, 0)); return label_widget, entry, button
        lora_label, lora_entry, lora_button = path_row("LoRA to evaluate", lora_var, lambda: lora_var.set(filedialog.askopenfilename(parent=dialog, filetypes=[("LoRA weights", "*.safetensors")]) or lora_var.get()))
        ToolTip(lora_label, "The existing or refined LoRA being tested. Evaluation reads it but never changes it."); ToolTip(lora_entry, "The existing or refined LoRA being tested. Evaluation reads it but never changes it.")
        trigger_text = str(config.get("trigger_word") or "").strip()
        trigger_label = ttk.Label(host, text=f"Trigger used in evaluation prompts: {trigger_text or '(none configured)'}", style="PageHelp.TLabel", wraplength=680)
        trigger_label.pack(anchor="w", pady=(0, 4))
        ToolTip(trigger_label, "The GUI replaces {trigger} with this text before generation. If a prompt has no {trigger}, it prefixes the trigger automatically. Check Configure Setup to change it.")
        baseline_label, baseline_entry, baseline_button = path_row("Baseline result", baseline_var, lambda: baseline_var.set(filedialog.askopenfilename(parent=dialog, filetypes=[("Evaluation result", "results.json"), ("JSON", "*.json")]) or baseline_var.get()))
        baseline_help_var = tk.StringVar(); baseline_help = ttk.Label(host, textvariable=baseline_help_var, style="PageHelp.TLabel", wraplength=680); baseline_help.pack(anchor="w", pady=(0, 5))
        def sync_baseline_mode(*_args):
            comparing = mode_var.get() == "compare"
            baseline_entry.configure(state="normal" if comparing else "disabled"); baseline_button.configure(state="normal" if comparing else "disabled")
            baseline_help_var.set("Choose the results.json created by the starting baseline." if comparing else "No path is needed. The GUI will automatically create and remember results.json after this baseline finishes.")
        mode_var.trace_add("write", sync_baseline_mode); sync_baseline_mode()
        baseline_tip = "This is not an output folder. It is the results.json automatically produced by the original baseline. You only choose it when comparing another LoRA."
        ToolTip(baseline_label, baseline_tip); ToolTip(baseline_entry, baseline_tip); ToolTip(baseline_button, baseline_tip); ToolTip(baseline_help, baseline_tip)
        settings_frame = ttk.LabelFrame(host, text="Evaluation size"); settings_frame.pack(fill="x", pady=10)
        fields = [("evaluation_prompts_per_pose", "Prompts per enabled pose", int, "How many different prompt descriptions to test for each enabled viewing angle."), ("evaluation_seeds_per_prompt", "Seeds per prompt", int, "How many image variations to generate for each prompt. More seeds give a more trustworthy average but take longer."), ("evaluation_resolution", "Turbo resolution", int, "Square image size used for both baseline and later comparison. Keep it identical between runs."), ("evaluation_steps", "Turbo denoising steps", int, "Turbo normally uses 8 steps. Keep this identical for before/after comparisons."), ("evaluation_seed", "Starting seed", int, "The fixed seed family used to reproduce exactly the same test images after refinement."), ("evaluation_lora_strength", "LoRA strength", float, "How strongly the selected LoRA is applied to Turbo. 1.0 is the normal full strength. Use the same value you normally render with; comparisons keep it fixed." )]
        variables = {}; grid = ttk.Frame(settings_frame); grid.pack(fill="x", padx=8, pady=8)
        for index, (key, label, kind, tip) in enumerate(fields):
            variable = tk.StringVar(value=str(config.get(key))); variables[key] = variable
            row, column = divmod(index, 2); cell = ttk.Frame(grid); cell.grid(row=row, column=column, sticky="ew", padx=5, pady=3)
            label_widget = ttk.Label(cell, text=label, width=24); label_widget.pack(side="left"); entry = ttk.Entry(cell, textvariable=variable, width=10); entry.pack(side="right"); ToolTip(label_widget, tip); ToolTip(entry, tip)
        grid.columnconfigure(0, weight=1); grid.columnconfigure(1, weight=1)
        ttk.Label(host, text="Generation count depends on enabled pose groups × prompts × seeds. Turbo weights, projector settings, LoRA strength, prompts, seeds, and resolution are held constant in comparisons.", style="PageHelp.TLabel", wraplength=680).pack(anchor="w", pady=6)
        actions = ttk.Frame(host); actions.pack(fill="x", pady=(12, 0))
        def run():
            try:
                from musubi_tuner.face_refinement.lora_validation import validate_krea2_lora
                validate_krea2_lora(lora_var.get())
                if not config.get("preflight_report"): raise ValueError("Run Analyze Faces & Poses and save Face Refinement settings first.")
                for key, _label, kind, _tip in fields: config[key] = kind(variables[key].get())
                if min(config["evaluation_prompts_per_pose"], config["evaluation_seeds_per_prompt"], config["evaluation_resolution"], config["evaluation_steps"]) < 1: raise ValueError("Evaluation counts, resolution, and steps must be positive.")
                if config["evaluation_lora_strength"] <= 0: raise ValueError("LoRA strength must be greater than zero.")
                baseline = baseline_var.get().strip() if mode_var.get() == "compare" else None
                if mode_var.get() == "compare" and (not baseline or not Path(baseline).is_file()): raise ValueError("Choose an existing baseline results.json before comparing.")
                settings = self.get_settings(); settings["python_executable"] = sys.executable or "python"
                prepared = krea2_face_eval_backend.prepare(settings, config, lora_var.get(), baseline_result=baseline, label=mode_var.get())
            except Exception as exc: messagebox.showerror("Turbo Evaluation", str(exc), parent=dialog); return
            config["input_lora"] = lora_var.get().strip(); config["input_mode"] = "existing_lora"
            self._face_refinement_config = config
            settings["face_refinement_config"] = copy.deepcopy(config)
            self._face_eval_context = {**prepared, "mode": mode_var.get(), "config": config, "commands": list(prepared["commands"])}
            self.output_text.delete("1.0", tk.END); self._select_page(5); self.run_status_var.set("🧪 Turbo face baseline evaluation")
            self.progress_label_var.set(f"Generating {prepared['cases']} fixed Turbo evaluation image(s)…")
            self._begin_job("sample_test", "Krea Turbo face evaluation", settings=settings, note=f"{mode_var.get()} · {prepared['cases']} fixed cases")
            dialog.destroy(); self.run_process(prepared["commands"][0], on_complete=self._on_face_eval_generation_complete, output_widget=self.output_text, job_context={"attach_to_active": True})
        cancel = ttk.Button(actions, text="Cancel", command=dialog.destroy); cancel.pack(side="right", padx=(6, 0)); run_button = ttk.Button(actions, text="Run Turbo Evaluation", style="Accent.TButton", command=run); run_button.pack(side="right"); ToolTip(run_button, "Starts read-only Turbo image generation, then scores those images against the saved face and pose references. No optimizer or training code is used.")

    def _on_face_eval_generation_complete(self, return_code):
        context = self._face_eval_context
        if return_code != 0 or not context:
            self._finalize_active_job("failed", return_code); self._face_eval_context = None; self.stop_all_activity(); return
        self.progress_label_var.set("Scoring generated faces and requested poses…")
        self.run_process(context["commands"][1], on_complete=self._on_face_eval_scoring_complete, output_widget=self.output_text, job_context={"attach_to_active": True})

    def _on_face_eval_scoring_complete(self, return_code):
        context = self._face_eval_context; self._face_eval_context = None
        if return_code != 0 or not context or not Path(context["result"]).is_file():
            self._finalize_active_job("failed", return_code); self.stop_all_activity(); return
        if context["mode"] == "baseline":
            self._face_refinement_config["evaluation_baseline_result"] = str(context["result"])
        self._face_refinement_config["evaluation_last_result"] = str(context["result"])
        self._finalize_active_job("completed", 0); self.stop_all_activity(); self._show_face_evaluation_results(context["result"])

    def _show_face_evaluation_results(self, result_path):
        self._display_face_evaluation_result(result_path)
        self._refresh_face_refinement_workspace()
        self._select_page(3)
        self.run_status_var.set("✅ Turbo face evaluation complete — review the report in Face Refinement")

    def _apply_turbo_evaluation_to_pose_plan(self, payload, results_dialog=None):
        from musubi_tuner.face_refinement.pose_plan import TRAINABLE_POSES, default_pose_plan, normalize_pose_plan
        config = self._default_face_refinement_config(); config.update(copy.deepcopy(self._face_refinement_config or {}))
        plan = copy.deepcopy(config.get("pose_plan") or default_pose_plan("custom")); plan["enabled"] = True; plan["preset"] = "custom"
        scores = {}
        for pose in TRAINABLE_POSES:
            cfg = plan["buckets"][pose]; metrics = payload.get("poses", {}).get(pose)
            if not metrics:
                cfg["enabled"] = False; cfg["share"] = 0; continue
            identity = metrics.get("pose_similarity") if metrics.get("pose_similarity") is not None else metrics.get("overall_similarity")
            target = float(cfg.get("target", 0.55)); pose_failure = 1.0 - float(metrics.get("pose_success_rate", 0.0))
            weakness = max(0.0, target - float(identity or 0.0)) + 0.20 * pose_failure
            cfg["enabled"] = weakness > 0.01; cfg["share"] = weakness * 100.0; scores[pose] = weakness
        if not any(cfg.get("enabled") for cfg in plan["buckets"].values()):
            messagebox.showinfo("Turbo Evaluation", "Every evaluated pose already meets its current target and pose-following threshold. No weak-pose plan was created.", parent=results_dialog); return
        counts = (config.get("preflight_report") or {}).get("pose_bucket_counts", {})
        try: plan, warnings = normalize_pose_plan(plan, counts, int(config.get("pose_min_references", 2)))
        except ValueError as exc: messagebox.showerror("Turbo Evaluation", str(exc), parent=results_dialog); return
        config["pose_plan"] = plan; config["pose_aware"] = True; config["evaluation_baseline_result"] = str(payload.get("baseline") or config.get("evaluation_baseline_result") or "")
        self._face_refinement_config = config
        if results_dialog: results_dialog.destroy()
        self._refresh_face_refinement_workspace()
        self._select_page(3)
        messagebox.showinfo("Pose plan created", "A pose-aware plan was built from the weak evaluation results. Review or edit it with Configure Pose Plan before training.", parent=self.root)
        if warnings: messagebox.showwarning("Pose plan safeguards", "\n".join(warnings), parent=self.root)

    def _open_pose_training_plan_dialog(self, face_config, trigger_word, on_save, parent=None):
        from musubi_tuner.face_refinement.pose import POSE_LABELS, parse_pose_prompt
        from musubi_tuner.face_refinement.pose_plan import (
            TRAINABLE_POSES, apply_preset, default_pose_plan, normalize_pose_plan, suggest_prompts,
        )

        plan = copy.deepcopy(face_config.get("pose_plan") or default_pose_plan())
        dialog = tk.Toplevel(parent or self.root); dialog.title("Pose Training Plan"); dialog.transient(parent or self.root); dialog.grab_set(); dialog.minsize(1040, 720)
        host = ttk.Frame(dialog, padding=14); host.pack(fill="both", expand=True)
        ttk.Label(host, text="Pose Training Plan", style="PageTitle.TLabel").pack(anchor="w")
        ttk.Label(host, text="Choose a goal, review its sampling balance, and edit prompts by pose. Overall identity always remains the anchor.", style="PageHelp.TLabel", wraplength=980).pack(anchor="w", pady=(2, 10))

        top = ttk.LabelFrame(host, text="1 · Goal and prompt variations"); top.pack(fill="x", pady=(0, 8))
        preset_var = tk.StringVar(value=plan.get("preset", "balanced_identity"))
        preset_names = {"Balanced identity": "balanced_identity", "Improve side profiles": "improve_profiles", "Improve three-quarter views": "improve_three_quarter", "Custom": "custom"}
        reverse_presets = {value: key for key, value in preset_names.items()}
        preset_display = tk.StringVar(value=reverse_presets.get(preset_var.get(), "Custom"))
        row = ttk.Frame(top); row.pack(fill="x", padx=8, pady=7)
        goal_label = ttk.Label(row, text="Training goal", width=20); goal_label.pack(side="left")
        preset_box = ttk.Combobox(row, textvariable=preset_display, values=list(preset_names), state="readonly", width=30); preset_box.pack(side="left")
        ToolTip(goal_label, "Choose what you mainly want to improve. A preset fills in sensible pose percentages; you can still change every value below.")
        ToolTip(preset_box, "Balanced spreads practice across available angles. Profile and three-quarter presets spend more training steps on those views. Custom keeps your own choices.")
        anchor_var = tk.StringVar(value=str(plan.get("overall_anchor_weight", 0.80)))
        anchor_label = ttk.Label(row, text="Overall identity anchor", padding=(18, 0, 6, 0)); anchor_label.pack(side="left")
        anchor_entry = ttk.Entry(row, textvariable=anchor_var, width=8); anchor_entry.pack(side="left")
        anchor_tip = "How strongly training must keep the person's general identity while improving one angle. Keep 0.80 unless you have a reason to change it. Higher is safer but makes pose-specific changes gentler."
        ToolTip(anchor_label, anchor_tip); ToolTip(anchor_entry, anchor_tip)
        variation_vars = {name: tk.BooleanVar(value=name in plan.get("variations", [])) for name in ("natural", "studio", "cinematic", "expression")}
        variations_row = ttk.Frame(top); variations_row.pack(fill="x", padx=8, pady=(0, 7)); variation_label = ttk.Label(variations_row, text="Prompt idea styles", width=20); variation_label.pack(side="left")
        ToolTip(variation_label, "Controls which kinds of editable prompt ideas the ‘Create Prompt Ideas’ button adds. This does not analyze or caption your images.")
        variation_tips = {"natural": "Adds daylight and realistic-photo ideas.", "studio": "Adds clean-background and soft studio-light ideas.", "cinematic": "Adds dramatic photographic-lighting ideas.", "expression": "Adds natural-expression and candid-photo ideas."}
        for name, variable in variation_vars.items():
            checkbox = ttk.Checkbutton(variations_row, text=name.title(), variable=variable); checkbox.pack(side="left", padx=(0, 8)); ToolTip(checkbox, variation_tips[name])

        body = ttk.Panedwindow(host, orient="vertical"); body.pack(fill="both", expand=True)
        table_frame = ttk.LabelFrame(body, text="2 · Per-pose targets and stopping"); prompts_frame = ttk.LabelFrame(body, text="3 · Prompts by pose")
        body.add(table_frame, weight=2); body.add(prompts_frame, weight=3)
        headers = ("Train", "Pose", "References", "Prompt share %", "Target", "Target patience", "Plateau patience", "Min evaluations")
        header_tips = (
            "Enable this viewing angle. Disabled rows receive no prompts or pose-specific stopping goal.",
            "The viewing angle being practiced.",
            "How many enabled reference photos were placed in this angle bucket. Too few references disables the bucket safely.",
            "Approximately what percentage of training steps should practice this angle. Enabled shares are automatically adjusted to total 100%.",
            "The face-similarity score this angle should reach before it can be considered good enough.",
            "How many successful checks in a row must meet the target before this angle is finished.",
            "How many checks without meaningful improvement count as being stuck. Training may stop instead of wasting steps.",
            "Do not make a stop decision for this angle until it has been checked at least this many times.",
        )
        for column, label in enumerate(headers):
            header = ttk.Label(table_frame, text=label); header.grid(row=0, column=column, padx=4, pady=4, sticky="w"); ToolTip(header, header_tips[column])
        bucket_counts = (face_config.get("preflight_report") or {}).get("pose_bucket_counts", {})
        row_vars = {}; prompt_boxes = {}
        prompt_tabs = ttk.Notebook(prompts_frame); prompt_tabs.pack(fill="both", expand=True, padx=6, pady=6)
        for index, pose in enumerate(TRAINABLE_POSES, start=1):
            cfg = plan.setdefault("buckets", {}).setdefault(pose, {})
            variables = {
                "enabled": tk.BooleanVar(value=cfg.get("enabled", False)), "share": tk.StringVar(value=str(round(float(cfg.get("share", 0)), 2))),
                "target": tk.StringVar(value=str(cfg.get("target", 0.55))), "patience": tk.StringVar(value=str(cfg.get("patience", 2))),
                "plateau_patience": tk.StringVar(value=str(cfg.get("plateau_patience", 4))), "min_evaluations": tk.StringVar(value=str(cfg.get("min_evaluations", 2))),
            }; row_vars[pose] = variables
            enabled_widget = ttk.Checkbutton(table_frame, variable=variables["enabled"]); enabled_widget.grid(row=index, column=0, padx=4); ToolTip(enabled_widget, f"Include {POSE_LABELS[pose].lower()} images and prompts in this training plan.")
            pose_label = ttk.Label(table_frame, text=POSE_LABELS[pose]); pose_label.grid(row=index, column=1, padx=4, sticky="w"); ToolTip(pose_label, "This is a virtual group created by Analyze Faces & Poses. Your source images are never moved.")
            count = int(bucket_counts.get(pose, 0)); count_label = ttk.Label(table_frame, text=str(count), foreground=self.colors["warning"] if count < int(face_config.get("pose_min_references", 2)) else self.colors["success"]); count_label.grid(row=index, column=2, padx=4); ToolTip(count_label, f"{count} detected reference face(s) are currently assigned to this angle. Orange means the group may be too small.")
            for column, key, width in ((3, "share", 10), (4, "target", 8), (5, "patience", 10), (6, "plateau_patience", 11), (7, "min_evaluations", 10)):
                entry = ttk.Entry(table_frame, textvariable=variables[key], width=width); entry.grid(row=index, column=column, padx=4, pady=2); ToolTip(entry, header_tips[column])
            tab = ttk.Frame(prompt_tabs); prompt_tabs.add(tab, text=POSE_LABELS[pose])
            text = self._themed_text(tab, height=6, wrap="word"); text.pack(fill="both", expand=True, padx=6, pady=(6, 3)); text.insert("1.0", "\n".join(cfg.get("prompts") or suggest_prompts(pose)))
            ToolTip(text, f"One editable {POSE_LABELS[pose].lower()} prompt per line. The [{pose}] label tells the trainer which reference-angle group to compare against; it is removed before Krea reads the prompt.")
            prompt_boxes[pose] = text
            actions = ttk.Frame(tab); actions.pack(fill="x", padx=6, pady=(0, 6))
            def generate(target=pose, box=text):
                variations = [name for name, variable in variation_vars.items() if variable.get()]
                suggestions = suggest_prompts(target, trigger_word.strip() or "{trigger}", variations)
                existing = [line.strip() for line in box.get("1.0", "end").splitlines() if line.strip()]
                for suggestion in suggestions:
                    if suggestion not in existing: existing.append(suggestion)
                box.delete("1.0", "end"); box.insert("1.0", "\n".join(existing))
            ideas_button = ttk.Button(actions, text="Create Pose Prompt Ideas…", command=generate); ideas_button.pack(side="left")
            ToolTip(ideas_button, "Adds a few editable prompt ideas for this angle using the styles checked above. It keeps your existing lines and skips exact duplicates. It does not inspect your images, call AI, or replace the trigger word system.")

        def read_rows():
            for pose, variables in row_vars.items():
                cfg = plan["buckets"][pose]
                cfg.update({"enabled": variables["enabled"].get(), "share": float(variables["share"].get()), "target": float(variables["target"].get()), "patience": int(variables["patience"].get()), "plateau_patience": int(variables["plateau_patience"].get()), "min_evaluations": int(variables["min_evaluations"].get())})
                if cfg["share"] < 0 or not -1.0 <= cfg["target"] <= 1.0 or min(cfg["patience"], cfg["plateau_patience"], cfg["min_evaluations"]) < 1:
                    raise ValueError(f"{POSE_LABELS[pose]} needs a non-negative share, target from -1 to 1, and stopping counts of at least 1.")
                cfg["prompts"] = [line.strip() for line in prompt_boxes[pose].get("1.0", "end").splitlines() if line.strip()]
                for prompt in cfg["prompts"]:
                    tag, _ = parse_pose_prompt(prompt)
                    if tag != pose: raise ValueError(f"Every {POSE_LABELS[pose]} prompt must begin with [{pose}].")
        def apply_selected_preset(_event=None):
            selected = preset_names[preset_display.get()]; preset_var.set(selected)
            if selected == "custom": return
            read_rows(); apply_preset(plan, selected)
            for pose, variables in row_vars.items():
                cfg = plan["buckets"][pose]; variables["enabled"].set(cfg["enabled"]); variables["share"].set(str(round(cfg["share"], 2)))
        preset_box.bind("<<ComboboxSelected>>", apply_selected_preset)

        def import_prompts():
            path = filedialog.askopenfilename(parent=dialog, filetypes=[("Prompt files", "*.txt *.json"), ("All files", "*.*")])
            if not path: return
            try:
                raw = Path(path).read_text(encoding="utf-8")
                if Path(path).suffix.lower() == ".json":
                    payload = json.loads(raw); lines = payload.get("prompts", payload) if isinstance(payload, dict) else payload
                else: lines = raw.splitlines()
                imported = 0
                for line in lines:
                    prompt = str(line).strip()
                    if not prompt: continue
                    pose, _ = parse_pose_prompt(prompt)
                    if pose in prompt_boxes:
                        box = prompt_boxes[pose]; current = box.get("1.0", "end").strip().splitlines()
                        if prompt not in current: box.insert("end", ("\n" if current and current != [""] else "") + prompt); imported += 1
                messagebox.showinfo("Prompt import", f"Imported {imported} pose-tagged prompt(s). Untagged lines were left unchanged.", parent=dialog)
            except Exception as exc: messagebox.showerror("Prompt import", str(exc), parent=dialog)

        footer = ttk.Frame(host); footer.pack(fill="x", pady=(8, 0)); import_button = ttk.Button(footer, text="Import Pose-Tagged Prompts…", command=import_prompts); import_button.pack(side="left")
        ToolTip(import_button, "Imports a TXT or JSON prompt list when its lines already begin with labels such as [profile_left]. Each line goes into the matching tab; unlabelled lines are ignored here.")
        def save():
            try:
                read_rows(); plan["enabled"] = True; plan["preset"] = preset_var.get(); plan["overall_anchor_weight"] = float(anchor_var.get()); plan["variations"] = [name for name, variable in variation_vars.items() if variable.get()]
                if not 0.65 <= plan["overall_anchor_weight"] <= 1.0: raise ValueError("Overall identity anchor must be between 0.65 and 1.0.")
                normalized, warnings = normalize_pose_plan(plan, bucket_counts, int(face_config.get("pose_min_references", 2)))
                if warnings and not messagebox.askyesno("Sparse pose buckets", "\n".join(warnings) + "\n\nSave the safely normalized plan?", parent=dialog): return
            except Exception as exc: messagebox.showerror("Pose Training Plan", str(exc), parent=dialog); return
            on_save(normalized); dialog.destroy()
        ttk.Button(footer, text="Cancel", command=dialog.destroy).pack(side="right", padx=(6, 0)); save_button = ttk.Button(footer, text="Save Plan", style="Accent.TButton", command=save); save_button.pack(side="right"); ToolTip(save_button, "Checks the plan, safely disables pose groups with too few references, adjusts enabled percentages to total 100%, and returns to Face Refinement settings. Training does not start.")

    def _open_staged_training_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Staged Resolution Training")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(True, True)
        dialog.minsize(1080, 380)

        host = ttk.Frame(dialog, padding=16)
        host.pack(fill="both", expand=True)
        host.grid_columnconfigure(0, weight=1)
        ttk.Label(host, text="Resolution progression", style="PageTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            host,
            text="Standard stages use a dataset TOML. Krea Face Refinement is a separate prompt-and-reference stage and uses its saved GUI settings.",
            style="PageHelp.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(3, 14))

        table = ttk.Frame(host)
        table.grid(row=2, column=0, sticky="nsew")
        table.grid_columnconfigure(3, weight=1)
        ttk.Label(table, text="Use").grid(row=0, column=0, sticky="w")
        ttk.Label(table, text="Stage Label").grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(table, text="Stage Type").grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Label(table, text="Dataset TOML").grid(row=0, column=3, sticky="w", padx=(8, 0))
        ttk.Label(table, text="Epochs").grid(row=0, column=4, sticky="w", padx=(8, 0))
        ttk.Label(table, text="Max Steps").grid(row=0, column=5, sticky="w", padx=(8, 0))

        rows = []
        next_row_index = [1]
        current_dataset = self.entries["dataset_config"].get().strip()
        initial_stages = self._staged_training_config or [
            {
                "label": label,
                "type": "standard",
                "enabled": True,
                "dataset_config": current_dataset if label == "256" else "",
                "epochs": self.entries["max_train_epochs"].get() or "1",
                "steps": "",
            }
            for label in ("256", "512", "1024")
        ]

        def add_stage_row(saved=None):
            saved = saved or {}
            row_index = next_row_index[0]
            next_row_index[0] += 1
            enabled = tk.BooleanVar(value=saved.get("enabled", True))
            label_var = tk.StringVar(value=str(saved.get("label", f"stage-{row_index}")))
            type_var = tk.StringVar(value=saved.get("type", "standard"))
            path_var = tk.StringVar(value=saved.get("dataset_config", ""))
            epochs_var = tk.StringVar(value=str(saved.get("epochs", self.entries["max_train_epochs"].get() or "1")))
            steps_var = tk.StringVar(value=str(saved.get("steps", "")))
            ttk.Checkbutton(table, variable=enabled).grid(row=row_index, column=0, sticky="w", pady=4)
            ttk.Entry(table, textvariable=label_var, width=14).grid(row=row_index, column=1, sticky="ew", padx=(8, 0), pady=4)
            type_widget = ttk.Combobox(table, textvariable=type_var, width=18, state="readonly", values=["standard", "face_refinement"])
            type_widget.grid(row=row_index, column=2, sticky="ew", padx=(8, 0), pady=4)
            path_host = ttk.Frame(table)
            path_host.grid(row=row_index, column=3, sticky="ew", padx=(8, 0), pady=4)
            path_entry = ttk.Entry(path_host, textvariable=path_var)
            path_entry.pack(side="left", fill="x", expand=True)

            def browse(var=path_var):
                selected = filedialog.askopenfilename(filetypes=[("TOML files", "*.toml")])
                if selected:
                    var.set(selected)

            ttk.Button(path_host, text="Browse", command=browse).pack(side="right", padx=(5, 0))
            epochs_entry = ttk.Entry(table, textvariable=epochs_var, width=8)
            epochs_entry.grid(row=row_index, column=4, sticky="ew", padx=(8, 0), pady=4)
            steps_entry = ttk.Entry(table, textvariable=steps_var, width=10)
            steps_entry.grid(row=row_index, column=5, sticky="ew", padx=(8, 0), pady=4)

            configure_button = ttk.Button(table, text="Settings…")
            configure_button.grid(row=row_index, column=6, sticky="ew", padx=(8, 0), pady=4)

            def sync_limit_fields(*_args, steps=steps_var, epochs_widget=epochs_entry, steps_widget=steps_entry, path_widget=path_entry, browse_parent=path_host, stage_type=type_var, button=configure_button):
                face = stage_type.get() == "face_refinement"
                epochs_widget.configure(state="disabled" if face or steps.get().strip() else "normal")
                steps_widget.configure(state="normal")
                path_widget.configure(state="disabled" if face else "normal")
                for child in browse_parent.winfo_children()[1:]: child.configure(state="disabled" if face else "normal")
                button.configure(state="normal" if face else "disabled")
                if face:
                    configured_steps = str((self._face_refinement_config or self._default_face_refinement_config())["steps"])
                    if steps.get() != configured_steps:
                        steps.set(configured_steps)

            steps_var.trace_add("write", sync_limit_fields)
            type_var.trace_add("write", sync_limit_fields)
            configure_button.configure(command=lambda: self._open_face_refinement_dialog(on_save=lambda cfg: steps_var.set(str(cfg["steps"]))) )
            sync_limit_fields()

            row = {
                "enabled": enabled,
                "label": label_var,
                "type": type_var,
                "path": path_var,
                "epochs": epochs_var,
                "steps": steps_var,
                "widgets": [],
            }

            def remove_stage(target=row):
                if target not in rows:
                    return
                for widget in target["widgets"]:
                    widget.destroy()
                rows.remove(target)

            remove_button = ttk.Button(table, text="Remove", command=remove_stage)
            remove_button.grid(row=row_index, column=7, sticky="ew", padx=(8, 0), pady=4)
            row["widgets"] = [
                table.grid_slaves(row=row_index, column=column)[0]
                for column in range(8)
            ]
            rows.append(row)

        for stage in initial_stages:
            add_stage_row(stage)

        ttk.Button(host, text="+ Add Stage", command=add_stage_row).grid(row=3, column=0, sticky="w", pady=(10, 0))

        recache_var = tk.BooleanVar(value=getattr(self, "_staged_recache_latents", True))
        ttk.Checkbutton(
            host,
            text="Re-cache latents before every stage (recommended when resolution changes)",
            variable=recache_var,
        ).grid(row=4, column=0, sticky="w", pady=(12, 0))

        actions = ttk.Frame(host)
        actions.grid(row=5, column=0, sticky="e", pady=(18, 0))

        def save_and_close():
            configured = []
            seen_labels = set()
            invalid_label_chars = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
            for row in rows:
                label = row["label"].get().strip()
                label_key = self._stage_label_text({"label": label}).casefold()
                if not label or invalid_label_chars.search(label) or label.endswith((".", " ")):
                    messagebox.showerror(
                        "Staged Run",
                        "Every stage needs a filename-safe label without < > : \" / \\ | ? * or a trailing dot/space.",
                        parent=dialog,
                    )
                    return
                if label_key in seen_labels:
                    messagebox.showerror("Staged Run", f"Stage labels must be unique: {label}", parent=dialog)
                    return
                seen_labels.add(label_key)

                enabled = row["enabled"]
                stage_type = row["type"].get()
                path_var = row["path"]
                steps_text = row["steps"].get().strip()
                epochs_text = "" if steps_text or stage_type == "face_refinement" else row["epochs"].get().strip()
                if not enabled.get():
                    configured.append({
                        "label": label,
                        "enabled": False,
                        "type": stage_type,
                        "dataset_config": path_var.get().strip(),
                        "epochs": epochs_text,
                        "steps": steps_text,
                    })
                    continue
                path = path_var.get().strip()
                try:
                    limit = int(steps_text or epochs_text)
                except ValueError:
                    field_name = "steps" if steps_text else "epochs"
                    messagebox.showerror("Staged Run", f"{label} {field_name} must be a whole number.", parent=dialog)
                    return
                if limit < 1 or (stage_type == "standard" and not os.path.isfile(path)):
                    limit_name = "step" if steps_text else "epoch"
                    requirement = "an existing TOML file and " if stage_type == "standard" else ""
                    messagebox.showerror("Staged Run", f"{label} needs {requirement}at least 1 {limit_name}.", parent=dialog)
                    return
                if stage_type == "face_refinement" and not self._face_refinement_config.get("preflight_report"):
                    messagebox.showerror("Staged Run", f"Configure {label} and complete Analyze Faces & Poses first.", parent=dialog)
                    return
                configured.append({
                    "label": label,
                    "enabled": True,
                    "type": stage_type,
                    "dataset_config": path if stage_type == "standard" else "",
                    "epochs": "" if steps_text or stage_type == "face_refinement" else str(limit),
                    "steps": str(limit) if steps_text else "",
                })
            if not any(item["enabled"] for item in configured):
                messagebox.showerror("Staged Run", "Enable at least one stage.", parent=dialog)
                return
            enabled_plan = [item for item in configured if item["enabled"]]
            face_positions = [index for index, item in enumerate(enabled_plan) if item.get("type") == "face_refinement"]
            if face_positions and face_positions[-1] != len(enabled_plan) - 1:
                if not messagebox.askyesno(
                    "Face Refinement ordering",
                    "A standard-training stage comes after Face Refinement. That later SFT stage starts a fresh optimizer from the refined LoRA and may weaken the identity gain.\n\nKeep this order?",
                    parent=dialog,
                ):
                    return
            self._staged_training_config = configured
            self._staged_recache_latents = recache_var.get()
            self._update_staged_summary()
            dialog.destroy()

        ttk.Button(actions, text="Cancel", command=dialog.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(actions, text="Save Plan", style="Accent.TButton", command=save_and_close).pack(side="right")
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)

    def _update_staged_summary(self):
        enabled = [item for item in self._staged_training_config if item.get("enabled")]
        if not enabled:
            self.staged_summary_var.set("No staged run configured")
            return
        self.staged_summary_var.set(" → ".join(f"{self._stage_label_text(item)} × {self._staged_limit_text(item)}" for item in enabled))

    @staticmethod
    def _staged_limit_text(stage):
        steps = str(stage.get("steps", "")).strip()
        if steps:
            return f"{steps} steps"
        return f"{stage.get('epochs', '')} epochs"

    @staticmethod
    def _stage_label_text(stage):
        label = str(stage.get("label", "")).strip()
        return f"{label}px" if label.isdigit() else label

    def _copy_console_output(self):
        text = self.output_text.get("1.0", tk.END).strip()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def create_jobs_tab(self):
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="7  Jobs")
        tab_frame.grid_columnconfigure(0, weight=1)
        tab_frame.grid_rowconfigure(2, weight=1)

        intro = ttk.Frame(tab_frame, style="Page.TFrame")
        intro.grid(row=0, column=0, sticky="ew", padx=12, pady=(14, 5))
        ttk.Label(intro, text="Recent jobs", style="PageTitle.TLabel").pack(anchor="w")
        ttk.Label(
            intro,
            text="Track completed, failed, and stopped runs locally. Run name identifies what was produced; Job type explains what operation was run. Right-click a job to repeat/edit its settings or load its latest state as a continuation.",
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
        columns = ("status", "mode", "started", "progress", "run_name", "job_type")
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
            "run_name": "Run name",
            "job_type": "Job type",
        }
        widths = {"status": 96, "mode": 92, "started": 132, "progress": 250, "run_name": 300, "job_type": 190}
        anchors = {"status": "center", "mode": "center", "started": "center", "progress": "center", "run_name": "w", "job_type": "w"}
        for key in columns:
            self._jobs_tree.heading(key, text=headings[key])
            self._jobs_tree.column(key, width=widths[key], minwidth=70, anchor=anchors[key], stretch=(key == "run_name"))
        list_scroll_y = ttk.Scrollbar(list_frame, orient="vertical", command=self._jobs_tree.yview)
        list_scroll_x = ttk.Scrollbar(list_frame, orient="horizontal", command=self._jobs_tree.xview)
        self._jobs_tree.configure(yscrollcommand=list_scroll_y.set, xscrollcommand=list_scroll_x.set)
        self._jobs_tree.pack(side="top", fill="both", expand=True, padx=6, pady=(6, 0))
        list_scroll_y.pack(side="right", fill="y", pady=(6, 6), padx=(0, 6))
        list_scroll_x.pack(side="bottom", fill="x", padx=6, pady=(0, 6))
        self._jobs_tree.bind("<<TreeviewSelect>>", lambda _e: self._show_selected_job_details())
        self._jobs_tree.bind("<Double-1>", lambda _e: self._open_selected_job_output())
        self._jobs_tree.bind("<Button-3>", self._show_jobs_context_menu)

        self._jobs_context_menu = tk.Menu(self.root, tearoff=False)
        self._jobs_context_menu.add_command(
            label="Repeat / Edit as New Run",
            command=self._load_selected_job_for_repeat,
        )
        self._jobs_context_menu.add_command(
            label="Apply Training Parameters to Current Settings",
            command=self._apply_selected_job_training_parameters,
        )
        self._jobs_context_menu.add_command(
            label="Import Prompts from Job",
            command=self._import_prompts_from_selected_job,
        )
        self._jobs_context_menu.add_command(
            label="Load as Continuation / Resume",
            command=self._load_selected_job_as_continuation,
        )
        self._jobs_context_menu.add_command(
            label="Refine Face Identity…",
            command=self._load_selected_job_for_face_refinement,
        )
        self._jobs_context_menu.add_separator()
        self._jobs_context_menu.add_command(label="Open Output", command=self._open_selected_job_output)
        self._jobs_context_menu.add_command(label="Open Logs", command=self._open_selected_job_logs)
        self._jobs_context_menu.add_command(label="Copy Command", command=self._copy_selected_job_command)

        details_frame = ttk.LabelFrame(split, text="Job Details")
        split.add(details_frame, weight=4)
        details_toolbar = ttk.Frame(details_frame)
        details_toolbar.pack(fill="x", padx=6, pady=(6, 0))
        repeat_button = ttk.Button(details_toolbar, text="Repeat / Edit", command=self._load_selected_job_for_repeat)
        repeat_button.pack(side="left")
        ToolTip(
            repeat_button,
            "Loads this job's saved settings as a new editable run, clears resume inputs, and chooses a new output name. Nothing starts automatically.",
        )
        apply_params_button = ttk.Button(
            details_toolbar,
            text="Apply Parameters",
            command=self._apply_selected_job_training_parameters,
        )
        apply_params_button.pack(side="left", padx=(6, 0))
        ToolTip(
            apply_params_button,
            "Copies training hyperparameters from this job into the current form without changing dataset, model, output, note, prompt, or resume fields.",
        )
        import_prompts_button = ttk.Button(
            details_toolbar,
            text="Import Prompts",
            command=self._import_prompts_from_selected_job,
        )
        import_prompts_button.pack(side="left", padx=(6, 0))
        ToolTip(
            import_prompts_button,
            "Merges saved sample prompts from this job into the current prompt list and skips duplicates.",
        )
        refine_face_button = ttk.Button(details_toolbar, text="Refine Face…", command=self._load_selected_job_for_face_refinement)
        refine_face_button.pack(side="left", padx=(6, 0))
        ToolTip(refine_face_button, "Loads this completed Krea run as the input to a new face-refinement-only plan. Nothing starts automatically.")
        ttk.Button(details_toolbar, text="Open Output", command=self._open_selected_job_output).pack(side="left", padx=(6, 0))
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
        if self._backfill_job_lineage():
            self._save_job_history()
        self._refresh_job_history_view()

    def _save_job_history(self):
        try:
            with open(self._job_history_path, "w", encoding="utf-8") as history_file:
                json.dump(self._job_history[:200], history_file, indent=2)
        except OSError:
            pass

    @staticmethod
    def _job_metric(job, key):
        try:
            return max(0, int(job.get(key) or 0))
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _job_cumulative_progress(cls, job):
        prior_epochs = cls._job_metric(job, "continuation_prior_epochs")
        prior_steps = cls._job_metric(job, "continuation_prior_steps")
        return {
            "current_epoch": prior_epochs + cls._job_metric(job, "current_epoch"),
            "total_epochs": prior_epochs + cls._job_metric(job, "total_epochs"),
            "current_step": prior_steps + cls._job_metric(job, "current_step"),
            "total_steps": prior_steps + cls._job_metric(job, "total_steps"),
            "prior_epochs": prior_epochs,
            "prior_steps": prior_steps,
        }

    @staticmethod
    def _normalise_history_path(path):
        value = str(path or "").strip().replace("\\", "/").rstrip("/")
        return os.path.normcase(value)

    def _job_run_paths(self, job):
        paths = set()

        def add_path(path):
            normalised = self._normalise_history_path(path)
            if normalised:
                paths.add(normalised)

        settings = job.get("settings_snapshot")
        if isinstance(settings, dict):
            output_dir = settings.get("output_dir", "")
            output_name = settings.get("output_name", "")
            if output_dir:
                add_path(output_dir)
            if output_dir and output_name:
                try:
                    run_name = self._effective_run_name(settings)
                except (KeyError, TypeError, ValueError):
                    run_name = output_name
                add_path(Path(output_dir) / run_name)

        output_dir = job.get("output_dir", "")
        output_name = job.get("output_name", "")
        if output_dir:
            add_path(output_dir)
        if output_dir and output_name:
            add_path(Path(output_dir) / output_name)

        for command in job.get("commands") or []:
            if not self._is_training_command(command):
                continue
            command_dir = self._command_option(command, "--output_dir")
            command_name = self._command_option(command, "--output_name")
            if command_dir:
                add_path(command_dir)
            if command_dir and command_name:
                add_path(Path(command_dir) / command_name)
        return paths

    def _find_resume_parent_job(self, resume_path, jobs=None, exclude_job=None, before_started=""):
        resume_parent = self._normalise_history_path(Path(str(resume_path)).parent)
        if not resume_parent:
            return None
        candidates = []
        before_timestamp = None
        if before_started:
            try:
                before_timestamp = datetime.fromisoformat(str(before_started)).timestamp()
            except ValueError:
                pass

        for job in jobs if jobs is not None else self._job_history:
            if job is exclude_job:
                continue
            if before_timestamp is not None and job.get("started_at"):
                try:
                    if datetime.fromisoformat(str(job["started_at"])).timestamp() > before_timestamp:
                        continue
                except ValueError:
                    pass
            if resume_parent not in self._job_run_paths(job):
                continue
            try:
                timestamp = datetime.fromisoformat(str(job.get("finished_at") or job.get("started_at") or "")).timestamp()
            except ValueError:
                timestamp = 0
            candidates.append((timestamp, job))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    def _continuation_metadata(self, resume_path, source_job=None, jobs=None, before_started=""):
        if not resume_path:
            return {}
        source_job = source_job or self._find_resume_parent_job(
            resume_path,
            jobs=jobs,
            before_started=before_started,
        )
        if source_job:
            cumulative = self._job_cumulative_progress(source_job)
            state_epoch = self._state_epoch_from_path(resume_path)
            source_local_epoch = self._job_metric(source_job, "current_epoch")
            if state_epoch and state_epoch != source_local_epoch:
                cumulative["current_epoch"] = cumulative["prior_epochs"] + state_epoch
                source_total_steps = self._job_metric(source_job, "total_steps")
                source_total_epochs = self._job_metric(source_job, "total_epochs")
                if source_total_steps and source_total_epochs:
                    state_local_step = round(source_total_steps * min(state_epoch, source_total_epochs) / source_total_epochs)
                    cumulative["current_step"] = cumulative["prior_steps"] + state_local_step
            return {
                "continuation_parent_id": source_job.get("job_id", ""),
                "continuation_parent_title": source_job.get("output_name") or source_job.get("title", ""),
                "continuation_prior_epochs": cumulative["current_epoch"],
                "continuation_prior_steps": cumulative["current_step"],
                "continuation_depth": self._job_metric(source_job, "continuation_depth") + 1,
                "continuation_resume_state": str(resume_path),
            }

        state_epoch = self._state_epoch_from_path(resume_path)
        return {
            "continuation_parent_id": "",
            "continuation_parent_title": Path(str(resume_path)).parent.name or Path(str(resume_path)).name,
            "continuation_prior_epochs": state_epoch,
            "continuation_prior_steps": 0,
            "continuation_depth": 1,
            "continuation_resume_state": str(resume_path),
        }

    def _backfill_job_lineage(self):
        changed = False
        chronological = sorted(
            self._job_history,
            key=lambda job: str(job.get("started_at") or ""),
        )
        processed = []
        for job in chronological:
            if not job.get("job_id"):
                job["job_id"] = uuid.uuid4().hex
                changed = True
            resume_path = str(job.get("resume_path") or "").strip()
            if (
                resume_path
                and job.get("kind") in ("training", "staged_training")
                and not job.get("continuation_resume_state")
            ):
                parent = self._find_resume_parent_job(
                    resume_path,
                    jobs=processed,
                    exclude_job=job,
                    before_started=job.get("started_at", ""),
                )
                job.update(
                    self._continuation_metadata(
                        resume_path,
                        source_job=parent,
                        jobs=processed,
                        before_started=job.get("started_at", ""),
                    )
                )
                changed = True
            processed.append(job)
        return changed

    def _job_progress_summary(self, job):
        local_epoch = self._job_metric(job, "current_epoch")
        local_epoch_total = self._job_metric(job, "total_epochs")
        local_step = self._job_metric(job, "current_step")
        local_step_total = self._job_metric(job, "total_steps")
        cumulative = self._job_cumulative_progress(job)
        has_lineage = cumulative["prior_epochs"] > 0 or cumulative["prior_steps"] > 0

        if has_lineage:
            parts = []
            if local_epoch_total or cumulative["prior_epochs"]:
                parts.append(
                    f"E {local_epoch}/{local_epoch_total or '?'}→"
                    f"{cumulative['current_epoch']}/{cumulative['total_epochs'] or '?'}"
                )
            if local_step_total or cumulative["prior_steps"]:
                parts.append(
                    f"S {local_step}/{local_step_total or '?'}→"
                    f"{cumulative['current_step']}/{cumulative['total_steps'] or '?'}"
                )
            return "  |  ".join(parts)
        if local_step_total:
            return f"{local_step}/{local_step_total}"
        if local_epoch_total:
            return f"e{local_epoch}/{local_epoch_total}"
        return "-"

    @staticmethod
    def _job_display_name(job):
        """Return the user-facing artifact/run name without changing history storage."""
        output_name = str(job.get("output_name") or "").strip()
        if not output_name and isinstance(job.get("settings_snapshot"), dict):
            output_name = str(job["settings_snapshot"].get("output_name") or "").strip()
        if output_name:
            return output_name
        output_dir = str(job.get("output_dir") or "").strip()
        if output_dir:
            return Path(output_dir).name or output_dir
        return str(job.get("title") or "Job")

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
            progress = self._job_progress_summary(job)
            self._jobs_tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    job.get("status", "unknown"),
                    job.get("mode", ""),
                    timestamp,
                    progress,
                    self._job_display_name(job),
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

    def _show_jobs_context_menu(self, event):
        row_id = self._jobs_tree.identify_row(event.y)
        if not row_id:
            return
        self._jobs_tree.selection_set(row_id)
        self._jobs_tree.focus(row_id)
        self._show_selected_job_details()
        job = self._selected_job()
        can_repeat = bool(
            job
            and job.get("kind") in ("training", "staged_training")
            and (
                (isinstance(job.get("settings_snapshot"), dict) and job.get("settings_snapshot"))
                or job.get("commands")
                or job.get("output_name")
            )
            and not self.current_process
        )
        can_continue = bool(
            can_repeat
            and isinstance(job.get("settings_snapshot"), dict)
            and job.get("settings_snapshot")
        )
        can_import_prompts = bool(
            job
            and isinstance(job.get("settings_snapshot"), dict)
            and isinstance(job.get("settings_snapshot", {}).get("sample_prompts_data"), list)
            and job.get("settings_snapshot", {}).get("sample_prompts_data")
            and not self.current_process
        )
        can_refine_face = bool(
            can_repeat
            and isinstance(job.get("settings_snapshot"), dict)
            and job.get("settings_snapshot", {}).get("training_mode") == "Krea 2"
            and self._resolve_job_face_lora(job) is not None
        )
        self._jobs_context_menu.entryconfigure(
            0,
            state="normal" if can_repeat else "disabled",
        )
        self._jobs_context_menu.entryconfigure(
            1,
            state="normal" if can_repeat else "disabled",
        )
        self._jobs_context_menu.entryconfigure(
            2,
            state="normal" if can_import_prompts else "disabled",
        )
        self._jobs_context_menu.entryconfigure(
            3,
            state="normal" if can_continue else "disabled",
        )
        self._jobs_context_menu.entryconfigure(4, state="normal" if can_refine_face else "disabled")
        try:
            self._jobs_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._jobs_context_menu.grab_release()

    @staticmethod
    def _command_option(command, option):
        match = re.search(
            rf"(?:^|\s){re.escape(option)}\s+(?:\"([^\"]*)\"|'([^']*)'|(\S+))",
            str(command or ""),
        )
        if not match:
            return ""
        return next((value for value in match.groups() if value is not None), "")

    @staticmethod
    def _is_training_command(command):
        lowered = str(command or "").lower()
        return "train_network.py" in lowered or re.search(r"(?:^|[/\\\s])[^/\\\s]*_train\.py(?:\s|$)", lowered) is not None

    def _job_training_command(self, job):
        commands = job.get("commands") or []
        return next((command for command in reversed(commands) if self._is_training_command(command)), "")

    @staticmethod
    def _state_epoch_from_path(state_path):
        match = re.search(r"-(\d+)-state$", Path(state_path).name, re.IGNORECASE)
        return int(match.group(1)) if match else 0

    def _continuation_state_candidates(self, job):
        candidates = []

        def add_candidate(path):
            try:
                resolved = Path(path).expanduser()
                if resolved.is_dir() and any(resolved.iterdir()) and resolved not in candidates:
                    candidates.append(resolved)
            except (OSError, TypeError, ValueError):
                pass

        def add_run_directory(run_dir, output_name):
            if not run_dir or not output_name:
                return
            try:
                directory = Path(run_dir).expanduser()
                add_candidate(directory / f"{output_name}-state")
                for state_dir in directory.glob(f"{output_name}-*-state"):
                    add_candidate(state_dir)
            except (OSError, TypeError, ValueError):
                pass

        settings = job.get("settings_snapshot")
        if isinstance(settings, dict):
            try:
                for state_path in self._candidate_final_state_paths(settings):
                    add_candidate(state_path)
                run_name = self._effective_run_name(settings)
                add_run_directory(Path(settings["output_dir"]) / run_name, run_name)
            except (KeyError, TypeError, ValueError):
                pass

        for command in reversed(job.get("commands") or []):
            if not self._is_training_command(command):
                continue
            command_output_dir = self._command_option(command, "--output_dir")
            command_output_name = self._command_option(command, "--output_name")
            add_run_directory(command_output_dir, command_output_name)

        job_output_dir = job.get("output_dir", "")
        job_output_name = job.get("output_name", "")
        if job_output_dir and job_output_name:
            add_run_directory(Path(job_output_dir) / job_output_name, job_output_name)
            add_run_directory(job_output_dir, job_output_name)

        return candidates

    def _resolve_job_continuation_state(self, job):
        candidates = self._continuation_state_candidates(job)
        if candidates:
            finished_raw = str(job.get("finished_at") or "").strip()
            if finished_raw:
                try:
                    finished_timestamp = datetime.fromisoformat(finished_raw).timestamp()
                    candidates_at_job_time = [
                        path for path in candidates
                        if path.stat().st_mtime <= finished_timestamp
                    ]
                    if not candidates_at_job_time:
                        candidates_at_job_time = [
                            path for path in candidates
                            if path.stat().st_mtime <= finished_timestamp + 300
                        ]
                    if candidates_at_job_time:
                        candidates = candidates_at_job_time
                except (OSError, ValueError):
                    pass

            expected_epoch = int(job.get("current_epoch") or 0)
            if expected_epoch:
                epoch_matches = [
                    path for path in candidates
                    if self._state_epoch_from_path(path) == expected_epoch
                ]
                if epoch_matches:
                    return max(epoch_matches, key=lambda path: path.stat().st_mtime)

            return max(
                candidates,
                key=lambda path: (
                    path.stat().st_mtime,
                    self._state_epoch_from_path(path),
                ),
            )
        original_resume = str(job.get("resume_path") or "").strip()
        if original_resume and Path(original_resume).is_dir():
            return Path(original_resume)
        return None

    def _resolve_job_face_lora(self, job):
        """Find a complete Krea LoRA produced by a recorded job."""
        candidates = []
        settings = job.get("settings_snapshot")
        if isinstance(settings, dict):
            if settings.get("face_output_path"):
                candidates.append(Path(settings["face_output_path"]))
            try:
                candidates.extend(self._candidate_final_model_paths(settings))
            except (KeyError, TypeError, ValueError):
                pass
        for state_dir in self._continuation_state_candidates(job):
            candidates.append(state_dir / "model.safetensors")

        from musubi_tuner.face_refinement.lora_validation import validate_krea2_lora
        seen = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                validate_krea2_lora(resolved)
                return resolved
            except (OSError, ValueError):
                continue
        return None

    @staticmethod
    def _continuation_progress_text(job, state_path):
        parts = []
        current_epoch = int(job.get("current_epoch") or 0)
        total_epochs = int(job.get("total_epochs") or 0)
        state_epoch = MusubiTunerGUI._state_epoch_from_path(state_path)
        if current_epoch:
            parts.append(f"epoch {current_epoch}/{total_epochs}" if total_epochs else f"epoch {current_epoch}")
        elif state_epoch:
            parts.append(f"epoch {state_epoch}")

        current_step = int(job.get("current_step") or 0)
        total_steps = int(job.get("total_steps") or 0)
        if current_step:
            parts.append(f"step {current_step}/{total_steps}" if total_steps else f"step {current_step}")
        return ", ".join(parts) if parts else f"state {Path(state_path).name}"

    def _next_continuation_output_name(self, settings, source_name):
        base_name = f"{source_name}-cont"
        output_dir = Path(str(settings.get("output_dir") or "")).expanduser()
        candidate = base_name
        suffix = 2
        while output_dir and (output_dir / candidate).exists():
            candidate = f"{base_name}{suffix}"
            suffix += 1
        return candidate

    def _repeat_output_name_exists(self, settings, output_name):
        output_dir_text = str(settings.get("output_dir") or "").strip()
        if not output_dir_text:
            return False
        output_dir = Path(output_dir_text).expanduser()
        trial = dict(settings)
        trial["output_name"] = output_name
        try:
            run_names = {self._effective_run_name(trial)}
        except (KeyError, TypeError, ValueError):
            run_names = {output_name}
        if trial.get("use_staged_training"):
            for stage in trial.get("staged_training_config") or []:
                if stage.get("enabled"):
                    run_names.add(f"{output_name}-{self._stage_label_text(stage)}")
        return any((output_dir / run_name).exists() for run_name in run_names)

    def _next_repeat_output_name(self, settings, source_name):
        base_name = f"{source_name}-rerun"
        candidate = base_name
        suffix = 2
        while self._repeat_output_name_exists(settings, candidate):
            candidate = f"{base_name}{suffix}"
            suffix += 1
        return candidate

    def _recover_partial_job_settings(self, job):
        mode = str(job.get("mode") or "")
        if mode not in ("Wan 2.2", "Krea 2", "Flux.2"):
            mode = self._infer_mode_from_path(
                " ".join(job.get("commands") or [])
                + " " + str(job.get("output_name") or "")
                + " " + str(job.get("output_dir") or "")
            )
        if mode not in ("Wan 2.2", "Krea 2", "Flux.2"):
            mode = "Wan 2.2"
        settings = {
            "training_mode": mode,
            "output_dir": job.get("output_dir", ""),
            "output_name": job.get("output_name", ""),
            "logging_dir": job.get("logging_dir", ""),
            "training_comment": job.get("note", ""),
            "use_staged_training": False,
        }
        command = self._job_training_command(job)
        option_map = {
            "--dataset_config": "dataset_config",
            "--learning_rate": "learning_rate",
            "--max_train_epochs": "max_train_epochs",
            "--save_every_n_epochs": "save_every_n_epochs",
            "--save_every_n_steps": "save_every_n_steps",
            "--seed": "seed",
            "--network_dim": "network_dim_low",
            "--network_alpha": "network_alpha_low",
            "--optimizer_type": "optimizer_type",
            "--max_grad_norm": "max_grad_norm",
            "--gradient_accumulation_steps": "gradient_accumulation_steps",
            "--max_data_loader_n_workers": "max_data_loader_n_workers",
            "--blocks_to_swap": "blocks_to_swap",
            "--timestep_sampling": "timestep_sampling",
            "--num_timestep_buckets": "num_timestep_buckets",
            "--discrete_flow_shift": "discrete_flow_shift",
            "--mixed_precision": "mixed_precision",
            "--logging_dir": "logging_dir",
            "--log_prefix": "log_prefix",
            "--training_comment": "training_comment",
            "--vae": "vae_model",
        }
        for option, key in option_map.items():
            value = self._command_option(command, option)
            if value:
                settings[key] = value

        mode = settings.get("training_mode")
        dit_path = self._command_option(command, "--dit")
        text_encoder = self._command_option(command, "--text_encoder")
        if mode == "Krea 2":
            settings["krea2_dit_model"] = dit_path
            settings["krea2_text_encoder"] = text_encoder
            settings["krea2_turbo_dit"] = self._command_option(command, "--turbo_dit")
            settings["krea2_projector_diff"] = self._command_option(command, "--projector_diff")
            settings["krea2_projector_diff_strength"] = self._command_option(command, "--projector_diff_strength")
        elif mode == "Flux.2":
            settings["flux2_dit_model"] = dit_path
            settings["flux2_text_encoder"] = text_encoder

        network_module = self._command_option(command, "--network_module").lower()
        if "lokr" in network_module:
            settings["network_type"] = "LoKr"
        elif "loha" in network_module:
            settings["network_type"] = "LoHa"
        elif network_module:
            settings["network_type"] = "LoRA"

        flag_map = {
            "--gradient_checkpointing": "gradient_checkpointing",
            "--persistent_data_loader_workers": "persistent_data_loader_workers",
            "--save_state": "save_state",
            "--fp8_base": "fp8_base",
            "--fp8_scaled": "fp8_scaled",
            "--compile": "compile",
            "--preserve_distribution_shape": "preserve_distribution_shape",
        }
        for flag, key in flag_map.items():
            settings[key] = re.search(rf"(?:^|\s){re.escape(flag)}(?:\s|$)", command) is not None

        if re.search(r"(?:^|\s)--sdpa(?:\s|$)", command):
            settings["attention_mechanism"] = "sdpa"
        elif re.search(r"(?:^|\s)--xformers(?:\s|$)", command):
            settings["attention_mechanism"] = "xformers"
        return {key: value for key, value in settings.items() if value not in (None, "")}

    @staticmethod
    def _portable_training_parameter_keys():
        return {
            "learning_rate",
            "max_train_epochs",
            "save_every_n_epochs",
            "save_every_n_steps",
            "seed",
            "network_type",
            "lokr_factor",
            "network_dim_low",
            "network_alpha_low",
            "optimizer_type",
            "max_grad_norm",
            "optimizer_args",
            "lr_scheduler",
            "lr_warmup_steps",
            "lr_scheduler_num_cycles",
            "lr_scheduler_power",
            "lr_scheduler_min_lr_ratio",
            "mixed_precision",
            "gradient_checkpointing",
            "persistent_data_loader_workers",
            "gradient_accumulation_steps",
            "max_data_loader_n_workers",
            "blocks_to_swap",
            "compile",
            "compile_backend",
            "compile_mode",
            "compile_dynamic",
            "compile_fullgraph",
            "compile_cache_size_limit",
            "attention_mechanism",
            "save_state",
            "rename_final_artifacts_to_epoch",
        }

    @staticmethod
    def _same_mode_training_parameter_keys():
        return {
            "network_dim_high",
            "network_alpha_high",
            "offload_inactive_dit",
            "timestep_sampling",
            "num_timestep_buckets",
            "timestep_boundary",
            "discrete_flow_shift",
            "preserve_distribution_shape",
            "fp8_base",
            "fp8_scaled",
            "fp8_t5",
            "fp8_llm",
            "force_v2_1_time_embedding",
        }

    def _apply_selected_job_training_parameters(self):
        if self.current_process:
            messagebox.showwarning("Apply parameters", "Stop the active process before changing training settings.")
            return
        job = self._selected_job()
        if not job:
            return
        settings_snapshot = job.get("settings_snapshot")
        has_full_snapshot = isinstance(settings_snapshot, dict) and bool(settings_snapshot)
        if has_full_snapshot:
            source_settings = copy.deepcopy(settings_snapshot)
        elif job.get("commands") or job.get("output_name"):
            source_settings = self._recover_partial_job_settings(job)
        else:
            messagebox.showerror(
                "Parameters unavailable",
                "This job has no settings snapshot or command from which training parameters can be recovered.",
            )
            return

        current_settings = self.get_settings()
        keys = self._portable_training_parameter_keys()
        source_mode = str(source_settings.get("training_mode") or job.get("mode") or "")
        current_mode = str(current_settings.get("training_mode") or "")
        same_mode = source_mode == current_mode
        if same_mode:
            keys |= self._same_mode_training_parameter_keys()

        updates = {
            key: copy.deepcopy(source_settings[key])
            for key in keys
            if key in source_settings and key in self.entries and source_settings[key] != current_settings.get(key)
        }
        if not updates:
            messagebox.showinfo(
                "No parameter changes",
                "The recoverable training parameters from this job already match the current settings.",
            )
            return

        labels = [self.field_label_text.get(key, key.replace("_", " ").title()) for key in sorted(updates)]
        preview = "\n".join(f"• {label}" for label in labels[:14])
        if len(labels) > 14:
            preview += f"\n• …and {len(labels) - 14} more"
        source_title = str(job.get("output_name") or job.get("title") or "selected job")
        partial_note = (
            "\n\nThis is an older job without a full snapshot, so only parameters recovered from its command will be applied."
            if not has_full_snapshot else ""
        )
        mode_note = (
            "\n\nThe source uses a different training mode. Mode-specific timestep, FP8, and dual-model parameters will be kept unchanged."
            if not same_mode else ""
        )
        if not messagebox.askyesno(
            "Apply training parameters?",
            f"Apply {len(updates)} training parameter(s) from {source_title}?\n\n{preview}"
            f"{partial_note}{mode_note}\n\n"
            "Dataset/model paths, output name and folders, comments, prompts, staged plans, and resume inputs will not be changed.",
        ):
            return

        self.set_values(updates)
        self._select_page(1)
        self.run_status_var.set(f"⚪ Applied training parameters from {source_title}")
        messagebox.showinfo(
            "Training parameters applied",
            f"Applied {len(updates)} parameter(s). Identity, path, output, notes, prompts, staging, and resume fields were preserved.",
        )

    @staticmethod
    def _sample_prompt_identity(prompt):
        return prompt_identity(prompt)

    def _import_prompts_from_selected_job(self):
        if self.current_process:
            messagebox.showwarning("Import prompts", "Stop the active process before changing the prompt list.")
            return
        job = self._selected_job()
        if not job:
            return
        snapshot = job.get("settings_snapshot")
        source_prompts = snapshot.get("sample_prompts_data") if isinstance(snapshot, dict) else None
        source_prompts = [prompt for prompt in (source_prompts or []) if isinstance(prompt, dict)]
        if not source_prompts:
            messagebox.showinfo("No saved prompts", "This job does not contain any saved sample prompts.")
            return

        existing_identities = {
            self._sample_prompt_identity(prompt)
            for prompt in self._sample_prompts_data
            if isinstance(prompt, dict)
        }
        additions = []
        duplicate_count = 0
        for source_prompt in source_prompts:
            identity = self._sample_prompt_identity(source_prompt)
            if identity in existing_identities:
                duplicate_count += 1
                continue
            imported = copy.deepcopy(source_prompt)
            imported.setdefault("enabled", True)
            additions.append(imported)
            existing_identities.add(identity)

        source_title = str(job.get("output_name") or job.get("title") or "selected job")
        if not additions:
            messagebox.showinfo(
                "Prompts already present",
                f"All {duplicate_count} prompt(s) from {source_title} already exist in the current prompt list.",
            )
            return
        if not messagebox.askyesno(
            "Import sample prompts?",
            f"Import {len(additions)} new prompt(s) from {source_title}?\n\n"
            f"Existing prompts will remain unchanged. {duplicate_count} duplicate(s) will be skipped.",
        ):
            return

        self._sample_prompts_data.extend(additions)
        self._rebuild_prompt_list()
        self.update_button_states()
        self._select_page(4)
        self.run_status_var.set(f"⚪ Imported {len(additions)} sample prompt(s) from {source_title}")
        messagebox.showinfo(
            "Sample prompts imported",
            f"Added {len(additions)} prompt(s) and skipped {duplicate_count} duplicate(s).",
        )

    def _load_selected_job_for_repeat(self):
        if self.current_process:
            messagebox.showwarning("Repeat job", "Stop the active process before loading another job.")
            return
        job = self._selected_job()
        if not job:
            return
        settings_snapshot = job.get("settings_snapshot")
        has_full_snapshot = isinstance(settings_snapshot, dict) and bool(settings_snapshot)
        if not has_full_snapshot and not (job.get("commands") or job.get("output_name")):
            messagebox.showerror(
                "Repeat unavailable",
                "This job does not contain a settings snapshot, training command, or output information to restore.",
            )
            return

        if has_full_snapshot:
            settings = copy.deepcopy(settings_snapshot)
        else:
            settings = self._recover_partial_job_settings(job)
            self.load_default_settings()
        source_name = str(settings.get("output_name") or job.get("output_name") or "training")
        settings["output_name"] = self._next_repeat_output_name(settings, source_name)
        settings["resume_path"] = ""
        settings["network_weights"] = ""

        existing_comment = str(settings.get("training_comment") or "")
        comment_lines = [
            line for line in existing_comment.splitlines()
            if not re.match(r"^\(Continuation from .+: .+\)$", line.strip(), re.IGNORECASE)
        ]
        source_title = str(job.get("output_name") or job.get("title") or source_name)
        repeat_note = f"(Settings repeated from {source_title}; loaded for editing)"
        cleaned_comment = "\n".join(comment_lines).strip()
        if repeat_note not in cleaned_comment:
            settings["training_comment"] = f"{cleaned_comment}\n{repeat_note}".strip()

        self._pending_continuation = None
        self.set_values(settings)
        self._select_page(0)
        self.run_status_var.set("⚪ Repeated job settings loaded for editing")
        if has_full_snapshot:
            messagebox.showinfo(
                "Job settings loaded",
                f"Loaded settings from:\n{source_title}\n\n"
                f"New output name:\n{settings['output_name']}\n\n"
                "Resume state and network weights were cleared, so this is a new run. Review or edit any setting before starting.",
            )
        else:
            messagebox.showwarning(
                "Partial job settings loaded",
                f"This older job had no complete settings snapshot. The GUI recovered available values from its command and job record.\n\n"
                f"New output name:\n{settings['output_name']}\n\n"
                "All unavailable fields were reset to defaults. Review model paths, noise-model selection, sampling, caching, and advanced options before running.",
            )

    def _load_selected_job_for_face_refinement(self):
        if self.current_process:
            messagebox.showwarning("Face Refinement", "Stop the active process before loading another job.")
            return
        job = self._selected_job()
        if not job:
            return
        snapshot = job.get("settings_snapshot")
        if not isinstance(snapshot, dict) or snapshot.get("training_mode") != "Krea 2":
            messagebox.showerror("Face Refinement unavailable", "Select a recorded Krea 2 training job with a complete settings snapshot.")
            return
        input_lora = self._resolve_job_face_lora(job)
        if input_lora is None:
            messagebox.showerror("Face Refinement unavailable", "No complete Krea 2 LoRA could be found for this job.")
            return

        settings = copy.deepcopy(snapshot)
        source_name = str(settings.get("output_name") or job.get("output_name") or "krea-lora")
        settings["output_name"] = self._next_repeat_output_name(settings, f"{source_name}-face")
        settings["resume_path"] = ""
        settings["network_weights"] = ""
        face_config = self._default_face_refinement_config()
        face_config.update(copy.deepcopy(snapshot.get("face_refinement_config") or {}))
        face_config.update({"input_mode": "existing_lora", "input_lora": str(input_lora)})
        settings["use_staged_training"] = True
        settings["staged_training_config"] = [{
            "label": "face-refinement", "enabled": True, "type": "face_refinement",
            "dataset_config": "", "epochs": "", "steps": str(face_config["steps"]),
        }]
        settings["face_refinement_config"] = face_config
        self._pending_continuation = None
        self.set_values(settings)
        self._select_page(1)
        self.run_status_var.set("⚪ Face-refinement continuation loaded for review")
        self._open_face_refinement_dialog()

    def _load_selected_job_as_continuation(self):
        if self.current_process:
            messagebox.showwarning("Continuation", "Stop the active process before loading another job.")
            return
        job = self._selected_job()
        if not job:
            return
        settings_snapshot = job.get("settings_snapshot")
        if not isinstance(settings_snapshot, dict) or not settings_snapshot:
            messagebox.showerror(
                "Continuation unavailable",
                "This imported job does not contain a complete saved settings snapshot, so restoring its model and training settings would be unsafe.",
            )
            return

        state_path = self._resolve_job_continuation_state(job)
        if state_path is None:
            messagebox.showerror(
                "Continuation state not found",
                "No saved Accelerate state folder could be found for this job. Continuation requires Save State to have been enabled.",
            )
            return
        self._pending_continuation = self._continuation_metadata(
            state_path,
            source_job=job,
        )

        settings = copy.deepcopy(settings_snapshot)
        training_command = self._job_training_command(job)
        command_output_name = self._command_option(training_command, "--output_name")
        command_dataset = self._command_option(training_command, "--dataset_config")
        command_epochs = self._command_option(training_command, "--max_train_epochs")
        if command_dataset:
            settings["dataset_config"] = command_dataset
        if command_epochs:
            settings["max_train_epochs"] = command_epochs

        if job.get("kind") == "staged_training" and command_output_name:
            source_name = command_output_name
        else:
            # Normal Wan commands add _LowNoise/_HighNoise internally. Keep the
            # saved base name so the continuation does not duplicate that suffix.
            source_name = str(
                settings.get("output_name")
                or job.get("output_name")
                or command_output_name
                or "training"
            )
        settings["output_name"] = self._next_continuation_output_name(settings, source_name)
        settings["resume_path"] = str(state_path)
        settings["network_weights"] = ""
        settings["save_state"] = True
        settings["recache_latents"] = False
        settings["recache_text"] = False
        settings["use_staged_training"] = False

        progress = self._continuation_progress_text(job, state_path)
        source_title = str(job.get("output_name") or job.get("title") or source_name)
        continuation_note = f"(Continuation from {source_title}: {progress})"
        existing_comment = str(settings.get("training_comment") or "").rstrip()
        if continuation_note not in existing_comment:
            settings["training_comment"] = f"{existing_comment}\n{continuation_note}".strip()

        self.set_values(settings)
        self._select_page(1)
        self.run_status_var.set(f"🟢 Continuation loaded from {Path(state_path).name}")
        messagebox.showinfo(
            "Continuation loaded",
            f"Resume state:\n{state_path}\n\n"
            f"New output name:\n{settings['output_name']}\n\n"
            "Recaching and staged progression were disabled. Review the epoch count and notes, then run training.",
        )

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
        cumulative = self._job_cumulative_progress(job)
        parent_title = str(job.get("continuation_parent_title") or "").strip()
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
            f"Run Step: {job.get('current_step', 'N/A')} / {job.get('total_steps', 'N/A')}",
            f"Overall Step: {cumulative['current_step']} / {cumulative['total_steps']}",
            f"Run Epoch: {job.get('current_epoch', 'N/A')} / {job.get('total_epochs', 'N/A')}",
            f"Overall Epoch: {cumulative['current_epoch']} / {cumulative['total_epochs']}",
            "",
            "Prompt / note:",
            job.get("note", "") or "(none)",
            "",
            "Commands:",
        ]
        if parent_title:
            lines[11:11] = [
                f"Continuation From: {parent_title}",
                f"Continuation Depth: {self._job_metric(job, 'continuation_depth')}",
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
            "settings_snapshot": dict(job.get("settings_snapshot", {})) if isinstance(job.get("settings_snapshot"), dict) else {},
            "job_id": job.get("job_id", ""),
            "continuation_parent_id": job.get("continuation_parent_id", ""),
            "continuation_parent_title": job.get("continuation_parent_title", ""),
            "continuation_prior_epochs": job.get("continuation_prior_epochs", 0),
            "continuation_prior_steps": job.get("continuation_prior_steps", 0),
            "continuation_depth": job.get("continuation_depth", 0),
            "continuation_resume_state": job.get("continuation_resume_state", ""),
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
        self._backfill_job_lineage()
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
                        "settings_snapshot": dict(data),
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
        resume_path = str(settings.get("resume_path") or "").strip()
        continuation = {}
        if kind in ("training", "staged_training") and resume_path:
            pending_resume = self._normalise_history_path(
                (self._pending_continuation or {}).get("continuation_resume_state", "")
            )
            if pending_resume and pending_resume == self._normalise_history_path(resume_path):
                continuation = dict(self._pending_continuation)
            else:
                continuation = self._continuation_metadata(resume_path)
        self._pending_continuation = None
        self.current_prior_epochs = self._job_metric(continuation, "continuation_prior_epochs")
        self.current_prior_steps = self._job_metric(continuation, "continuation_prior_steps")
        if kind in ("training", "staged_training"):
            self.current_step = 0
            self.current_total_steps = 0
            self.current_epoch_num = 0
            self.current_epoch_total = 0
        training_comment = str(settings.get("training_comment") or "").strip()
        if training_comment:
            note = f"{note}\n\n{training_comment}".strip()
        self._active_job = {
            "job_id": uuid.uuid4().hex,
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
            "settings_snapshot": dict(settings),
        }
        self._active_job.update(continuation)
        self.update_training_counters()

    @staticmethod
    def _effective_run_name(settings):
        run_name = settings["output_name"]
        if settings.get("training_mode") == "Wan 2.2":
            train_low = bool(settings.get("train_low_noise"))
            train_high = bool(settings.get("train_high_noise"))
            combined = train_low and train_high and not (
                str(settings.get("network_dim_high") or "").strip()
                or str(settings.get("network_alpha_high") or "").strip()
            )
            if not combined:
                run_name += "_HighNoise" if train_high else "_LowNoise"
        return run_name

    def _rename_final_training_artifacts(self, settings):
        if not settings.get("rename_final_artifacts_to_epoch", True):
            return
        output_dir = str(settings.get("output_dir") or "").strip()
        output_name = str(settings.get("output_name") or "").strip()
        epoch_text = str(settings.get("max_train_epochs") or "").strip()
        if not output_dir or not output_name or not epoch_text.isdigit():
            return

        run_name = self._effective_run_name(settings)
        run_dir = Path(output_dir) / run_name
        if not run_dir.is_dir():
            return

        epoch_suffix = f"{int(epoch_text):06d}"
        rename_pairs = [
            (run_dir / f"{run_name}.safetensors", run_dir / f"{run_name}-{epoch_suffix}.safetensors"),
            (run_dir / f"{run_name}-state", run_dir / f"{run_name}-{epoch_suffix}-state"),
        ]

        for source_path, target_path in rename_pairs:
            if not source_path.exists():
                continue
            if target_path.exists():
                self.output_text.insert(
                    tk.END,
                    f"\n--- Skipped rename because target already exists: {target_path.name} ---\n",
                )
                continue
            try:
                source_path.rename(target_path)
                self.output_text.insert(
                    tk.END,
                    f"\n--- Renamed final artifact: {source_path.name} -> {target_path.name} ---\n",
                )
            except Exception as e:
                self.output_text.insert(
                    tk.END,
                    f"\n--- Failed to rename {source_path.name}: {e} ---\n",
                )

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

    @staticmethod
    def _factor_dimension(dimension, factor=-1):
        if factor > 0 and dimension % factor == 0:
            first, second = factor, dimension // factor
            return (first, second) if first <= second else (second, first)
        limit = dimension if factor < 0 else factor
        first, second = 1, dimension
        best_length = first + second
        while first < second:
            candidate = first + 1
            while dimension % candidate != 0:
                candidate += 1
            other = dimension // candidate
            if candidate + other > best_length or candidate > limit:
                break
            first, second = candidate, other
        return (first, second) if first <= second else (second, first)

    def _target_linear_shapes(self, model_path, mode):
        path = Path(model_path).expanduser()
        if path.suffix.lower() != ".safetensors" or not path.is_file():
            raise ValueError("select an existing .safetensors DiT model")
        stat = path.stat()
        cache_key = (str(path), mode, stat.st_size, stat.st_mtime_ns)
        cached = self._lora_shape_cache.get(cache_key)
        if cached is not None:
            return cached

        with path.open("rb") as model_file:
            header_size_raw = model_file.read(8)
            if len(header_size_raw) != 8:
                raise ValueError("the model does not have a valid safetensors header")
            header_size = struct.unpack("<Q", header_size_raw)[0]
            if header_size > 64 * 1024**2:
                raise ValueError("the safetensors header is unexpectedly large")
            header = json.loads(model_file.read(header_size))

        shapes = []
        for key, tensor in header.items():
            if key == "__metadata__" or not key.endswith(".weight"):
                continue
            shape = tensor.get("shape", [])
            if len(shape) != 2:
                continue
            if mode == "Wan 2.2":
                targeted = re.search(r"(?:^|\.)blocks\.\d+\.", key) is not None
            elif mode in ("Flux.2 Klein", "Flux.2 Dev"):
                targeted = "double_blocks." in key or "single_blocks." in key
            else:
                targeted = True  # Krea 2 targets every Linear in its DiT.
            if targeted:
                shapes.append((int(shape[0]), int(shape[1])))

        if not shapes:
            raise ValueError("no target Linear layers were found in this model")
        result = tuple(shapes)
        self._lora_shape_cache = {cache_key: result}
        return result

    def _estimate_adapter_bytes(self, model_path, mode, rank, network_type, lokr_factor):
        shapes = self._target_linear_shapes(model_path, mode)
        parameter_count = 0
        tensor_count = 0
        for out_dim, in_dim in shapes:
            if network_type == "LoHa":
                parameter_count += 2 * rank * (out_dim + in_dim) + 1
                tensor_count += 5
            elif network_type == "LoKr":
                in_small, in_large = self._factor_dimension(in_dim, lokr_factor)
                out_small, out_large = self._factor_dimension(out_dim, lokr_factor)
                parameter_count += out_small * in_small + 1
                if rank < max(out_large, in_large) / 2:
                    parameter_count += rank * (out_large + in_large)
                    tensor_count += 4
                else:
                    parameter_count += out_large * in_large
                    tensor_count += 3
            else:
                parameter_count += rank * (out_dim + in_dim) + 1
                tensor_count += 3

        # Network weights are saved as FP32 by default. Safetensors headers and
        # training metadata add a small per-tensor overhead.
        estimated_bytes = parameter_count * 4 + tensor_count * 120 + 8 * 1024
        return estimated_bytes, len(shapes)

    @staticmethod
    def _format_estimated_size(byte_count):
        if byte_count >= 1024**3:
            return f"{byte_count / 1024**3:.2f} GiB"
        return f"{byte_count / 1024**2:.1f} MiB"

    def _update_lora_size_estimate(self):
        if not hasattr(self, "lora_size_estimate_var"):
            return
        try:
            mode = self.training_mode_var.get()
            network_type = self.entries["network_type"].get()
            factor_text = self.entries["lokr_factor"].get().strip()
            lokr_factor = int(factor_text) if factor_text else -1

            estimates = []
            if mode == "Wan 2.2":
                train_low = self.entries["train_low_noise"].var.get()
                train_high = self.entries["train_high_noise"].var.get()
                high_rank_text = self.entries["network_dim_high"].get().strip()
                separate = train_low and train_high and bool(
                    high_rank_text or self.entries["network_alpha_high"].get().strip()
                )
                if train_low:
                    estimates.append((
                        "Low-noise" if separate else "Final",
                        self.entries["dit_low_noise"].get().strip(),
                        self.entries["network_dim_low"].get().strip(),
                    ))
                if train_high and (separate or not train_low):
                    estimates.append((
                        "High-noise" if separate else "Final",
                        self.entries["dit_high_noise"].get().strip(),
                        high_rank_text or self.entries["network_dim_low"].get().strip(),
                    ))
            elif mode == "Krea 2":
                estimates.append(("Final", self.entries["krea2_dit_model"].get().strip(), self.entries["network_dim_low"].get().strip()))
            else:
                estimates.append(("Final", self.entries["flux2_dit_model"].get().strip(), self.entries["network_dim_low"].get().strip()))

            rendered = []
            layer_counts = []
            for label, model_path, rank_text in estimates:
                rank = int(rank_text)
                if rank < 1:
                    raise ValueError("rank must be at least 1")
                estimated_bytes, layer_count = self._estimate_adapter_bytes(
                    model_path, mode, rank, network_type, lokr_factor
                )
                rendered.append(f"{label}: ≈ {self._format_estimated_size(estimated_bytes)}")
                layer_counts.append(layer_count)

            if not rendered:
                raise ValueError("enable at least one model")
            layers_text = f"{layer_counts[0]} targeted Linear layers" if len(set(layer_counts)) == 1 else "architecture target layers"
            self.lora_size_estimate_var.set(
                f"{' · '.join(rendered)} ({network_type}, FP32 save, {layers_text}). "
                "Network alpha changes scaling, not file size."
            )
        except (ValueError, OSError, json.JSONDecodeError, struct.error) as exc:
            self.lora_size_estimate_var.set(f"Estimated final LoRA size unavailable: {exc}. Network alpha does not affect size.")

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
            self._update_run_mode_controls()
            self._update_lora_size_estimate()
            self._refresh_face_refinement_workspace()
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

        refinement_only = bool(
            self.entries.get("use_staged_training")
            and self.entries["use_staged_training"].var.get()
            and next((item for item in self._staged_training_config if item.get("enabled")), {}).get("type") == "face_refinement"
            and (self._face_refinement_config or {}).get("input_mode") == "existing_lora"
        )
        self.entries["dataset_config"].is_required = not refinement_only
        # A first-stage existing-LoRA face refinement has its own step count,
        # learning rate, and network shape. Standard SFT-only fields must not
        # block launch even when an older job snapshot left them blank.
        self.entries["max_train_epochs"].is_required = not refinement_only
        self.entries["learning_rate"].is_required = not refinement_only

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
            self.entries["krea2_text_encoder"].is_required = is_krea2 and (self.entries["recache_text"].var.get() or wants_samples or refinement_only)
            self.entries["t5_model"].is_required = False
            self.entries["dit_high_noise"].is_required = False
            self.entries["dit_low_noise"].is_required = False
            self.entries["clip_model"].is_required = False
            self.entries["network_dim_low"].is_required = not refinement_only
            self.entries["network_alpha_low"].is_required = not refinement_only
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
        self.start_btn.config(state="normal" if all_valid and not self.current_process else "disabled")
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

        if self.entries["compile"].var.get():
            self.hidden_frames["compile_options"].pack(fill="x")
        else:
            self.hidden_frames["compile_options"].pack_forget()

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
        try:
            if is_krea2:
                self.hidden_frames['krea2_regularization'].pack(fill="x", padx=10, pady=10)
            else:
                self.hidden_frames['krea2_regularization'].pack_forget()
        except (KeyError, tk.TclError):
            pass

    def _apply_krea2_generalization_preset(self):
        preset = self.entries["krea2_generalization_preset"].get()
        values = {
            "Off (Baseline)": ("0", "0"),
            "Weight Noise Only": ("0.0125", "0"),
            "Balanced Experimental": ("0.0125", "0.01"),
        }.get(preset)
        if values is None:
            return
        self.entries["krea2_weight_noise_sigma"].set(values[0])
        self.entries["krea2_weight_noise_mode"].set("relative")
        self.entries["krea2_depth_anchor_weight"].set(values[1])
        self.entries["krea2_depth_anchor_model"].set("depth-anything/Depth-Anything-V2-Small-hf")
        self.entries["krea2_depth_anchor_input_size"].set("518")
        self.entries["krea2_depth_anchor_gradient_weight"].set("0.5")
        self.entries["krea2_depth_anchor_grad_checkpoint"].var.set(True)

    def get_settings(self):
        settings = {}
        for key, widget in self.entries.items():
            if isinstance(widget, (tk.BooleanVar, tk.StringVar)): settings[key] = widget.get()
            elif hasattr(widget, 'var'): settings[key] = widget.var.get()
            elif isinstance(widget, tk.Text): settings[key] = widget.get("1.0", "end-1c")
            else: settings[key] = widget.get()
        settings["training_mode"] = self.training_mode_var.get()
        settings["appearance_mode"] = self.appearance_mode_var.get()
        settings["sample_prompts_data"] = self._sample_prompts_data
        settings["staged_training_config"] = self._staged_training_config
        settings["staged_recache_latents"] = self._staged_recache_latents
        settings["face_refinement_config"] = self._face_refinement_config
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
        if "staged_training_config" in settings:
            self._staged_training_config = settings.get("staged_training_config") or []
        if "staged_recache_latents" in settings:
            self._staged_recache_latents = bool(settings.get("staged_recache_latents"))
        if "face_refinement_config" in settings:
            self._face_refinement_config = settings.get("face_refinement_config") or {}
        for key, value in settings.items():
            if key in ("training_mode", "appearance_mode", "sample_prompts_data", "sample_prompts", "staged_training_config", "staged_recache_latents", "face_refinement_config"): continue
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
                elif isinstance(widget, tk.Text):
                    widget.delete("1.0", tk.END)
                    widget.insert("1.0", str(value) if value is not None else "")
        try: self._update_staged_summary()
        except AttributeError: pass
        self.update_button_states()

    def load_default_settings(self):
        defaults = {
            "dataset_config": "", "project_root": "", "dit_high_noise": "", "dit_low_noise": "", "is_i2v": False,
            "train_high_noise": True, "train_low_noise": True,
            "min_timestep_low": "0", "max_timestep_low": "875", "min_timestep_high": "875", "max_timestep_high": "1000",
            "vae_model": "", "clip_model": "", "t5_model": "",
            "flux2_model_version": "Klein Base 4B ★", "flux2_dit_model": "", "flux2_text_encoder": "", "fp8_text_encoder": False,
            "krea2_dit_model": "", "krea2_text_encoder": "", "krea2_turbo_dit": "", "krea2_turbo_dit_cache": False,
            "krea2_projector_diff": "", "krea2_projector_diff_strength": "1.0",
            "krea2_generalization_preset": "Off (Baseline)",
            "krea2_weight_noise_sigma": "0", "krea2_weight_noise_mode": "relative", "krea2_weight_noise_bound_norm": False,
            "krea2_depth_anchor_weight": "0", "krea2_depth_anchor_model": "depth-anything/Depth-Anything-V2-Small-hf",
            "krea2_depth_anchor_input_size": "518", "krea2_depth_anchor_gradient_weight": "0.5", "krea2_depth_anchor_grad_checkpoint": True,
            "output_dir": "", "output_name": "my-lora",
            "training_comment": "",
            "learning_rate": "2e-4", "max_train_epochs": "10", "save_every_n_epochs": "1", "save_every_n_steps": "", "seed": "42",
            "network_type": "LoRA", "lokr_factor": "", "network_dim_low": "32", "network_alpha_low": "16", "network_dim_high": "", "network_alpha_high": "",
            "optimizer_type": "adamw8bit", "max_grad_norm": "1.0", "optimizer_args": "", "lr_scheduler": "cosine",
            "lr_warmup_steps": "0", "lr_scheduler_num_cycles": "1",
            "mixed_precision": "fp16", "gradient_accumulation_steps": "1",
            "max_data_loader_n_workers": "2", "blocks_to_swap": "10", "timestep_sampling": "shift",
            "compile": False, "compile_backend": "inductor", "compile_mode": "default",
            "compile_dynamic": "auto", "compile_fullgraph": False, "compile_cache_size_limit": "32",
            "num_timestep_buckets": "", "timestep_boundary": "875", "discrete_flow_shift": "3.0", "preserve_distribution_shape": False,
            "gradient_checkpointing": True, "persistent_data_loader_workers": True, "save_state": True,
            "rename_final_artifacts_to_epoch": True,
            "fp8_base": False, "fp8_scaled": False, "fp8_t5": False, "fp8_llm": False, "force_v2_1_time_embedding": False, "offload_inactive_dit": False,
            "attention_mechanism": "xformers", "resume_path": "", "network_weights": "",
            "log_with": "none", "logging_dir": "", "log_prefix": "",
            "recache_latents": False, "recache_text": False,
            "convert_lora_path": "", "convert_output_dir": "",
            "training_mode": "Wan 2.2",
            "sample_every_n_epochs": "", "sample_every_n_steps": "", "sample_at_first": False,
            "sample_prompts_data": [],
            "use_staged_training": False, "staged_training_config": [], "staged_recache_latents": True, "face_refinement_config": {},
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
            pynvml.nvmlInit()
            self._vram_gpu_index = None
            self._vram_previous_gpu_index = None
            self._vram_baseline = [
                pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(index)).used
                for index in range(pynvml.nvmlDeviceGetCount())
            ]
            self.monitoring_active = True
            self.peak_vram = 0
            self.vram_label_var.set("VRAM: Waiting for training GPU...")
            self.peak_vram_label_var.set("Peak VRAM: N/A")
            self.vram_thread = threading.Thread(target=self.vram_monitor_loop, daemon=True); self.vram_thread.start()
        except pynvml.NVMLError: self.vram_label_var.set(f"VRAM: NVML Error")

    def stop_vram_monitor(self):
        self.monitoring_active = False
        if PYNVML_AVAILABLE:
            try: pynvml.nvmlShutdown()
            except pynvml.NVMLError: pass

    def _training_process_ids(self):
        process = self.current_process
        if not process or process.poll() is not None:
            return set()

        process_ids = {process.pid}
        if PSUTIL_AVAILABLE:
            try:
                process_ids.update(child.pid for child in psutil.Process(process.pid).children(recursive=True))
            except (psutil.Error, OSError):
                pass
        return process_ids

    @staticmethod
    def _gpu_process_ids(handle):
        process_ids = set()
        queries = (
            getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", None),
            getattr(pynvml, "nvmlDeviceGetGraphicsRunningProcesses", None),
        )
        for query in queries:
            if query is None:
                continue
            try:
                process_ids.update(process.pid for process in query(handle))
            except pynvml.NVMLError:
                pass
        return process_ids

    def _detect_training_gpu(self, handles, memory_info):
        training_process_ids = self._training_process_ids()
        if training_process_ids:
            for index, handle in enumerate(handles):
                if training_process_ids & self._gpu_process_ids(handle):
                    return index

            # NVML process accounting is unavailable on some Windows driver
            # modes. In that case, select the GPU whose usage grew after this
            # training run started instead of assuming physical GPU 0.
            deltas = [
                info.used - self._vram_baseline[index]
                for index, info in enumerate(memory_info)
            ]
            if deltas:
                index = max(range(len(deltas)), key=deltas.__getitem__)
                if deltas[index] >= 64 * 1024**2:
                    return index
        return None

    def vram_monitor_loop(self):
        try:
            handles = [
                pynvml.nvmlDeviceGetHandleByIndex(index)
                for index in range(pynvml.nvmlDeviceGetCount())
            ]
            while self.monitoring_active:
                memory_info = [pynvml.nvmlDeviceGetMemoryInfo(handle) for handle in handles]
                if self._vram_gpu_index is None:
                    detected_index = self._detect_training_gpu(handles, memory_info)
                    if detected_index is not None:
                        if self._vram_previous_gpu_index is not None and detected_index != self._vram_previous_gpu_index:
                            self.peak_vram = 0
                        self._vram_gpu_index = detected_index
                        self._vram_previous_gpu_index = None

                if self._vram_gpu_index is not None:
                    info = memory_info[self._vram_gpu_index]
                    used_gb = info.used / (1024**3)
                    if used_gb > self.peak_vram: self.peak_vram = used_gb
                    self.root.after(
                        0,
                        self.update_vram_display,
                        used_gb,
                        self.peak_vram,
                        info.total / (1024**3),
                        self._vram_gpu_index,
                    )
                time.sleep(1)
        except pynvml.NVMLError:
            if self.monitoring_active:
                self.root.after(0, lambda: self.vram_label_var.set("VRAM: Monitoring Error"))

    def update_vram_display(self, used, peak, total, gpu_index):
        self.vram_label_var.set(f"GPU {gpu_index} VRAM: {used:.2f} GB / {total:.2f} GB")
        self.peak_vram_label_var.set(f"GPU {gpu_index} Peak VRAM: {peak:.2f} GB")

    def update_loss_graph(self, step=None, loss_value=None):
        if not MATPLOTLIB_AVAILABLE: return
        if step is not None and loss_value is not None:
            step = int(step)
            loss_value = float(loss_value)
            if step < self._last_loss_step:
                return
            if self.loss_data and step == self._last_loss_step:
                self.loss_data[-1] = (step, loss_value)
            else:
                self.loss_data.append((step, loss_value))
            self._last_loss_step = step
        self.ax.clear(); self.setup_graph_style()
        if self.loss_data:
            steps, losses = zip(*self.loss_data)
            self.ax.plot(steps, losses, color='#68bcece8')
        self.canvas.draw()

    def update_progress_bar(self, current, total):
        percentage = (current / total) * 100 if total > 0 else 0
        self.progress_var.set(percentage)
        if total > 0 and self.current_prior_epochs:
            self.progress_label_var.set(
                f"Run epoch {current} of {total}  ·  Overall {self.current_prior_epochs + current} of "
                f"{self.current_prior_epochs + total}"
            )
        else:
            self.progress_label_var.set(f"Epoch {current} of {total}" if total > 0 else "Epochs complete")

    def update_face_refinement_progress(self, current, total):
        self.progress_var.set((current / total) * 100 if total > 0 else 0)
        self.progress_label_var.set(f"Face refinement step {current} of {total}")

    def update_turbo_evaluation_progress(self, current, total):
        self.progress_var.set((current / total) * 100 if total > 0 else 0)
        self.progress_label_var.set(f"Turbo evaluation case {current} of {total}")

    def update_training_counters(self, current_step=None, total_steps=None, current_epoch=None, total_epochs=None):
        if current_step is not None:
            parsed_step = int(current_step)
            if parsed_step >= self.current_step:
                self.current_step = parsed_step
        if total_steps is not None:
            self.current_total_steps = total_steps
        if current_epoch is not None:
            self.current_epoch_num = current_epoch
        if total_epochs is not None:
            self.current_epoch_total = total_epochs

        if self.current_epoch_num > 0 and self.current_epoch_total > 0:
            epoch_text = f"Epoch: {self.current_epoch_num} / {self.current_epoch_total}"
            if self.current_prior_epochs:
                epoch_text += (
                    f"  ·  Overall: {self.current_prior_epochs + self.current_epoch_num}"
                    f" / {self.current_prior_epochs + self.current_epoch_total}"
                )
            self.epoch_counter_var.set(epoch_text)
        elif self.current_prior_epochs:
            self.epoch_counter_var.set(f"Epoch: waiting  ·  Previously completed: {self.current_prior_epochs}")
        else:
            self.epoch_counter_var.set("Epoch: N/A")

        if self.current_step > 0 and self.current_total_steps > 0:
            step_text = f"Step: {self.current_step} / {self.current_total_steps}"
            if self.current_prior_steps:
                step_text += (
                    f"  ·  Overall: {self.current_prior_steps + self.current_step}"
                    f" / {self.current_prior_steps + self.current_total_steps}"
                )
            self.step_counter_var.set(step_text)
        elif self.current_prior_steps:
            self.step_counter_var.set(f"Step: waiting  ·  Previously completed: {self.current_prior_steps}")
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
        if output_widget == self.output_text:
            try:
                depth_strength = float(self.entries["krea2_depth_anchor_weight"].get() or 0)
            except (KeyError, ValueError):
                depth_strength = 0.0
            self.depth_anchor_status_var.set(
                f"Depth anchor: waiting for first training step · strength {depth_strength:g}"
                if self.training_mode_var.get() == "Krea 2" and depth_strength > 0 else "Depth anchor: Off"
            )
        self.last_progress_milestone = 0
        command_display = ' '.join(f'"{part}"' if ' ' in part else part for part in command)
        output_widget.insert(tk.END, f"\n--- Running command ---\n{command_display}\n\n")
        if job_context and job_context.get("attach_to_active"):
            self._record_job_command(command)

        try:
            env = os.environ.copy(); env['PYTHONUNBUFFERED'] = '1'; env['PYTHONUTF8'] = '1'
            project_root = os.getcwd(); src_path = os.path.join(project_root, 'src')
            env['PYTHONPATH'] = f"{src_path}{os.pathsep}{env.get('PYTHONPATH', '')}"

            if self.monitoring_active and PYNVML_AVAILABLE:
                # Cache and Accelerate training commands may use different GPUs.
                # Re-detect for every subprocess and preserve the peak only when
                # the physical GPU stays the same.
                self._vram_previous_gpu_index = self._vram_gpu_index
                self._vram_gpu_index = None
                try:
                    self._vram_baseline = [
                        pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(index)).used
                        for index in range(pynvml.nvmlDeviceGetCount())
                    ]
                except pynvml.NVMLError:
                    pass
                self.vram_label_var.set("VRAM: Detecting process GPU...")

            process_options = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "cwd": project_root,
                "bufsize": 0,
                "env": env,
            }
            if sys.platform == "win32":
                process_options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                process_options["start_new_session"] = True

            self.current_process = subprocess.Popen(command, **process_options)
            # Popen's text mode applies universal-newline conversion, changing the
            # carriage returns used by tqdm into newlines. Preserve them so the
            # console can update one progress line in place.
            self.current_process.stdout = io.TextIOWrapper(
                self.current_process.stdout,
                encoding='utf-8',
                errors='replace',
                newline='',
            )
            self._update_run_mode_controls()
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
        self.current_prior_steps = 0
        self.current_prior_epochs = 0
        self._last_loss_step = 0
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
            # Milestone records are useful for the main training bar, but not for
            # every short setup/cache tqdm bar emitted by the same process.
            percent_match = (
                re.search(r"(?<!\d)(\d{1,3})%", clean_line)
                if clean_line.lower().startswith("steps:")
                else None
            )
            percent = int(percent_match.group(1)) if percent_match else None
            milestone = (percent // 10) * 10 if percent is not None else None
            if milestone is not None and 10 <= milestone < 100 and milestone > self.last_progress_milestone:
                # Keep one permanent console record per 10% completed. The next
                # carriage-return update starts a new replaceable line below it.
                self.last_progress_milestone = milestone
                self.last_line_was_progress = False
            else:
                self.last_line_was_progress = True
        else:
            output_widget.insert(tk.END, line)
            self.last_line_was_progress = False

        # Only auto-scroll if user was already at the bottom
        if is_at_bottom:
            output_widget.see(tk.END)

    @staticmethod
    def _parse_main_training_progress(text):
        """Parse only the trainer's named `steps:` bar, never model-loading tqdm bars."""
        match = re.search(r"steps:\s*\d{1,3}%.*?(\d+)\s*/\s*(\d+)\s*\[", text)
        if not match:
            return None
        number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        def metric(name):
            found = re.search(rf"{re.escape(name)}=({number})", text)
            return float(found.group(1)) if found else None
        return {
            "step": int(match.group(1)),
            "total": int(match.group(2)),
            "loss": metric("avr_loss"),
            "depth_loss": metric("loss/depth_anchor"),
            "diffusion_loss": metric("loss/diffusion"),
        }

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
                chunk = None
                if char == '\n':
                    # Keep CRLF together so normal Windows log lines are not
                    # mistaken for tqdm's standalone carriage-return updates.
                    chunk, buffer = buffer, ""
                elif len(buffer) >= 2 and buffer[-2] == '\r':
                    chunk, buffer = buffer[:-1], buffer[-1]

                if chunk is not None:
                    self.root.after(0, self.process_console_output, chunk, output_widget)
                    if output_widget == self.output_text:
                        training_progress = self._parse_main_training_progress(chunk)
                        face_step_match = re.search(r"^step=(\d+)/(\d+)\s", chunk)
                        face_refinement_active = bool(
                            self._staged_run
                            and (self._staged_run.get("previous_settings") or {}).get("stage_type") == "face_refinement"
                        )
                        face_evaluation_active = bool(self._face_eval_context)
                        eval_prompt_match = re.search(r"\[(\d+)\]\s+Prompt:", chunk)
                        if face_evaluation_active and eval_prompt_match:
                            completed = int(eval_prompt_match.group(1)) + 1
                            self.root.after(0, self.update_turbo_evaluation_progress, completed, int(self._face_eval_context.get("cases", completed)))
                        if face_step_match:
                            self.root.after(
                                0, self.update_training_counters,
                                int(face_step_match.group(1)), int(face_step_match.group(2)), None, None,
                            )
                            self.root.after(0, self.update_face_refinement_progress, int(face_step_match.group(1)), int(face_step_match.group(2)))
                        pose_metric_match = re.search(r"^pose_metric=([a-z_]+)\s+similarity=([\d.]+)\s+evaluations=(\d+)\s+target=([\d.]+)", chunk)
                        if pose_metric_match:
                            pose_label = pose_metric_match.group(1).replace("_", " ").title()
                            self.root.after(0, self.progress_label_var.set, f"{pose_label}: {float(pose_metric_match.group(2)):.3f} / target {float(pose_metric_match.group(4)):.3f} · {pose_metric_match.group(3)} evaluations")
                        stop_match = re.search(r"^(?:early_stop|pose_stop)=([a-z_]+)", chunk)
                        if stop_match:
                            self.root.after(0, self.progress_label_var.set, f"Face refinement stopping: {stop_match.group(1).replace('_', ' ')}")
                        if training_progress and not face_refinement_active and not face_evaluation_active:
                            self.root.after(
                                0,
                                self.update_training_counters,
                                training_progress["step"],
                                training_progress["total"],
                                None,
                                None,
                            )

                        face_loss_match = re.search(r"(?:avr_loss|loss)=([-+\d.eE]+)", chunk) if face_step_match else None
                        loss_value = training_progress["loss"] if training_progress else float(face_loss_match.group(1)) if face_loss_match else None
                        graph_step = training_progress["step"] if training_progress else int(face_step_match.group(1)) if face_step_match else 0
                        if loss_value is not None and graph_step > 0:
                            self._last_loss_value = loss_value
                            self.root.after(0, self.update_loss_graph, graph_step, loss_value)
                        if "Loading frozen depth perceptor:" in chunk:
                            self.root.after(0, self.depth_anchor_status_var.set, "Depth anchor: loading model and first target…")
                        if training_progress and training_progress["depth_loss"] is not None:
                            try:
                                strength = float(self.entries["krea2_depth_anchor_weight"].get() or 0)
                            except (KeyError, ValueError):
                                strength = 0.0
                            depth_loss = training_progress["depth_loss"]
                            self.root.after(
                                0,
                                self.depth_anchor_status_var.set,
                                f"Depth anchor: active · loss {depth_loss:.4f} · weighted {depth_loss * strength:.5f}",
                            )

                        epoch_match = re.search(r"epoch\s*=?\s*(\d+)\s*/\s*(\d+)", chunk, re.IGNORECASE)
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
            self._staged_run = None
            self.stop_all_activity(); return
        if self.command_sequence:
            self.loss_data.clear()
            self.current_step = 0
            self._last_loss_step = 0
            self.update_loss_graph()
            next_command = self.command_sequence.pop(0)
            self.run_process(next_command, self._run_next_command_in_sequence, self.output_text, job_context={"attach_to_active": True})
        elif self._staged_run:
            previous = self._staged_run.get("previous_settings")
            if previous:
                self._rename_final_training_artifacts(previous)
            self._advance_staged_training()
        else:
            if self._active_job and self._active_job.get("kind") == "training":
                self._rename_final_training_artifacts(self._active_job.get("settings_snapshot", {}))
            self._finalize_active_job("completed", 0)
            self.output_text.insert(tk.END, f"\n--- All steps completed successfully. ---\n")
            self.stop_all_activity()

    def _commands_for_settings(self, settings):
        python_executable = sys.executable or "python"
        mode = settings.get("training_mode", "Wan 2.2")
        if mode == "Wan 2.2":
            cache_commands = wan_backend.build_cache_commands(
                settings, python_executable, temp_config_fn=self._create_temp_cache_config
            )
            training_commands = wan_backend.build_commands(settings)
        elif mode == "Krea 2":
            cache_commands = krea2_backend.build_cache_commands(settings, python_executable)
            training_commands = krea2_backend.build_commands(settings)
        else:
            cache_commands = flux2_backend.build_cache_commands(settings, python_executable)
            training_commands = flux2_backend.build_commands(settings)
        return cache_commands + training_commands

    @staticmethod
    def _final_state_path(settings):
        run_name = MusubiTunerGUI._effective_run_name(settings)
        if not settings.get("rename_final_artifacts_to_epoch", True):
            return Path(settings["output_dir"]) / run_name / f"{run_name}-state"
        epoch_text = str(settings.get("max_train_epochs", "")).strip()
        if epoch_text.isdigit():
            return Path(settings["output_dir"]) / run_name / f"{run_name}-{int(epoch_text):06d}-state"
        return Path(settings["output_dir"]) / run_name / f"{run_name}-state"

    @staticmethod
    def _candidate_final_state_paths(settings):
        run_name = MusubiTunerGUI._effective_run_name(settings)
        base_dir = Path(settings["output_dir"]) / run_name
        epoch_text = str(settings.get("max_train_epochs", "")).strip()
        numbered = None
        if epoch_text.isdigit():
            numbered = base_dir / f"{run_name}-{int(epoch_text):06d}-state"
        legacy = base_dir / f"{run_name}-state"

        preferred = MusubiTunerGUI._final_state_path(settings)
        candidates = [preferred]
        for extra in (numbered, legacy):
            if extra is not None and extra not in candidates:
                candidates.append(extra)
        return candidates

    @staticmethod
    def _candidate_final_model_paths(settings):
        run_name = MusubiTunerGUI._effective_run_name(settings)
        base_dir = Path(settings["output_dir"]) / run_name
        candidates = [base_dir / f"{run_name}.safetensors"]
        epoch_text = str(settings.get("max_train_epochs", "")).strip()
        if epoch_text.isdigit():
            candidates.insert(0, base_dir / f"{run_name}-{int(epoch_text):06d}.safetensors")
        return candidates

    def start_staged_training(self):
        if self.current_process:
            messagebox.showwarning("Staged Run", "A process is already running.")
            return
        stages = [dict(item) for item in self._staged_training_config if item.get("enabled")]
        if not stages:
            self._open_staged_training_dialog()
            return

        def valid_stage(item):
            steps = str(item.get("steps", "")).strip()
            epochs = str(item.get("epochs", "")).strip()
            limit = steps or epochs
            has_source = item.get("type", "standard") == "face_refinement" or os.path.isfile(item.get("dataset_config", ""))
            return has_source and limit.isdigit() and int(limit) >= 1

        invalid_stage = next(
            (item for item in stages if not valid_stage(item)),
            None,
        )
        if invalid_stage:
            messagebox.showerror("Staged Run", f"The {self._stage_label_text(invalid_stage)} stage has an invalid TOML path or training limit.")
            return
        first_is_face = stages[0].get("type", "standard") == "face_refinement"
        if first_is_face:
            face_config = self._face_refinement_config or {}
            if face_config.get("input_mode") != "existing_lora":
                messagebox.showerror(
                    "Staged Run",
                    "Face Refinement can be the first stage only when ‘Refine an existing Krea 2 LoRA’ is selected in its settings.",
                )
                return
            try:
                from musubi_tuner.face_refinement.lora_validation import validate_krea2_lora
                validate_krea2_lora(face_config.get("input_lora", ""))
            except ValueError as exc:
                messagebox.showerror("Staged Run", str(exc)); return
        if any(item.get("type") == "face_refinement" for item in stages) and self.training_mode_var.get() != "Krea 2":
            messagebox.showerror("Staged Run", "Face Refinement is currently available only in Krea 2 mode.")
            return
        if any(item.get("type") == "face_refinement" for item in stages) and not self._check_face_refinement_dependencies():
            return
        if not first_is_face:
            first_dataset = stages[0]["dataset_config"]
            self.entries["dataset_config"].delete(0, tk.END)
            self.entries["dataset_config"].insert(0, first_dataset)
        self.update_button_states()
        if self.start_btn["state"] == "disabled":
            messagebox.showerror("Validation Error", "Complete the required model and training fields before starting the staged run.")
            return

        base_settings = self.get_settings()
        if base_settings.get("training_mode") == "Wan 2.2":
            separate_wan_runs = (
                base_settings.get("train_low_noise")
                and base_settings.get("train_high_noise")
                and (str(base_settings.get("network_dim_high") or "").strip() or str(base_settings.get("network_alpha_high") or "").strip())
            )
            if separate_wan_runs:
                messagebox.showerror(
                    "Staged Run",
                    "Staged continuation cannot map one resume state to two separate Wan low/high-noise runs. Use a combined run or stage each noise model separately.",
                )
                return
        if not self._check_logging_dependencies(base_settings.get("log_with")):
            return
        if not self._check_compile_dependencies(base_settings):
            return
        self.loss_data.clear()
        self.current_step = 0
        self._last_loss_step = 0
        self.update_loss_graph()
        self.start_vram_monitor()
        self._start_sample_watcher()
        self.progress_var.set(0)
        self.progress_label_var.set("Starting staged run...")
        self.output_text.delete("1.0", tk.END)
        self.command_sequence = []
        self._staged_run = {
            "base_settings": base_settings,
            "base_output_name": base_settings["output_name"],
            "stages": stages,
            "next_index": 0,
            "previous_settings": None,
        }
        self._begin_job(
            "staged_training",
            "Staged resolution training",
            settings=base_settings,
            note=" → ".join(f"{self._stage_label_text(item)} × {self._staged_limit_text(item)}" for item in stages),
        )
        self._advance_staged_training()

    def _advance_staged_training(self):
        run = self._staged_run
        if not run:
            return
        previous = run.get("previous_settings")
        next_index = run["next_index"]
        state_path = None
        input_lora = None
        if previous is None and next_index < len(run["stages"]) and run["stages"][next_index].get("type") == "face_refinement":
            configured_input = str((self._face_refinement_config or {}).get("input_lora", "")).strip()
            input_lora = Path(configured_input) if configured_input else None
        if previous is not None and next_index < len(run["stages"]):
            previous_type = previous.get("stage_type", "standard")
            next_type = run["stages"][next_index].get("type", "standard")
            if previous_type == "face_refinement":
                input_lora = Path(previous["face_output_path"])
            elif next_type == "face_refinement":
                input_lora = next((path for path in self._candidate_final_model_paths(previous) if path.is_file()), None)
            else:
                state_candidates = self._candidate_final_state_paths(previous)
                state_path = next((path for path in state_candidates if path.is_dir()), None)
            if (next_type == "face_refinement" and input_lora is None) or (previous_type == "face_refinement" and not input_lora.is_file()) or (next_type == "standard" and previous_type == "standard" and state_path is None):
                self.output_text.insert(
                    tk.END,
                    "\n--- Expected staged handoff artifact was not created. ---\n",
                )
                self._finalize_active_job("failed", -1)
                self._staged_run = None
                self.stop_all_activity()
                return

        if next_index >= len(run["stages"]):
            if previous is not None and previous.get("stage_type") == "face_refinement" and not Path(previous.get("face_output_path", "")).is_file():
                self.output_text.insert(tk.END, "\n--- Face Refinement finished without creating its expected LoRA. ---\n")
                self._finalize_active_job("failed", -1)
                self._staged_run = None
                self.stop_all_activity()
                return
            self._staged_run = None
            self._finalize_active_job("completed", 0)
            self.output_text.insert(tk.END, "\n--- All staged training runs completed successfully. ---\n")
            self.stop_all_activity()
            return

        stage = run["stages"][next_index]
        settings = dict(run["base_settings"])
        stage_type = stage.get("type", "standard")
        settings["dataset_config"] = stage.get("dataset_config", "") if stage_type == "standard" else settings.get("dataset_config", "")
        stage_steps = str(stage.get("steps", "")).strip()
        if stage_steps:
            settings["max_train_steps"] = stage_steps
            settings["max_train_epochs"] = ""
        else:
            settings["max_train_steps"] = ""
            settings["max_train_epochs"] = str(stage["epochs"])
        stage_label = self._stage_label_text(stage)
        settings["output_name"] = f"{run['base_output_name']}-{stage_label}"
        settings["stage_type"] = stage_type
        settings["save_state"] = stage_type == "standard"
        settings["recache_latents"] = stage_type == "standard" and bool(self._staged_recache_latents)
        settings["recache_text"] = stage_type == "standard" and (bool(settings.get("recache_text")) if next_index == 0 else False)
        settings["resume_path"] = str(state_path) if state_path is not None else ""
        if input_lora is not None and stage_type == "standard":
            settings["network_weights"] = str(input_lora)
        else:
            settings["network_weights"] = "" if settings["resume_path"] else settings.get("network_weights", "")
        settings["sample_prompts"] = self._build_sample_prompts_txt()

        run["previous_settings"] = settings
        run["next_index"] = next_index + 1
        self.set_values(settings)
        self.run_status_var.set(f"🟢 Stage {next_index + 1}/{len(run['stages'])}: {stage_label}")
        stage_limit = self._staged_limit_text(stage)
        self.progress_label_var.set(f"Stage {next_index + 1}/{len(run['stages'])} · {stage_label} · {stage_limit}")
        self.output_text.insert(
            tk.END,
            f"\n=== Stage {next_index + 1}/{len(run['stages'])}: {stage_label} for {stage_limit} ===\n"
            f"Type: {stage_type}\n"
            f"Dataset: {stage.get('dataset_config') or 'not used'}\n"
            f"Handoff: {settings['resume_path'] or settings.get('network_weights') or input_lora or 'new training state'}\n",
        )
        if stage_type == "face_refinement":
            face_config = dict(self._face_refinement_config)
            face_config["steps"] = int(stage_steps)
            self.current_step = 0; self.current_total_steps = int(stage_steps)
            self.current_epoch_num = 0; self.current_epoch_total = 0
            self.update_training_counters(0, int(stage_steps), 0, 0)
            self.progress_var.set(0)
            output_path = krea2_face_backend.output_path(run["base_settings"], stage_label)
            if input_lora is None:
                raise RuntimeError("Face Refinement has no input LoRA. Select an existing LoRA or place it after a standard stage.")
            if output_path.resolve() == input_lora.resolve():
                raise RuntimeError("Face Refinement output would overwrite its input LoRA. Change the output name or stage label.")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            prompts_path = output_path.parent / "face_refinement_prompts.json"
            from musubi_tuner.face_refinement.lora_validation import render_trigger_prompts
            pose_plan = copy.deepcopy(face_config.get("pose_plan") or {})
            if face_config.get("pose_aware") and pose_plan.get("enabled"):
                from musubi_tuner.face_refinement.pose_plan import normalize_pose_plan, weighted_prompt_records
                excluded_for_plan = set(face_config.get("excluded_reference_images") or [])
                effective_counts = {}
                for item in face_config.get("preflight_report", {}).get("scored_images", []):
                    if item["path"] not in excluded_for_plan:
                        bucket = item.get("bucket", "uncertain"); effective_counts[bucket] = effective_counts.get(bucket, 0) + 1
                pose_plan, plan_warnings = normalize_pose_plan(pose_plan, effective_counts, int(face_config.get("pose_min_references", 2)))
                prompt_records = weighted_prompt_records(pose_plan)
                rendered_prompts = render_trigger_prompts([item["prompt"] for item in prompt_records], face_config.get("trigger_word", ""))
                for item, rendered in zip(prompt_records, rendered_prompts): item["prompt"] = rendered
                prompt_payload = {"prompts": rendered_prompts, "prompt_records": prompt_records, "pose_plan": pose_plan, "warnings": plan_warnings}
            else:
                rendered_prompts = render_trigger_prompts(face_config["prompts"], face_config.get("trigger_word", ""))
                prompt_payload = {"prompts": rendered_prompts}
            prompts_path.write_text(json.dumps(prompt_payload, indent=2), encoding="utf-8")
            excluded = set(face_config.get("excluded_reference_images") or [])
            scored_references = face_config.get("preflight_report", {}).get("scored_images", [])
            reference_entries = [
                {"path": item["path"], "pose": item.get("bucket", "uncertain"), "pose_confidence": item.get("confidence", 0.0), "enabled": item["path"] not in excluded}
                for item in scored_references
            ]
            if not any(item["enabled"] for item in reference_entries):
                raise RuntimeError("No enabled face references remain. Review the face-analysis results and enable at least one detected face.")
            manifest_path = output_path.parent / "face_refinement_references.json"
            manifest_path.write_text(json.dumps({"reference_images": reference_entries}, indent=2), encoding="utf-8")
            face_config["reference_manifest"] = str(manifest_path)
            settings["python_executable"] = sys.executable or "python"
            settings["face_output_path"] = str(output_path)
            run["previous_settings"] = settings
            self.command_sequence = [krea2_face_backend.build_command(settings, face_config, input_lora, output_path, prompts_path)]
        else:
            self.command_sequence = self._commands_for_settings(settings)
        self._run_next_command_in_sequence(0)

    def _check_logging_dependencies(self, log_with):
        if log_with in ["wandb", "all"]:
            try: import wandb
            except Exception: messagebox.showerror("Missing Dependency", "Please run: pip install wandb"); return False
        if log_with in ["tensorboard", "all"]:
            try: import tensorboard
            except Exception: messagebox.showerror("Missing Dependency", "Please run: pip install tensorboard"); return False
        return True

    def _check_face_refinement_dependencies(self, parent=None):
        import importlib.util

        missing = [name for name in ("onnx", "onnxruntime", "insightface") if importlib.util.find_spec(name) is None]
        if not missing:
            return True
        messagebox.showerror(
            "Face Refinement dependencies",
            "Face Refinement is optional and needs additional face-analysis packages.\n\n"
            f"Missing: {', '.join(missing)}\n\n"
            "Install them in this project's environment with:\n"
            'pip install -e ".[face_refinement]"',
            parent=parent or self.root,
        )
        return False

    def _check_compile_dependencies(self, settings):
        if not settings.get("compile") or settings.get("compile_backend", "inductor") != "inductor":
            return True
        try:
            from torch.utils._triton import has_triton

            if has_triton():
                return True
        except Exception:
            pass
        messagebox.showerror(
            "Torch Compile unavailable",
            "The Inductor backend cannot find a working Triton installation.\n\n"
            "Effect: training would fail when the first DiT block is compiled.\n\n"
            "Disable “Enable Torch Compile” to train normally, or install a Windows-compatible "
            "Triton build in this virtual environment before enabling it.",
        )
        return False

    def start_training(self):
        self._staged_run = None
        self.update_button_states(); settings = self.get_settings()
        if not self._check_logging_dependencies(settings.get("log_with")): return
        if not self._check_compile_dependencies(settings): return
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
            try:
                noise_strength = float(settings.get("krea2_weight_noise_sigma") or 0)
                depth_strength = float(settings.get("krea2_depth_anchor_weight") or 0)
                depth_size = int(settings.get("krea2_depth_anchor_input_size") or 518)
                if noise_strength < 0 or depth_strength < 0:
                    raise ValueError("strengths cannot be negative")
                if depth_size <= 0 or depth_size % 14:
                    raise ValueError("depth resolution must be a positive multiple of 14")
            except ValueError as exc:
                messagebox.showerror("Validation Error", f"Invalid Krea 2 generalization setting: {exc}.")
                return
            if depth_strength > 0 and not messagebox.askokcancel(
                "Experimental Depth Anchor",
                "Depth anchoring is experimental for Krea 2. It downloads a frozen depth model on first use, "
                "decodes predicted images during every training step, and can substantially increase VRAM use and training time.\n\n"
                "Run a short comparison first and keep a baseline with the same seed and dataset. Continue?",
            ):
                return
        # Warn if sample frequency is set but no prompts were added
        wants_samples = (settings.get("sample_every_n_epochs") or settings.get("sample_every_n_steps") or settings.get("sample_at_first"))
        if wants_samples and self._count_enabled_sample_prompts() == 0:
            messagebox.showwarning("No Active Sample Prompts", "You set a sample frequency but have no enabled prompts in the Samples tab.\nNo samples will be generated.\n\nEnable at least one saved prompt or add a new one.")

        self.loss_data.clear(); self.current_step = 0; self._last_loss_step = 0
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
            self._staged_run = None
            self.run_status_var.set("🛑 Stopping Run")
            self.progress_label_var.set("Stopping current process...")
            self.stop_btn.config(state="disabled")
            process = self.current_process
            threading.Thread(target=self._terminate_process_tree, args=(process,), daemon=True).start()

    def _terminate_process_tree(self, process):
        try:
            if sys.platform == "win32":
                result = subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    check=False,
                )
                if result.returncode != 0 and process.poll() is None:
                    raise RuntimeError(result.stdout.strip() or f"taskkill exited with code {result.returncode}")
                return

            process_group = os.getpgid(process.pid)
            os.killpg(process_group, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process_group, signal.SIGKILL)
        except Exception as exc:
            if process.poll() is not None:
                return
            try:
                if PSUTIL_AVAILABLE:
                    parent = psutil.Process(process.pid)
                    descendants = parent.children(recursive=True)
                    for child in descendants:
                        child.kill()
                    parent.kill()
                elif process.poll() is None:
                    process.kill()
            except Exception as fallback_exc:
                message = f"Could not stop the training process tree:\n{exc}\n\nFallback also failed:\n{fallback_exc}"
                self.root.after(0, messagebox.showerror, "Stop Training Failed", message)
                self.root.after(0, self.stop_btn.config, {"state": "normal"})

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
