import copy
import os
import subprocess
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import toml

try:
    import tomlkit
    from tomlkit.items import Whitespace

    TOMLKIT_AVAILABLE = True
except ImportError:
    tomlkit = None
    Whitespace = None
    TOMLKIT_AVAILABLE = False


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv"}
GENERAL_KEYS = ("caption_extension", "batch_size", "enable_bucket", "bucket_no_upscale")
COMMON_DATASET_KEYS = (
    "resolution",
    "cache_directory",
    "num_repeats",
    "batch_size",
    "caption_extension",
    "enable_bucket",
    "bucket_no_upscale",
)
IMAGE_KEYS = (
    "image_directory",
    "image_jsonl_file",
    "control_directory",
    "multiple_target",
    "no_resize_control",
    "control_resolution",
)
VIDEO_KEYS = (
    "video_directory",
    "video_jsonl_file",
    "control_directory",
    "target_frames",
    "frame_extraction",
    "frame_stride",
    "frame_sample",
    "max_frames",
    "source_fps",
)


def _new_table():
    return tomlkit.table() if TOMLKIT_AVAILABLE else {}


def _parse_toml(text):
    return tomlkit.parse(text) if TOMLKIT_AVAILABLE else toml.loads(text)


def _dump_toml(document):
    return tomlkit.dumps(document) if TOMLKIT_AVAILABLE else toml.dumps(document)


def _plain_document(document):
    return toml.loads(_dump_toml(document))


def _dataset_kind(table):
    if "video_directory" in table or "video_jsonl_file" in table:
        return "video"
    return "image"


def _source_details(table, kind):
    directory_key = f"{kind}_directory"
    jsonl_key = f"{kind}_jsonl_file"
    if table.get(jsonl_key):
        return "JSONL file", str(table.get(jsonl_key))
    return "Directory", str(table.get(directory_key, ""))


def _positive_int(value, label, allow_blank=False):
    value = str(value).strip()
    if allow_blank and not value:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a whole number.") from exc
    if parsed < 1:
        raise ValueError(f"{label} must be at least 1.")
    return parsed


def _positive_float(value, label, allow_blank=False):
    value = str(value).strip()
    if allow_blank and not value:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number.") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be greater than 0.")
    return parsed


def _integer_list(value, label):
    pieces = [piece.strip() for piece in str(value).replace(";", ",").split(",") if piece.strip()]
    if not pieces:
        raise ValueError(f"{label} must contain at least one whole number.")
    values = [_positive_int(piece, label) for piece in pieces]
    return values


def _set_optional(table, key, value):
    if value in ("", None):
        table.pop(key, None)
    else:
        table[key] = value


def _trim_tomlkit_table_separator(table):
    """Remove separators retained by a parsed table before adding it to a new AoT."""
    if not TOMLKIT_AVAILABLE:
        return
    body = table.value.body
    while body and body[-1][0] is None and isinstance(body[-1][1], Whitespace):
        body.pop()


class BuilderToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.window = None
        widget.bind("<Enter>", self.show, add="+")
        widget.bind("<Leave>", self.hide, add="+")

    def show(self, _event=None):
        self.hide()
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry(
            f"+{self.widget.winfo_rootx() + 24}+{self.widget.winfo_rooty() + self.widget.winfo_height() + 6}"
        )
        tk.Label(
            self.window,
            text=self.text,
            justify="left",
            background="#E2E8F0",
            foreground="#0F172A",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9),
            wraplength=420,
            padx=8,
            pady=6,
        ).pack()

    def hide(self, _event=None):
        if self.window:
            self.window.destroy()
        self.window = None


class DatasetConfigBuilder:
    """Visual editor for the common dataset TOML options with a raw escape hatch."""

    def __init__(self, parent, initial_path="", on_use=None, colors=None):
        self.parent = parent
        self.on_use = on_use
        self.colors = colors or {}
        self.path = Path(initial_path).expanduser() if initial_path else None
        self.document = None
        self.datasets = []
        self.selected_index = None
        self._active_tab = 0
        self._changing_tab = False
        self._suspend_dirty = True
        self.is_dirty = False

        self.window = tk.Toplevel(parent)
        self.window.title("Dataset Config Builder")
        self.window.geometry("1040x760")
        self.window.minsize(880, 640)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self.window.bind("<MouseWheel>", self._route_popup_mousewheel)
        self.window.bind("<Button-4>", self._route_popup_mousewheel)
        self.window.bind("<Button-5>", self._route_popup_mousewheel)

        self.path_var = tk.StringVar(value=str(self.path or ""))
        self.status_var = tk.StringVar(value="Build a dataset configuration, then validate or save it.")
        self.general_vars = {
            "caption_extension": tk.StringVar(value=".txt"),
            "batch_size": tk.StringVar(value="1"),
            "enable_bucket": tk.BooleanVar(value=True),
            "bucket_no_upscale": tk.BooleanVar(value=False),
        }
        self.dataset_vars = {
            "source_format": tk.StringVar(value="Directory"),
            "source": tk.StringVar(),
            "cache_directory": tk.StringVar(),
            "width": tk.StringVar(value="512"),
            "height": tk.StringVar(value="512"),
            "num_repeats": tk.StringVar(value="1"),
            "batch_size": tk.StringVar(),
            "caption_extension": tk.StringVar(),
            "enable_bucket_override": tk.StringVar(value="Use general"),
            "bucket_no_upscale_override": tk.StringVar(value="Use general"),
            "control_directory": tk.StringVar(),
            "multiple_target": tk.BooleanVar(value=False),
            "no_resize_control": tk.BooleanVar(value=False),
            "control_width": tk.StringVar(),
            "control_height": tk.StringVar(),
            "target_frames": tk.StringVar(value="1"),
            "frame_extraction": tk.StringVar(value="head"),
            "frame_stride": tk.StringVar(),
            "frame_sample": tk.StringVar(),
            "max_frames": tk.StringVar(),
            "source_fps": tk.StringVar(),
        }

        self._build_ui()
        self._install_dirty_tracking()
        if self.path and self.path.is_file():
            self._load_path(self.path)
        else:
            self._load_text(
                '[general]\ncaption_extension = ".txt"\nbatch_size = 1\nenable_bucket = true\n'
                "bucket_no_upscale = false\n\n[[datasets]]\nresolution = [512, 512]\n"
                'image_directory = ""\nnum_repeats = 1\n'
            )
            self._mark_clean("New configuration. Changes stay staged until you save.")
        self._suspend_dirty = False
        self.window.grab_set()

    def _build_ui(self):
        outer = ttk.Frame(self.window, padding=12)
        outer.pack(fill="both", expand=True)

        file_row = ttk.Frame(outer)
        file_row.pack(fill="x", pady=(0, 10))
        ttk.Label(file_row, text="TOML file:", width=12).pack(side="left")
        ttk.Entry(file_row, textvariable=self.path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(file_row, text="Open", command=self._open_file).pack(side="left", padx=(6, 0))
        external_button = ttk.Button(file_row, text="External Editor", command=self._open_external_editor)
        external_button.pack(side="left", padx=(6, 0))
        BuilderToolTip(external_button, "Opens the saved TOML in your system editor. Save first if this is a new configuration.")

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)
        self.builder_tab = ttk.Frame(self.notebook, padding=8)
        self.raw_tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.builder_tab, text="Builder")
        self.notebook.add(self.raw_tab, text="Raw TOML")
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self._build_builder_tab()
        self._build_raw_tab()

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(10, 0))
        ttk.Label(footer, textvariable=self.status_var, style="Muted.TLabel").pack(side="left", fill="x", expand=True)
        ttk.Button(footer, text="Discard / Close", command=self._close).pack(side="right")
        ttk.Button(footer, text="Save As", command=self._save_as).pack(side="right", padx=(6, 0))
        ttk.Button(footer, text="Validate", command=self._validate_action).pack(side="right", padx=(6, 0))
        ttk.Button(footer, text="Save and Use", style="Accent.TButton", command=self._save_and_use).pack(
            side="right", padx=(6, 0)
        )

    def _build_builder_tab(self):
        general = ttk.LabelFrame(self.builder_tab, text="General defaults", padding=8)
        general.pack(fill="x", pady=(0, 8))
        for column in range(4):
            general.columnconfigure(column, weight=1 if column in (1, 3) else 0)

        self._labeled_entry(
            general,
            "Caption extension",
            self.general_vars["caption_extension"],
            0,
            0,
            "Caption files beside each image or video use this extension, normally .txt.",
        )
        self._labeled_entry(
            general,
            "Batch size",
            self.general_vars["batch_size"],
            0,
            2,
            "Number of samples processed together. Higher values use more VRAM.",
        )
        bucket = ttk.Checkbutton(general, text="Enable aspect-ratio buckets", variable=self.general_vars["enable_bucket"])
        bucket.grid(row=1, column=0, columnspan=2, sticky="w", pady=(7, 0))
        BuilderToolTip(bucket, "Groups media with similar aspect ratios to reduce cropping and distortion.")
        upscale = ttk.Checkbutton(
            general,
            text="Do not upscale smaller media",
            variable=self.general_vars["bucket_no_upscale"],
        )
        upscale.grid(row=1, column=2, columnspan=2, sticky="w", pady=(7, 0))
        BuilderToolTip(upscale, "Prevents bucket processing from enlarging media below the target resolution.")

        split = ttk.PanedWindow(self.builder_tab, orient=tk.HORIZONTAL)
        split.pack(fill="both", expand=True)
        list_panel = ttk.LabelFrame(split, text="Datasets", padding=6)
        editor_panel = ttk.LabelFrame(split, text="Selected dataset", padding=8)
        split.add(list_panel, weight=2)
        split.add(editor_panel, weight=5)

        self.dataset_tree = ttk.Treeview(
            list_panel,
            columns=("type", "resolution", "source"),
            show="headings",
            selectmode="browse",
            height=12,
        )
        self.dataset_tree.heading("type", text="Type")
        self.dataset_tree.heading("resolution", text="Resolution")
        self.dataset_tree.heading("source", text="Source")
        self.dataset_tree.column("type", width=65, stretch=False)
        self.dataset_tree.column("resolution", width=85, stretch=False)
        self.dataset_tree.column("source", width=210)
        tree_scroll = ttk.Scrollbar(list_panel, orient="vertical", command=self.dataset_tree.yview)
        self.dataset_tree.configure(yscrollcommand=tree_scroll.set)
        self.dataset_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        self.dataset_tree.bind("<<TreeviewSelect>>", self._select_dataset)
        self._bind_popup_wheel(self.dataset_tree)

        list_buttons = ttk.Frame(self.builder_tab)
        list_buttons.pack(fill="x", pady=(7, 0))
        ttk.Button(list_buttons, text="+ Image", command=lambda: self._add_dataset("image")).pack(side="left")
        ttk.Button(list_buttons, text="+ Video", command=lambda: self._add_dataset("video")).pack(side="left", padx=(5, 0))
        ttk.Button(list_buttons, text="Duplicate", command=self._duplicate_dataset).pack(side="left", padx=(12, 0))
        ttk.Button(list_buttons, text="Remove", style="Danger.TButton", command=self._remove_dataset).pack(
            side="left", padx=(5, 0)
        )
        ttk.Button(list_buttons, text="Move up", command=lambda: self._move_dataset(-1)).pack(side="left", padx=(12, 0))
        ttk.Button(list_buttons, text="Move down", command=lambda: self._move_dataset(1)).pack(side="left", padx=(5, 0))

        self.editor_canvas = tk.Canvas(
            editor_panel,
            highlightthickness=0,
            background=self.colors.get("page", "#111827"),
        )
        editor_scroll = ttk.Scrollbar(editor_panel, orient="vertical", command=self.editor_canvas.yview)
        self.editor_content = ttk.Frame(self.editor_canvas)
        self._editor_window = self.editor_canvas.create_window((0, 0), window=self.editor_content, anchor="nw")
        self.editor_canvas.configure(yscrollcommand=editor_scroll.set)
        self.editor_canvas.pack(side="left", fill="both", expand=True)
        editor_scroll.pack(side="right", fill="y")
        self.editor_content.bind(
            "<Configure>",
            lambda _event: self.editor_canvas.configure(scrollregion=self.editor_canvas.bbox("all")),
        )
        self.editor_canvas.bind(
            "<Configure>",
            lambda event: self.editor_canvas.itemconfigure(self._editor_window, width=event.width),
        )
        self._build_dataset_editor()

    def _build_dataset_editor(self):
        content = self.editor_content
        content.columnconfigure(1, weight=1)
        row = 0
        self.dataset_type_label = ttk.Label(content, text="Image dataset", style="PageTitle.TLabel")
        self.dataset_type_label.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 8))
        row += 1

        ttk.Label(content, text="Source format").grid(row=row, column=0, sticky="w", pady=4)
        source_format = ttk.Combobox(
            content,
            textvariable=self.dataset_vars["source_format"],
            values=("Directory", "JSONL file"),
            state="readonly",
            width=16,
        )
        source_format.grid(row=row, column=1, sticky="ew", pady=4)
        source_format.bind("<<ComboboxSelected>>", lambda _event: self._update_source_button())
        self._bind_popup_wheel(source_format)
        BuilderToolTip(source_format, "Use a media folder for normal datasets or a JSONL manifest for custom mappings.")
        row += 1

        ttk.Label(content, text="Media source").grid(row=row, column=0, sticky="w", pady=4)
        source_frame = ttk.Frame(content)
        source_frame.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Entry(source_frame, textvariable=self.dataset_vars["source"]).pack(side="left", fill="x", expand=True)
        self.source_browse_button = ttk.Button(source_frame, text="Browse", command=self._browse_source)
        self.source_browse_button.pack(side="right", padx=(5, 0))
        BuilderToolTip(self.source_browse_button, "Select the directory or JSONL file containing this dataset.")
        row += 1

        row = self._path_row(
            content,
            row,
            "Cache directory",
            self.dataset_vars["cache_directory"],
            "Stores latent and text-encoder caches. Each dataset must use a different cache directory.",
        )

        resolution = ttk.Frame(content)
        resolution.grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Label(resolution, text="Resolution", width=18).pack(side="left")
        ttk.Entry(resolution, textvariable=self.dataset_vars["width"], width=8).pack(side="left")
        ttk.Label(resolution, text=" × ").pack(side="left")
        ttk.Entry(resolution, textvariable=self.dataset_vars["height"], width=8).pack(side="left")
        preset = ttk.Combobox(
            resolution,
            values=("256 × 256", "512 × 512", "768 × 768", "1024 × 1024", "960 × 544", "544 × 960"),
            state="readonly",
            width=14,
        )
        preset.pack(side="left", padx=(10, 0))
        preset.bind("<<ComboboxSelected>>", lambda _event: self._apply_resolution_preset(preset.get()))
        self._bind_popup_wheel(preset)
        BuilderToolTip(preset, "Sets width and height together. You can still enter any custom dimensions.")
        row += 1

        row = self._simple_row(
            content,
            row,
            "Repeats",
            self.dataset_vars["num_repeats"],
            "Repeats every item this many times per epoch, increasing this dataset's influence.",
        )
        row = self._simple_row(
            content,
            row,
            "Batch size override",
            self.dataset_vars["batch_size"],
            "Overrides the general batch size only for this dataset. Leave empty to inherit.",
        )
        row = self._simple_row(
            content,
            row,
            "Caption extension override",
            self.dataset_vars["caption_extension"],
            "Overrides the general caption extension. Leave empty to inherit.",
        )

        row = self._combo_row(
            content,
            row,
            "Buckets override",
            self.dataset_vars["enable_bucket_override"],
            ("Use general", "Enabled", "Disabled"),
            "Overrides aspect-ratio buckets only for this dataset.",
        )
        row = self._combo_row(
            content,
            row,
            "No-upscale override",
            self.dataset_vars["bucket_no_upscale_override"],
            ("Use general", "Enabled", "Disabled"),
            "Overrides the no-upscale behavior only for this dataset.",
        )

        self.video_frame = ttk.LabelFrame(content, text="Video options", padding=7)
        self.video_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        self.video_frame.columnconfigure(1, weight=1)
        video_row = 0
        video_row = self._simple_row(
            self.video_frame,
            video_row,
            "Target frames",
            self.dataset_vars["target_frames"],
            "Comma-separated clip lengths, for example 17, 33, 65.",
        )
        video_row = self._combo_row(
            self.video_frame,
            video_row,
            "Frame extraction",
            self.dataset_vars["frame_extraction"],
            ("head", "chunk", "slide", "uniform"),
            "head uses the start; chunk splits clips; slide uses a stride; uniform samples across the video.",
        )
        video_row = self._simple_row(
            self.video_frame,
            video_row,
            "Frame stride",
            self.dataset_vars["frame_stride"],
            "Distance between sliding windows. Used by slide extraction.",
        )
        video_row = self._simple_row(
            self.video_frame,
            video_row,
            "Uniform samples",
            self.dataset_vars["frame_sample"],
            "Number of uniformly distributed clips. Used by uniform extraction.",
        )
        video_row = self._simple_row(
            self.video_frame,
            video_row,
            "Maximum frames",
            self.dataset_vars["max_frames"],
            "Limits how many source frames are considered from each video.",
        )
        self._simple_row(
            self.video_frame,
            video_row,
            "Source FPS",
            self.dataset_vars["source_fps"],
            "Overrides the detected source frame rate when sampling video.",
        )
        row += 1

        self.image_frame = ttk.LabelFrame(content, text="Image options", padding=7)
        self.image_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        multiple = ttk.Checkbutton(
            self.image_frame,
            text="Multiple target images per source",
            variable=self.dataset_vars["multiple_target"],
        )
        multiple.pack(anchor="w")
        BuilderToolTip(multiple, "Treats matching numbered files as multiple target images for one source item.")
        no_resize = ttk.Checkbutton(
            self.image_frame,
            text="Do not resize control images",
            variable=self.dataset_vars["no_resize_control"],
        )
        no_resize.pack(anchor="w", pady=(5, 0))
        BuilderToolTip(no_resize, "Keeps control images at their original size instead of matching the target resolution.")
        row += 1

        advanced = ttk.LabelFrame(content, text="Optional control input", padding=7)
        advanced.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        advanced.columnconfigure(1, weight=1)
        control_row = self._path_row(
            advanced,
            0,
            "Control directory",
            self.dataset_vars["control_directory"],
            "Optional matching control images. Leave empty for ordinary LoRA datasets.",
        )
        control_resolution = ttk.Frame(advanced)
        control_resolution.grid(row=control_row, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Label(control_resolution, text="Control resolution", width=18).pack(side="left")
        ttk.Entry(control_resolution, textvariable=self.dataset_vars["control_width"], width=8).pack(side="left")
        ttk.Label(control_resolution, text=" × ").pack(side="left")
        ttk.Entry(control_resolution, textvariable=self.dataset_vars["control_height"], width=8).pack(side="left")
        BuilderToolTip(control_resolution, "Optional image-only control resolution. Leave both fields empty to inherit.")

        inspect_row = ttk.Frame(content)
        inspect_row.grid(row=row + 1, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ttk.Button(inspect_row, text="Inspect files", command=self._inspect_selected_source).pack(side="left")
        self.inspect_label = ttk.Label(inspect_row, text="", style="Muted.TLabel")
        self.inspect_label.pack(side="left", padx=(10, 0), fill="x", expand=True)

    def _build_raw_tab(self):
        help_text = (
            "Edit any supported TOML option directly. Switching back to Builder parses these changes. "
            "Visual edits preserve options that are not shown in the builder."
        )
        ttk.Label(self.raw_tab, text=help_text, style="PageHelp.TLabel", wraplength=900).pack(
            anchor="w", pady=(0, 7)
        )
        text_frame = ttk.Frame(self.raw_tab)
        text_frame.pack(fill="both", expand=True)
        self.raw_text = tk.Text(
            text_frame,
            wrap=tk.NONE,
            undo=True,
            font=("Cascadia Mono", 10),
            background=self.colors.get("field", "#0F172A"),
            foreground=self.colors.get("text", "#F8FAFC"),
            insertbackground=self.colors.get("text", "#F8FAFC"),
            selectbackground=self.colors.get("selection", "#075985"),
        )
        y_scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.raw_text.yview)
        x_scroll = ttk.Scrollbar(text_frame, orient="horizontal", command=self.raw_text.xview)
        self.raw_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self._bind_popup_wheel(self.raw_text)
        self.raw_text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)

    def _install_dirty_tracking(self):
        for variable in (*self.general_vars.values(), *self.dataset_vars.values()):
            variable.trace_add("write", self._mark_dirty)
        self.raw_text.bind("<<Modified>>", self._raw_text_modified, add="+")
        self.raw_text.edit_modified(False)

    def _bind_popup_wheel(self, widget):
        widget.bind("<MouseWheel>", self._route_popup_mousewheel, add="+")
        widget.bind("<Button-4>", self._route_popup_mousewheel, add="+")
        widget.bind("<Button-5>", self._route_popup_mousewheel, add="+")

    @staticmethod
    def _wheel_direction(event):
        if getattr(event, "num", None) == 4:
            return -1
        if getattr(event, "num", None) == 5:
            return 1
        return -int(event.delta / 120) if event.delta else 0

    def _route_popup_mousewheel(self, event):
        direction = self._wheel_direction(event)
        if not direction:
            return "break"
        try:
            selected_tab = self.notebook.index(self.notebook.select())
            if selected_tab == 1:
                self.raw_text.yview_scroll(direction * 3, "units")
            elif event.widget is self.dataset_tree:
                self.dataset_tree.yview_scroll(direction * 3, "units")
            else:
                self.editor_canvas.yview_scroll(direction * 3, "units")
        except tk.TclError:
            pass
        return "break"

    def _raw_text_modified(self, _event=None):
        if self.raw_text.edit_modified():
            self.raw_text.edit_modified(False)
            self._mark_dirty()

    def _mark_dirty(self, *_args):
        if self._suspend_dirty:
            return
        if not self.is_dirty:
            self.is_dirty = True
            self.window.title("Dataset Config Builder *")
            self.status_var.set("Unsaved changes are staged in the builder.")

    def _mark_clean(self, status=None):
        self.is_dirty = False
        self.window.title("Dataset Config Builder")
        if status:
            self.status_var.set(status)

    def _replace_raw_text(self, text):
        previous = self._suspend_dirty
        self._suspend_dirty = True
        try:
            self.raw_text.delete("1.0", tk.END)
            self.raw_text.insert("1.0", text)
            self.raw_text.edit_modified(False)
        finally:
            self._suspend_dirty = previous

    def _labeled_entry(self, parent, label, variable, row, column, tooltip):
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=column, sticky="w", padx=(0, 6), pady=3)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=column + 1, sticky="ew", padx=(0, 14), pady=3)
        BuilderToolTip(label_widget, tooltip)
        BuilderToolTip(entry, tooltip)

    def _simple_row(self, parent, row, label, variable, tooltip):
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        BuilderToolTip(label_widget, tooltip)
        BuilderToolTip(entry, tooltip)
        return row + 1

    def _combo_row(self, parent, row, label, variable, values, tooltip):
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        self._bind_popup_wheel(combo)
        BuilderToolTip(label_widget, tooltip)
        BuilderToolTip(combo, tooltip)
        return row + 1

    def _path_row(self, parent, row, label, variable, tooltip):
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        path_frame = ttk.Frame(parent)
        path_frame.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        entry = ttk.Entry(path_frame, textvariable=variable)
        entry.pack(side="left", fill="x", expand=True)
        button = ttk.Button(
            path_frame,
            text="Browse",
            command=lambda: self._browse_directory(variable),
        )
        button.pack(side="right", padx=(5, 0))
        BuilderToolTip(label_widget, tooltip)
        BuilderToolTip(entry, tooltip)
        BuilderToolTip(button, tooltip)
        return row + 1

    def _browse_directory(self, variable):
        selected = filedialog.askdirectory(parent=self.window, initialdir=variable.get() or None)
        if selected:
            variable.set(selected)

    def _browse_source(self):
        current = self.dataset_vars["source"].get()
        if self.dataset_vars["source_format"].get() == "JSONL file":
            selected = filedialog.askopenfilename(
                parent=self.window,
                initialdir=str(Path(current).parent) if current else None,
                filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")],
            )
        else:
            selected = filedialog.askdirectory(parent=self.window, initialdir=current or None)
        if selected:
            self.dataset_vars["source"].set(selected)
            if not self.dataset_vars["cache_directory"].get().strip():
                source_path = Path(selected)
                source_base = source_path.stem if source_path.is_file() else source_path.name
                self.dataset_vars["cache_directory"].set(str(source_path.parent / f"{source_base}_cache"))
            self._inspect_selected_source()

    def _update_source_button(self):
        return

    def _apply_resolution_preset(self, preset):
        dimensions = [part.strip() for part in preset.replace("×", "x").split("x")]
        if len(dimensions) == 2:
            self.dataset_vars["width"].set(dimensions[0])
            self.dataset_vars["height"].set(dimensions[1])

    def _load_path(self, path):
        try:
            text = Path(path).read_text(encoding="utf-8")
            self._load_text(text)
            self.path = Path(path)
            self.path_var.set(str(self.path))
            self._mark_clean(f"Loaded {self.path.name}")
        except (OSError, ValueError, toml.TomlDecodeError) as exc:
            messagebox.showerror("Cannot open TOML", str(exc), parent=self.window)

    def _load_text(self, text):
        previous = self._suspend_dirty
        self._suspend_dirty = True
        try:
            document = _parse_toml(text)
            plain = _plain_document(document)
            if not isinstance(plain.get("general", {}), dict):
                raise ValueError("[general] must be a TOML table.")
            if not isinstance(plain.get("datasets", []), list):
                raise ValueError("[[datasets]] must be an array of tables.")
            self.document = document
            self._populate_general(plain.get("general", {}))
            raw_tables = list(document.get("datasets", []))
            self.datasets = [
                {"kind": _dataset_kind(raw_table), "raw": raw_table}
                for raw_table in raw_tables
            ]
            if not self.datasets:
                self.datasets = [{"kind": "image", "raw": _new_table()}]
            self.selected_index = None
            self._refresh_dataset_tree(select=0)
            self._replace_raw_text(_dump_toml(self.document))
        finally:
            self._suspend_dirty = previous

    def _populate_general(self, general):
        self.general_vars["caption_extension"].set(str(general.get("caption_extension", ".txt")))
        self.general_vars["batch_size"].set(str(general.get("batch_size", 1)))
        self.general_vars["enable_bucket"].set(bool(general.get("enable_bucket", True)))
        self.general_vars["bucket_no_upscale"].set(bool(general.get("bucket_no_upscale", False)))

    def _refresh_dataset_tree(self, select=None):
        self.dataset_tree.delete(*self.dataset_tree.get_children())
        for index, dataset in enumerate(self.datasets):
            self.dataset_tree.insert(
                "",
                "end",
                iid=str(index),
                values=self._dataset_summary(dataset),
            )
        if self.datasets:
            index = min(select if select is not None else 0, len(self.datasets) - 1)
            self.dataset_tree.selection_set(str(index))
            self.dataset_tree.focus(str(index))
            self._show_dataset(index)

    @staticmethod
    def _dataset_summary(dataset):
        table = dataset["raw"]
        resolution = table.get("resolution", [512, 512])
        if isinstance(resolution, (int, float)):
            resolution = [resolution, resolution]
        resolution_text = " × ".join(str(value) for value in list(resolution)[:2])
        _source_format, source = _source_details(table, dataset["kind"])
        source_text = Path(source).name if source else "Not selected"
        return dataset["kind"].title(), resolution_text, source_text

    def _select_dataset(self, _event=None):
        selection = self.dataset_tree.selection()
        if not selection:
            return
        new_index = int(selection[0])
        if self.selected_index is not None and self.selected_index != new_index:
            try:
                self._capture_selected_dataset()
            except ValueError as exc:
                messagebox.showerror("Invalid dataset value", str(exc), parent=self.window)
                self.dataset_tree.selection_set(str(self.selected_index))
                return
        self._show_dataset(new_index)

    def _show_dataset(self, index):
        if not 0 <= index < len(self.datasets):
            return
        previous = self._suspend_dirty
        self._suspend_dirty = True
        try:
            self.selected_index = index
            dataset = self.datasets[index]
            table = dataset["raw"]
            kind = dataset["kind"]
            self.dataset_type_label.configure(text=f"{kind.title()} dataset")
            source_format, source = _source_details(table, kind)
            self.dataset_vars["source_format"].set(source_format)
            self.dataset_vars["source"].set(source)
            self.dataset_vars["cache_directory"].set(str(table.get("cache_directory", "")))
            resolution = table.get("resolution", [512, 512])
            if isinstance(resolution, (int, float)):
                resolution = [resolution, resolution]
            resolution = list(resolution) + [512, 512]
            self.dataset_vars["width"].set(str(resolution[0]))
            self.dataset_vars["height"].set(str(resolution[1]))
            self.dataset_vars["num_repeats"].set(str(table.get("num_repeats", 1)))
            self.dataset_vars["batch_size"].set(str(table.get("batch_size", "")))
            self.dataset_vars["caption_extension"].set(str(table.get("caption_extension", "")))
            self.dataset_vars["enable_bucket_override"].set(self._override_text(table, "enable_bucket"))
            self.dataset_vars["bucket_no_upscale_override"].set(self._override_text(table, "bucket_no_upscale"))
            self.dataset_vars["control_directory"].set(str(table.get("control_directory", "")))
            self.dataset_vars["multiple_target"].set(bool(table.get("multiple_target", False)))
            self.dataset_vars["no_resize_control"].set(bool(table.get("no_resize_control", False)))
            control_resolution = list(table.get("control_resolution", []))
            self.dataset_vars["control_width"].set(str(control_resolution[0]) if len(control_resolution) > 0 else "")
            self.dataset_vars["control_height"].set(str(control_resolution[1]) if len(control_resolution) > 1 else "")
            self.dataset_vars["target_frames"].set(", ".join(str(value) for value in table.get("target_frames", [1])))
            self.dataset_vars["frame_extraction"].set(str(table.get("frame_extraction", "head")))
            self.dataset_vars["frame_stride"].set(str(table.get("frame_stride", "")))
            self.dataset_vars["frame_sample"].set(str(table.get("frame_sample", "")))
            self.dataset_vars["max_frames"].set(str(table.get("max_frames", "")))
            self.dataset_vars["source_fps"].set(str(table.get("source_fps", "")))
            if kind == "video":
                self.video_frame.grid()
                self.image_frame.grid_remove()
            else:
                self.image_frame.grid()
                self.video_frame.grid_remove()
            self.inspect_label.configure(text="")
            self._update_source_button()
        finally:
            self._suspend_dirty = previous

    @staticmethod
    def _override_text(table, key):
        if key not in table:
            return "Use general"
        return "Enabled" if bool(table[key]) else "Disabled"

    @staticmethod
    def _override_value(value):
        if value == "Use general":
            return None
        return value == "Enabled"

    def _capture_selected_dataset(self):
        if self.selected_index is None or not 0 <= self.selected_index < len(self.datasets):
            return
        dataset = self.datasets[self.selected_index]
        table = dataset["raw"]
        kind = dataset["kind"]

        width = _positive_int(self.dataset_vars["width"].get(), "Resolution width")
        height = _positive_int(self.dataset_vars["height"].get(), "Resolution height")
        repeats = _positive_int(self.dataset_vars["num_repeats"].get(), "Repeats")
        source = self.dataset_vars["source"].get().strip()
        source_format = self.dataset_vars["source_format"].get()
        directory_key = f"{kind}_directory"
        jsonl_key = f"{kind}_jsonl_file"
        if source_format == "JSONL file":
            table.pop(directory_key, None)
            _set_optional(table, jsonl_key, source)
        else:
            table.pop(jsonl_key, None)
            _set_optional(table, directory_key, source)

        table["resolution"] = [width, height]
        table["num_repeats"] = repeats
        _set_optional(table, "cache_directory", self.dataset_vars["cache_directory"].get().strip())
        _set_optional(
            table,
            "batch_size",
            _positive_int(self.dataset_vars["batch_size"].get(), "Dataset batch size", allow_blank=True),
        )
        _set_optional(table, "caption_extension", self.dataset_vars["caption_extension"].get().strip())
        _set_optional(
            table,
            "enable_bucket",
            self._override_value(self.dataset_vars["enable_bucket_override"].get()),
        )
        _set_optional(
            table,
            "bucket_no_upscale",
            self._override_value(self.dataset_vars["bucket_no_upscale_override"].get()),
        )
        _set_optional(table, "control_directory", self.dataset_vars["control_directory"].get().strip())

        if kind == "video":
            table["target_frames"] = _integer_list(self.dataset_vars["target_frames"].get(), "Target frames")
            table["frame_extraction"] = self.dataset_vars["frame_extraction"].get()
            _set_optional(
                table,
                "frame_stride",
                _positive_int(self.dataset_vars["frame_stride"].get(), "Frame stride", allow_blank=True),
            )
            _set_optional(
                table,
                "frame_sample",
                _positive_int(self.dataset_vars["frame_sample"].get(), "Uniform samples", allow_blank=True),
            )
            _set_optional(
                table,
                "max_frames",
                _positive_int(self.dataset_vars["max_frames"].get(), "Maximum frames", allow_blank=True),
            )
            _set_optional(
                table,
                "source_fps",
                _positive_float(self.dataset_vars["source_fps"].get(), "Source FPS", allow_blank=True),
            )
        else:
            table["multiple_target"] = bool(self.dataset_vars["multiple_target"].get())
            table["no_resize_control"] = bool(self.dataset_vars["no_resize_control"].get())
            control_width = _positive_int(
                self.dataset_vars["control_width"].get(),
                "Control resolution width",
                allow_blank=True,
            )
            control_height = _positive_int(
                self.dataset_vars["control_height"].get(),
                "Control resolution height",
                allow_blank=True,
            )
            if (control_width is None) != (control_height is None):
                raise ValueError("Enter both control-resolution dimensions or leave both empty.")
            _set_optional(
                table,
                "control_resolution",
                [control_width, control_height] if control_width is not None else None,
            )
        item_id = str(self.selected_index)
        if self.dataset_tree.exists(item_id):
            self.dataset_tree.item(item_id, values=self._dataset_summary(dataset))

    def _add_dataset(self, kind):
        self._capture_selected_dataset()
        table = _new_table()
        table["resolution"] = [512, 512]
        table[f"{kind}_directory"] = ""
        table["num_repeats"] = 1
        if kind == "video":
            table["target_frames"] = [1]
            table["frame_extraction"] = "head"
        self.datasets.append({"kind": kind, "raw": table})
        self._refresh_dataset_tree(select=len(self.datasets) - 1)
        self._mark_dirty()

    def _duplicate_dataset(self):
        if self.selected_index is None:
            return
        self._capture_selected_dataset()
        source = self.datasets[self.selected_index]
        duplicate = {"kind": source["kind"], "raw": copy.deepcopy(source["raw"])}
        duplicate["raw"].pop("cache_directory", None)
        self.datasets.insert(self.selected_index + 1, duplicate)
        self._refresh_dataset_tree(select=self.selected_index + 1)
        self._mark_dirty()

    def _remove_dataset(self):
        if self.selected_index is None:
            return
        if len(self.datasets) == 1:
            messagebox.showwarning("Dataset required", "A configuration must contain at least one dataset.", parent=self.window)
            return
        index = self.selected_index
        del self.datasets[index]
        self.selected_index = None
        self._refresh_dataset_tree(select=min(index, len(self.datasets) - 1))
        self._mark_dirty()

    def _move_dataset(self, direction):
        if self.selected_index is None:
            return
        self._capture_selected_dataset()
        destination = self.selected_index + direction
        if not 0 <= destination < len(self.datasets):
            return
        self.datasets[self.selected_index], self.datasets[destination] = (
            self.datasets[destination],
            self.datasets[self.selected_index],
        )
        self.selected_index = None
        self._refresh_dataset_tree(select=destination)
        self._mark_dirty()

    def _apply_builder_to_document(self):
        self._capture_selected_dataset()
        general = self.document.get("general")
        if general is None:
            general = _new_table()
            self.document["general"] = general
        general["caption_extension"] = self.general_vars["caption_extension"].get().strip() or ".txt"
        general["batch_size"] = _positive_int(self.general_vars["batch_size"].get(), "General batch size")
        general["enable_bucket"] = bool(self.general_vars["enable_bucket"].get())
        general["bucket_no_upscale"] = bool(self.general_vars["bucket_no_upscale"].get())

        if TOMLKIT_AVAILABLE:
            dataset_array = tomlkit.aot()
            for dataset in self.datasets:
                _trim_tomlkit_table_separator(dataset["raw"])
                dataset_array.append(dataset["raw"])
            self.document["datasets"] = dataset_array
        else:
            self.document["datasets"] = [dataset["raw"] for dataset in self.datasets]
        return _dump_toml(self.document)

    def _validate_document(self, document):
        plain = _plain_document(document)
        datasets = plain.get("datasets", [])
        if not datasets:
            raise ValueError("Add at least one image or video dataset.")

        effective_caches = []
        general = plain.get("general", {})
        for index, dataset in enumerate(datasets, start=1):
            kind = _dataset_kind(dataset)
            directory = str(dataset.get(f"{kind}_directory", "")).strip()
            jsonl = str(dataset.get(f"{kind}_jsonl_file", "")).strip()
            if not directory and not jsonl:
                raise ValueError(f"Dataset {index} needs a {kind} directory or JSONL file.")
            effective_caches.append(
                os.path.normcase(os.path.abspath(str(dataset.get("cache_directory") or directory or jsonl)))
            )
            if dataset.get("frame_extraction") == "chunk" and 1 in dataset.get("target_frames", []):
                raise ValueError(f"Dataset {index}: chunk extraction cannot include 1 in Target frames.")

        if len(set(effective_caches)) != len(effective_caches):
            raise ValueError("Each dataset must use a different cache directory.")

        try:
            from musubi_tuner.dataset.config_utils import ConfigSanitizer

            ConfigSanitizer().sanitize_user_config({"general": general, "datasets": datasets})
        except ImportError:
            pass
        except Exception as exc:
            raise ValueError(f"The training dataset schema rejected this file: {exc}") from exc
        return plain

    def _current_document(self):
        if self.notebook.index(self.notebook.select()) == 1:
            return _parse_toml(self.raw_text.get("1.0", "end-1c"))
        text = self._apply_builder_to_document()
        self._replace_raw_text(text)
        return self.document

    def _validate_action(self):
        try:
            document = self._current_document()
            plain = self._validate_document(document)
            count = len(plain.get("datasets", []))
            self.status_var.set(f"Valid configuration: {count} dataset{'s' if count != 1 else ''}.")
            messagebox.showinfo(
                "Dataset configuration is valid",
                f"The TOML contains {count} valid dataset{'s' if count != 1 else ''}.",
                parent=self.window,
            )
        except Exception as exc:
            self.status_var.set(f"Validation failed: {exc}")
            messagebox.showerror("Invalid dataset configuration", str(exc), parent=self.window)

    def _on_tab_changed(self, _event=None):
        if self._changing_tab:
            return
        selected = self.notebook.index(self.notebook.select())
        if selected == self._active_tab:
            return
        try:
            if selected == 1:
                text = self._apply_builder_to_document()
                self._replace_raw_text(text)
            else:
                self._load_text(self.raw_text.get("1.0", "end-1c"))
            self._active_tab = selected
        except Exception as exc:
            self.status_var.set(f"Cannot switch views: {exc}")
            messagebox.showerror("Invalid TOML", str(exc), parent=self.window)
            self._changing_tab = True
            self.notebook.select(self._active_tab)
            self._changing_tab = False

    def _inspect_selected_source(self):
        source = Path(self.dataset_vars["source"].get().strip()).expanduser()
        if not source.exists():
            self.inspect_label.configure(text="Source does not exist yet.")
            return
        if source.is_file():
            self.inspect_label.configure(text=f"Manifest: {source.name}")
            return
        kind = self.datasets[self.selected_index]["kind"] if self.selected_index is not None else "image"
        extensions = VIDEO_EXTENSIONS if kind == "video" else IMAGE_EXTENSIONS
        try:
            media = [path for path in source.iterdir() if path.is_file() and path.suffix.lower() in extensions]
            caption_extension = (
                self.dataset_vars["caption_extension"].get().strip()
                or self.general_vars["caption_extension"].get().strip()
                or ".txt"
            )
            captions = sum(1 for path in media if path.with_suffix(caption_extension).is_file())
            repeats = _positive_int(self.dataset_vars["num_repeats"].get(), "Repeats")
            self.inspect_label.configure(
                text=f"{len(media)} media · {captions} captions · {len(media) * repeats} effective samples"
            )
        except OSError as exc:
            self.inspect_label.configure(text=f"Cannot inspect source: {exc}")

    def _open_file(self):
        if self.is_dirty and not messagebox.askyesno(
            "Discard staged changes?",
            "Opening another file will discard the unsaved changes currently staged in the builder.",
            parent=self.window,
        ):
            return
        selected = filedialog.askopenfilename(
            parent=self.window,
            initialdir=str(self.path.parent) if self.path else None,
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if selected:
            self._load_path(Path(selected))

    def _choose_save_path(self):
        current = Path(self.path_var.get()).expanduser() if self.path_var.get().strip() else self.path
        selected = filedialog.asksaveasfilename(
            parent=self.window,
            initialdir=str(current.parent) if current else None,
            initialfile=current.name if current else "dataset_config.toml",
            defaultextension=".toml",
            filetypes=[("TOML files", "*.toml")],
        )
        return Path(selected) if selected else None

    def _save(self, force_choose=False):
        try:
            document = self._current_document()
            self._validate_document(document)
        except Exception as exc:
            messagebox.showerror("Cannot save invalid TOML", str(exc), parent=self.window)
            return None

        destination = None if force_choose else (
            Path(self.path_var.get()).expanduser() if self.path_var.get().strip() else self.path
        )
        if destination is None:
            destination = self._choose_save_path()
        if destination is None:
            return None
        if destination.suffix.lower() != ".toml":
            destination = destination.with_suffix(".toml")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(_dump_toml(document), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Could not save TOML", str(exc), parent=self.window)
            return None
        self.path = destination
        self.path_var.set(str(destination))
        self.document = document
        self._mark_clean(f"Saved {destination.name}")
        return destination

    def _save_as(self):
        self._save(force_choose=True)

    def _save_and_use(self):
        destination = self._save()
        if destination is None:
            return
        if callable(self.on_use):
            self.on_use(str(destination))
        self._close()

    def _open_external_editor(self):
        destination = self._save()
        if destination is None:
            return
        try:
            if os.name == "nt":
                os.startfile(str(destination))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(destination)])
            else:
                subprocess.Popen(["xdg-open", str(destination)])
        except OSError as exc:
            messagebox.showerror("Could not open editor", str(exc), parent=self.window)

    def _close(self):
        if self.is_dirty and not messagebox.askyesno(
            "Discard staged changes?",
            "Close the builder and discard all unsaved changes?",
            parent=self.window,
        ):
            return
        try:
            self.window.grab_release()
        except tk.TclError:
            pass
        self.window.destroy()
