import copy
import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


LIBRARY_VERSION = 1


def default_library_root():
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "MusubiTuner" / "prompt_library"


def prompt_identity(prompt):
    identity = {}
    for key, value in prompt.items():
        if key == "enabled" or str(key).startswith("_library_"):
            continue
        if isinstance(value, str):
            value = value.strip()
        identity[key] = value
    return json.dumps(identity, sort_keys=True, ensure_ascii=False, default=str)


def prompt_display_name(prompt):
    text = " ".join(str(prompt.get("prompt") or "Untitled prompt").split())
    return text[:64] + ("…" if len(text) > 64 else "")


class PromptLibraryStore:
    def __init__(self, root=None):
        self.root = Path(root) if root else default_library_root()
        self.path = self.root / "library.json"
        self.thumbnail_root = self.root / "thumbnails"
        self.data = {"version": LIBRARY_VERSION, "prompts": []}
        self.load()

    @property
    def prompts(self):
        return self.data["prompts"]

    def load(self):
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("prompts"), list):
                self.data = loaded
        except (OSError, ValueError, TypeError):
            pass
        self.data["version"] = LIBRARY_VERSION
        self.data.setdefault("prompts", [])
        for entry in self.prompts:
            entry.setdefault("id", uuid.uuid4().hex)
            entry.setdefault("name", prompt_display_name(entry.get("prompt_data", {})))
            entry.setdefault("prompt_data", {})
            entry.setdefault("tags", [])
            entry.setdefault("collection", "")
            entry.setdefault("favorite", False)
            entry.setdefault("revision", 1)
            entry.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
            entry.setdefault("updated_at", entry["created_at"])
            entry.setdefault("sources", [])
            entry.setdefault("thumbnails", [])

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.data, indent=2, ensure_ascii=False)
        fd, temporary = tempfile.mkstemp(prefix="prompt-library-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(temporary, self.path)
        finally:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass

    def find(self, prompt=None, library_id=""):
        if library_id:
            match = next((entry for entry in self.prompts if entry.get("id") == library_id), None)
            if match and (prompt is None or prompt_identity(match.get("prompt_data", {})) == prompt_identity(prompt)):
                return match
        if prompt is None:
            return None
        identity = prompt_identity(prompt)
        return next((entry for entry in self.prompts if prompt_identity(entry.get("prompt_data", {})) == identity), None)

    def add_or_merge(self, prompt, source=None, name=""):
        clean_prompt = copy.deepcopy(prompt)
        library_id = str(clean_prompt.pop("_library_id", "") or "")
        clean_prompt.pop("_library_revision", None)
        entry = self.find(clean_prompt, library_id=library_id)
        created = False
        now = datetime.now().isoformat(timespec="seconds")
        if entry is None:
            entry = {
                "id": uuid.uuid4().hex,
                "name": name or prompt_display_name(clean_prompt),
                "prompt_data": clean_prompt,
                "tags": [],
                "collection": "",
                "favorite": False,
                "revision": 1,
                "created_at": now,
                "updated_at": now,
                "sources": [],
                "thumbnails": [],
            }
            self.prompts.append(entry)
            created = True
        if source and source not in entry["sources"]:
            entry["sources"].append(source)
            entry["updated_at"] = now
        return entry, created

    def import_prompts(self, prompts, source=None):
        added = 0
        merged = 0
        for prompt in prompts:
            if not isinstance(prompt, dict) or not str(prompt.get("prompt") or "").strip():
                continue
            _entry, created = self.add_or_merge(prompt, source=source)
            added += int(created)
            merged += int(not created)
        if added or merged:
            self.save()
        return added, merged

    def import_jobs(self, jobs):
        added = merged = jobs_with_prompts = 0
        for job in jobs:
            snapshot = job.get("settings_snapshot")
            prompts = snapshot.get("sample_prompts_data") if isinstance(snapshot, dict) else None
            if not isinstance(prompts, list) or not prompts:
                continue
            jobs_with_prompts += 1
            source = {
                "type": "job",
                "job_id": job.get("job_id", ""),
                "title": job.get("output_name") or job.get("title", ""),
                "started_at": job.get("started_at", ""),
            }
            current_added, current_merged = self.import_prompts(prompts, source=source)
            added += current_added
            merged += current_merged
        return added, merged, jobs_with_prompts

    def update_entry(self, entry_id, *, name, prompt_data, tags, collection):
        entry = self.find(library_id=entry_id)
        if not entry:
            raise KeyError(entry_id)
        entry["name"] = name.strip() or prompt_display_name(prompt_data)
        entry["prompt_data"] = copy.deepcopy(prompt_data)
        entry["tags"] = sorted({tag.strip() for tag in tags if tag.strip()}, key=str.casefold)
        entry["collection"] = collection.strip()
        entry["revision"] = int(entry.get("revision", 1)) + 1
        entry["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.save()

    def toggle_favorite(self, entry_id):
        entry = self.find(library_id=entry_id)
        if entry:
            entry["favorite"] = not bool(entry.get("favorite"))
            entry["updated_at"] = datetime.now().isoformat(timespec="seconds")
            self.save()

    def delete(self, entry_id):
        entry = self.find(library_id=entry_id)
        if not entry:
            return False
        self.prompts.remove(entry)
        thumb_dir = self.thumbnail_root / entry_id
        if thumb_dir.is_dir():
            shutil.rmtree(thumb_dir, ignore_errors=True)
        self.save()
        return True

    def prompt_copy(self, entry):
        prompt = copy.deepcopy(entry.get("prompt_data", {}))
        prompt["_library_id"] = entry["id"]
        prompt["_library_revision"] = entry.get("revision", 1)
        prompt.setdefault("enabled", True)
        return prompt

    def thumbnail_path(self, thumbnail):
        value = Path(str(thumbnail.get("file") or ""))
        return value if value.is_absolute() else self.root / value

    def latest_thumbnail(self, entry):
        valid = [thumb for thumb in entry.get("thumbnails", []) if self.thumbnail_path(thumb).is_file()]
        return max(valid, key=lambda thumb: str(thumb.get("created_at") or "")) if valid else None

    def capture_thumbnail(self, prompt, image_path, mode, metadata=None):
        image_path = Path(image_path)
        if not image_path.is_file():
            return None, False
        entry, created = self.add_or_merge(
            prompt,
            source={"type": "standalone_test", "mode": mode},
        )
        destination_dir = self.thumbnail_root / entry["id"]
        destination_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        destination = destination_dir / f"{stamp}{image_path.suffix.lower()}"
        shutil.copy2(image_path, destination)
        relative = destination.relative_to(self.root).as_posix()
        thumbnail = {
            "file": relative,
            "mode": mode,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_file": str(image_path),
        }
        if metadata:
            thumbnail["metadata"] = copy.deepcopy(metadata)
        entry.setdefault("thumbnails", []).append(thumbnail)
        if len(entry["thumbnails"]) > 12:
            removed = sorted(entry["thumbnails"], key=lambda item: str(item.get("created_at") or ""))[:-12]
            entry["thumbnails"] = [item for item in entry["thumbnails"] if item not in removed]
            for old_thumbnail in removed:
                try:
                    self.thumbnail_path(old_thumbnail).unlink(missing_ok=True)
                except OSError:
                    pass
        entry["updated_at"] = thumbnail["created_at"]
        self.save()
        return entry, created

    def export_to(self, destination):
        destination = Path(destination)
        destination.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    def import_file(self, source):
        imported = json.loads(Path(source).read_text(encoding="utf-8"))
        prompts = imported.get("prompts") if isinstance(imported, dict) else None
        if not isinstance(prompts, list):
            raise ValueError("The selected file is not a prompt-library export.")
        added = merged = 0
        for item in prompts:
            if not isinstance(item, dict) or not isinstance(item.get("prompt_data"), dict):
                continue
            entry, created = self.add_or_merge(item["prompt_data"], source={"type": "library_import"}, name=item.get("name", ""))
            if created:
                entry["tags"] = list(item.get("tags") or [])
                entry["collection"] = str(item.get("collection") or "")
                entry["favorite"] = bool(item.get("favorite"))
            added += int(created)
            merged += int(not created)
        self.save()
        return added, merged


class PromptLibraryDialog:
    def __init__(self, parent, store, colors, current_prompts, jobs, on_use):
        self.parent = parent
        self.store = store
        self.colors = colors
        self.current_prompts = current_prompts
        self.jobs = jobs
        self.on_use = on_use
        self.images = {}
        self.search_var = tk.StringVar()
        self.collection_var = tk.StringVar(value="All collections")
        self.mode_var = tk.StringVar(value="All models")
        self.status_var = tk.StringVar()

        self.window = tk.Toplevel(parent)
        self.window.title("Global Prompt Library")
        self.window.geometry("1120x780")
        self.window.minsize(860, 600)
        self.window.transient(parent)

        self._build_ui()
        self.refresh()

    def _build_ui(self):
        host = ttk.Frame(self.window, padding=12)
        host.pack(fill="both", expand=True)
        header = ttk.Frame(host)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="Global Prompt Library", style="PageTitle.TLabel").pack(side="left")
        ttk.Button(header, text="Export", command=self._export).pack(side="right", padx=(6, 0))
        ttk.Button(header, text="Import File", command=self._import_file).pack(side="right", padx=(6, 0))
        ttk.Button(header, text="Collect from Jobs", command=self._collect_jobs).pack(side="right", padx=(6, 0))
        ttk.Button(header, text="Add Current Prompts", command=self._add_current).pack(side="right")
        ttk.Label(
            host,
            text="Library prompts are copied into the current run. Editing the library never changes an existing run or job snapshot.",
            style="PageHelp.TLabel",
            wraplength=900,
        ).pack(anchor="w", pady=(0, 8))

        filters = ttk.Frame(host)
        filters.pack(fill="x", pady=(0, 8))
        ttk.Label(filters, text="Search").pack(side="left")
        search = ttk.Entry(filters, textvariable=self.search_var)
        search.pack(side="left", fill="x", expand=True, padx=(6, 12))
        self.collection_combo = ttk.Combobox(filters, textvariable=self.collection_var, state="readonly", width=22)
        self.collection_combo.pack(side="left", padx=(0, 8))
        self.mode_combo = ttk.Combobox(filters, textvariable=self.mode_var, state="readonly", width=16)
        self.mode_combo.pack(side="left")
        ttk.Button(filters, text="Use Visible", command=self._use_visible).pack(side="left", padx=(8, 0))
        self.search_var.trace_add("write", lambda *_args: self.refresh_cards())
        self.collection_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_cards())
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_cards())

        gallery_host = ttk.Frame(host)
        gallery_host.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(gallery_host, background=self.colors.get("page", "#111827"), highlightthickness=0)
        scrollbar = ttk.Scrollbar(gallery_host, orient="vertical", command=self.canvas.yview)
        self.cards = ttk.Frame(self.canvas)
        self.cards.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.card_window = self.canvas.create_window((0, 0), window=self.cards, anchor="nw")
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(self.card_window, width=event.width))
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.window.bind("<MouseWheel>", self._scroll)

        footer = ttk.Frame(host)
        footer.pack(fill="x", pady=(8, 0))
        ttk.Label(footer, textvariable=self.status_var, style="Muted.TLabel").pack(side="left", fill="x", expand=True)
        ttk.Button(footer, text="Close", command=self.window.destroy).pack(side="right")

    def _scroll(self, event):
        direction = -int(event.delta / 120) if event.delta else 0
        if direction:
            self.canvas.yview_scroll(direction * 3, "units")
        return "break"

    def refresh(self):
        collections = sorted({str(entry.get("collection") or "") for entry in self.store.prompts if entry.get("collection")})
        modes = sorted({str(thumb.get("mode") or "") for entry in self.store.prompts for thumb in entry.get("thumbnails", []) if thumb.get("mode")})
        self.collection_combo.configure(values=["All collections", *collections])
        self.mode_combo.configure(values=["All models", *modes])
        if self.collection_var.get() not in self.collection_combo["values"]:
            self.collection_var.set("All collections")
        if self.mode_var.get() not in self.mode_combo["values"]:
            self.mode_var.set("All models")
        self.refresh_cards()

    def visible_entries(self):
        query = self.search_var.get().strip().casefold()
        collection = self.collection_var.get()
        mode = self.mode_var.get()
        entries = []
        for entry in self.store.prompts:
            haystack = " ".join([
                str(entry.get("name") or ""),
                str(entry.get("prompt_data", {}).get("prompt") or ""),
                " ".join(entry.get("tags") or []),
                str(entry.get("collection") or ""),
            ]).casefold()
            if query and query not in haystack:
                continue
            if collection != "All collections" and entry.get("collection") != collection:
                continue
            if mode != "All models" and not any(thumb.get("mode") == mode for thumb in entry.get("thumbnails", [])):
                continue
            entries.append(entry)
        return sorted(entries, key=lambda entry: (not bool(entry.get("favorite")), str(entry.get("name") or "").casefold()))

    def refresh_cards(self):
        for child in self.cards.winfo_children():
            child.destroy()
        self.images = {}
        entries = self.visible_entries()
        columns = 3
        for column in range(columns):
            self.cards.grid_columnconfigure(column, weight=1, uniform="library")
        for index, entry in enumerate(entries):
            card = ttk.Frame(self.cards, style="Surface.TFrame", padding=8)
            card.grid(row=index // columns, column=index % columns, sticky="nsew", padx=6, pady=6)
            self._build_card(card, entry)
        if not entries:
            ttk.Label(self.cards, text="No prompts match this view.", style="Muted.TLabel").grid(row=0, column=0, padx=20, pady=20)
        self.status_var.set(
            f"{len(entries)} visible · {len(self.store.prompts)} total prompts · {self.store.path}"
        )

    def _build_card(self, card, entry):
        thumb_frame = tk.Frame(card, bg=self.colors.get("surface_alt", "#1E293B"), height=150)
        thumb_frame.pack(fill="x")
        thumb_frame.pack_propagate(False)
        thumbnail = self.store.latest_thumbnail(entry)
        rendered = False
        if thumbnail and PIL_AVAILABLE:
            try:
                with Image.open(self.store.thumbnail_path(thumbnail)) as source:
                    preview = source.copy()
                preview.thumbnail((300, 145))
                image = ImageTk.PhotoImage(preview)
                self.images[entry["id"]] = image
                tk.Label(thumb_frame, image=image, bg=self.colors.get("surface_alt", "#1E293B")).pack(expand=True)
                rendered = True
            except Exception:
                pass
        if not rendered:
            tk.Label(
                thumb_frame,
                text="No test thumbnail yet",
                bg=self.colors.get("surface_alt", "#1E293B"),
                fg=self.colors.get("muted", "#94A3B8"),
            ).pack(expand=True)
        if thumbnail:
            tk.Label(
                thumb_frame,
                text=str(thumbnail.get("mode") or "Test"),
                bg=self.colors.get("accent_hover", "#0284C7"),
                fg="#FFFFFF",
                font=("Segoe UI Semibold", 8),
                padx=6,
                pady=2,
            ).place(relx=1.0, rely=1.0, x=-6, y=-6, anchor="se")

        title_row = ttk.Frame(card, style="Surface.TFrame")
        title_row.pack(fill="x", pady=(7, 2))
        ttk.Label(title_row, text=("★ " if entry.get("favorite") else "") + entry.get("name", "Prompt"), style="Header.TLabel", wraplength=260).pack(side="left", fill="x", expand=True)
        ttk.Button(title_row, text="★", width=3, command=lambda item=entry: self._favorite(item)).pack(side="right")
        text = str(entry.get("prompt_data", {}).get("prompt") or "")
        ttk.Label(card, text=text[:150] + ("…" if len(text) > 150 else ""), wraplength=280, justify="left").pack(anchor="w", fill="x")
        metadata = []
        if entry.get("collection"):
            metadata.append(entry["collection"])
        if entry.get("tags"):
            metadata.append("#" + " #".join(entry["tags"][:4]))
        ttk.Label(card, text=" · ".join(metadata) or "Uncategorized", style="Muted.TLabel", wraplength=280).pack(anchor="w", pady=(4, 0))
        actions = ttk.Frame(card, style="Surface.TFrame")
        actions.pack(fill="x", pady=(7, 0))
        ttk.Button(actions, text="Use", command=lambda item=entry: self._use([item])).pack(side="left", fill="x", expand=True)
        ttk.Button(actions, text="Edit", command=lambda item=entry: self._edit(item)).pack(side="left", padx=(5, 0))
        ttk.Button(actions, text="Delete", style="Danger.TButton", command=lambda item=entry: self._delete(item)).pack(side="left", padx=(5, 0))

    def _use(self, entries):
        prompts = [self.store.prompt_copy(entry) for entry in entries]
        added, duplicates = self.on_use(prompts)
        messagebox.showinfo("Prompts copied", f"Added {added} prompt(s); skipped {duplicates} duplicate(s).", parent=self.window)

    def _use_visible(self):
        entries = self.visible_entries()
        if entries:
            self._use(entries)

    def _favorite(self, entry):
        self.store.toggle_favorite(entry["id"])
        self.refresh_cards()

    def _delete(self, entry):
        if messagebox.askyesno("Delete library prompt?", f"Delete “{entry.get('name', 'Prompt')}” and its stored thumbnails?", parent=self.window):
            self.store.delete(entry["id"])
            self.refresh()

    def _edit(self, entry):
        dialog = tk.Toplevel(self.window)
        dialog.title("Edit Library Prompt")
        dialog.geometry("720x690")
        dialog.transient(self.window)
        dialog.grab_set()
        host = ttk.Frame(dialog, padding=12)
        host.pack(fill="both", expand=True)
        name_var = tk.StringVar(value=entry.get("name", ""))
        tags_var = tk.StringVar(value=", ".join(entry.get("tags") or []))
        collection_var = tk.StringVar(value=entry.get("collection", ""))
        ttk.Label(host, text="Name").pack(anchor="w")
        ttk.Entry(host, textvariable=name_var).pack(fill="x", pady=(2, 7))
        ttk.Label(host, text="Prompt").pack(anchor="w")
        prompt_text = tk.Text(host, height=7, wrap=tk.WORD)
        prompt_text.insert("1.0", entry.get("prompt_data", {}).get("prompt", ""))
        prompt_text.pack(fill="both", expand=True, pady=(2, 7))
        ttk.Label(host, text="Negative prompt").pack(anchor="w")
        negative_text = tk.Text(host, height=4, wrap=tk.WORD)
        negative_text.insert("1.0", entry.get("prompt_data", {}).get("neg", ""))
        negative_text.pack(fill="x", pady=(2, 7))
        ttk.Label(host, text="Tags (comma-separated)").pack(anchor="w")
        ttk.Entry(host, textvariable=tags_var).pack(fill="x", pady=(2, 7))
        ttk.Label(host, text="Collection").pack(anchor="w")
        ttk.Entry(host, textvariable=collection_var).pack(fill="x", pady=(2, 7))

        params = ttk.Frame(host)
        params.pack(fill="x", pady=(4, 8))
        parameter_vars = {}
        for label, key in (("Width", "width"), ("Height", "height"), ("Steps", "steps"), ("Guidance", "guidance"), ("Seed", "seed"), ("Mu", "mu")):
            column = ttk.Frame(params)
            column.pack(side="left", padx=(0, 8))
            ttk.Label(column, text=label).pack(anchor="w")
            variable = tk.StringVar(value=str(entry.get("prompt_data", {}).get(key, "")))
            ttk.Entry(column, textvariable=variable, width=9).pack()
            parameter_vars[key] = variable

        def save():
            prompt = " ".join(prompt_text.get("1.0", "end-1c").split())
            if not prompt:
                messagebox.showerror("Prompt required", "Prompt text cannot be empty.", parent=dialog)
                return
            prompt_data = copy.deepcopy(entry.get("prompt_data", {}))
            prompt_data["prompt"] = prompt
            negative = " ".join(negative_text.get("1.0", "end-1c").split())
            if negative:
                prompt_data["neg"] = negative
            else:
                prompt_data.pop("neg", None)
            for key, variable in parameter_vars.items():
                value = variable.get().strip()
                if value:
                    prompt_data[key] = value
                else:
                    prompt_data.pop(key, None)
            self.store.update_entry(
                entry["id"],
                name=name_var.get(),
                prompt_data=prompt_data,
                tags=tags_var.get().split(","),
                collection=collection_var.get(),
            )
            dialog.destroy()
            self.refresh()

        buttons = ttk.Frame(host)
        buttons.pack(fill="x", pady=(6, 0))
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
        ttk.Button(buttons, text="Save", style="Accent.TButton", command=save).pack(side="right", padx=(0, 6))

    def _add_current(self):
        prompts = self.current_prompts()
        added, merged = self.store.import_prompts(prompts, source={"type": "current_settings"})
        messagebox.showinfo("Current prompts collected", f"Added {added}; merged {merged} existing prompt(s).", parent=self.window)
        self.refresh()

    def _collect_jobs(self):
        added, merged, job_count = self.store.import_jobs(self.jobs())
        messagebox.showinfo(
            "Job prompts collected",
            f"Scanned {job_count} job(s) containing prompts. Added {added}; merged {merged} existing prompt(s).",
            parent=self.window,
        )
        self.refresh()

    def _export(self):
        destination = filedialog.asksaveasfilename(parent=self.window, defaultextension=".json", filetypes=[("JSON", "*.json")])
        if destination:
            self.store.export_to(destination)

    def _import_file(self):
        source = filedialog.askopenfilename(parent=self.window, filetypes=[("JSON", "*.json")])
        if not source:
            return
        try:
            added, merged = self.store.import_file(source)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not import library", str(exc), parent=self.window)
            return
        messagebox.showinfo("Library imported", f"Added {added}; merged {merged} existing prompt(s).", parent=self.window)
        self.refresh()
