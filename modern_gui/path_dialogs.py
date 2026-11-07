"""Local file selection without loading the Classic GUI or training runtime."""
from pathlib import Path


def initial_location(initial: str, *, directory: bool = False) -> dict[str, str]:
    """Tk's initialdir must be a folder, even when the field holds a filename."""
    value = str(initial or "").strip().strip('"')
    if not value:
        return {}
    path = Path(value).expanduser()
    try:
        if path.is_dir():
            return {"initialdir": str(path)}
        parent = path.parent
        if parent.is_dir():
            result = {"initialdir": str(parent)}
            if not directory and (path.is_file() or path.suffix):
                result["initialfile"] = path.name
            return result
        # Old recipes can point at a file or folder that has since moved.
        for parent in path.parents:
            if parent.is_dir():
                return {"initialdir": str(parent)}
    except (OSError, ValueError):
        pass
    return {}


def choose_path(kind: str = "file", initial: str = "") -> str:
    import tkinter as tk
    from tkinter import filedialog

    directory = kind == "directory"
    options = initial_location(initial, directory=directory)
    root = tk.Tk()
    root.withdraw()
    root.title("Musubi Studio · Choose a path")
    root.attributes("-topmost", True)
    try:
        if directory:
            selected = filedialog.askdirectory(parent=root, title="Musubi Studio · Choose folder", **options)
        else:
            selected = filedialog.askopenfilename(parent=root, title="Musubi Studio · Choose file", **options)
        return selected or ""
    finally:
        root.destroy()
