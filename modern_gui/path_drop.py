"""Small native Windows folder-drop dialog used by the local web UI.

Browsers can accept a directory dropped on a page, but they intentionally do
not expose the directory's absolute Windows path to ordinary JavaScript.  The
local GUI therefore uses this short-lived Win32 window as a bridge: Explorer
can drop one or more folders onto it and their paths are returned to the caller.
"""

from __future__ import annotations

import ctypes
import os
import uuid
from pathlib import Path
from typing import Any


def choose_directories(title: str = "Drop dataset folders") -> list[str]:
    """Show a native folder-drop window and return every dropped folder.

    The web server is also importable on non-Windows systems for tests and
    documentation tooling.  There is no useful native Explorer bridge there,
    so return an empty selection rather than importing ``ctypes.wintypes`` or
    failing at module import time.
    """

    if os.name != "nt":
        return []
    return _choose_windows_directories(title)


def choose_directory(title: str = "Drop a dataset folder") -> str:
    """Backward-compatible single-folder wrapper."""

    return (choose_directories(title) or [""])[0]


def _choose_windows_directories(title: str) -> list[str]:  # pragma: no cover - GUI exercised manually on Windows
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    HINSTANCE = wintypes.HINSTANCE
    HWND = wintypes.HWND
    HMENU = wintypes.HMENU
    HICON = wintypes.HICON
    # Some Python Windows builds do not expose HCURSOR as a named wintypes
    # alias even though it is represented by the same opaque handle type.
    HCURSOR = getattr(wintypes, "HCURSOR", wintypes.HANDLE)
    HBRUSH = wintypes.HBRUSH
    LPVOID = wintypes.LPVOID
    LRESULT = ctypes.c_ssize_t

    WNDPROC = ctypes.WINFUNCTYPE(LRESULT, HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

    class WNDCLASSW(ctypes.Structure):
        _fields_ = [
            ("style", wintypes.UINT),
            ("lpfnWndProc", WNDPROC),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", HINSTANCE),
            ("hIcon", HICON),
            ("hCursor", HCURSOR),
            ("hbrBackground", HBRUSH),
            ("lpszMenuName", wintypes.LPCWSTR),
            ("lpszClassName", wintypes.LPCWSTR),
        ]

    user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
    user32.RegisterClassW.restype = wintypes.ATOM
    user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, HINSTANCE]
    user32.UnregisterClassW.restype = wintypes.BOOL
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        HWND,
        HMENU,
        HINSTANCE,
        LPVOID,
    ]
    user32.CreateWindowExW.restype = HWND
    user32.DestroyWindow.argtypes = [HWND]
    user32.DestroyWindow.restype = wintypes.BOOL
    user32.DefWindowProcW.argtypes = [HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.DefWindowProcW.restype = LRESULT
    user32.ShowWindow.argtypes = [HWND, ctypes.c_int]
    user32.ShowWindow.restype = wintypes.BOOL
    user32.UpdateWindow.argtypes = [HWND]
    user32.UpdateWindow.restype = wintypes.BOOL
    user32.SetForegroundWindow.argtypes = [HWND]
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.SetWindowTextW.argtypes = [HWND, wintypes.LPCWSTR]
    user32.SetWindowTextW.restype = wintypes.BOOL
    user32.PostQuitMessage.argtypes = [ctypes.c_int]
    user32.PostQuitMessage.restype = None
    user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), HWND, wintypes.UINT, wintypes.UINT]
    user32.GetMessageW.restype = ctypes.c_int
    user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.TranslateMessage.restype = wintypes.BOOL
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.DispatchMessageW.restype = LRESULT
    user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    user32.GetSystemMetrics.restype = ctypes.c_int
    user32.GetSysColorBrush.argtypes = [ctypes.c_int]
    user32.GetSysColorBrush.restype = HBRUSH
    user32.LoadCursorW.argtypes = [HINSTANCE, wintypes.LPCWSTR]
    user32.LoadCursorW.restype = HCURSOR

    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = HINSTANCE

    shell32.DragAcceptFiles.argtypes = [HWND, wintypes.BOOL]
    shell32.DragAcceptFiles.restype = None
    shell32.DragQueryFileW.argtypes = [wintypes.HANDLE, wintypes.UINT, wintypes.LPWSTR, wintypes.UINT]
    shell32.DragQueryFileW.restype = wintypes.UINT
    shell32.DragFinish.argtypes = [wintypes.HANDLE]
    shell32.DragFinish.restype = None

    WM_DROPFILES = 0x0233
    WM_COMMAND = 0x0111
    WM_CLOSE = 0x0010
    WM_DESTROY = 0x0002
    WS_EX_TOPMOST = 0x00000008
    WS_EX_APPWINDOW = 0x00040000
    WS_OVERLAPPED = 0x00000000
    WS_CAPTION = 0x00C00000
    WS_SYSMENU = 0x00080000
    WS_MINIMIZEBOX = 0x00020000
    WS_VISIBLE = 0x10000000
    WS_CHILD = 0x40000000
    WS_TABSTOP = 0x00010000
    SS_LEFT = 0x00000000
    BS_DEFPUSHBUTTON = 0x00000001
    SW_SHOW = 5
    COLOR_WINDOW = 5
    IDC_ARROW = ctypes.cast(ctypes.c_void_p(32512), wintypes.LPCWSTR)
    STATIC = "STATIC"
    BUTTON = "BUTTON"
    cancel_id = 1002
    instance = kernel32.GetModuleHandleW(None)
    class_name = f"MusubiFolderDrop_{uuid.uuid4().hex}"
    result: list[str] = []
    state: dict[str, Any] = {}

    def set_status(message: str) -> None:
        status = state.get("status")
        if status:
            user32.SetWindowTextW(status, message)

    def window_proc(hwnd: HWND, message: int, wparam: int, lparam: int) -> int:
        if message == WM_DROPFILES:
            hdrop = wintypes.HANDLE(wparam)
            count = shell32.DragQueryFileW(hdrop, 0xFFFFFFFF, None, 0)
            paths: list[str] = []
            for index in range(count):
                length = shell32.DragQueryFileW(hdrop, index, None, 0)
                buffer = ctypes.create_unicode_buffer(length + 1)
                shell32.DragQueryFileW(hdrop, index, buffer, length + 1)
                paths.append(buffer.value)
            shell32.DragFinish(hdrop)
            for candidate in paths:
                try:
                    resolved = str(Path(candidate).resolve())
                    if Path(resolved).is_dir() and resolved not in result:
                        result.append(resolved)
                except (OSError, ValueError):
                    continue
            if result:
                user32.DestroyWindow(hwnd)
            else:
                set_status("Please drop one or more folders (not individual files).")
            return 0
        if message == WM_COMMAND and (wparam & 0xFFFF) == cancel_id:
            user32.DestroyWindow(hwnd)
            return 0
        if message == WM_CLOSE:
            user32.DestroyWindow(hwnd)
            return 0
        if message == WM_DESTROY:
            user32.PostQuitMessage(0)
            return 0
        return int(user32.DefWindowProcW(hwnd, message, wparam, lparam))

    callback = WNDPROC(window_proc)
    window_class = WNDCLASSW()
    window_class.lpfnWndProc = callback
    window_class.hInstance = instance
    window_class.hCursor = user32.LoadCursorW(None, IDC_ARROW)
    window_class.hbrBackground = user32.GetSysColorBrush(COLOR_WINDOW)
    window_class.lpszClassName = class_name
    if not user32.RegisterClassW(ctypes.byref(window_class)):
        return []

    width, height = 600, 245
    left = max(0, (user32.GetSystemMetrics(0) - width) // 2)
    top = max(0, (user32.GetSystemMetrics(1) - height) // 2)
    hwnd = user32.CreateWindowExW(
        WS_EX_TOPMOST | WS_EX_APPWINDOW,
        class_name,
        title,
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_VISIBLE,
        left,
        top,
        width,
        height,
        None,
        None,
        instance,
        None,
    )
    if not hwnd:
        user32.UnregisterClassW(class_name, instance)
        return []

    try:
        state["status"] = user32.CreateWindowExW(
            0,
            STATIC,
            "Drag a folder from Windows Explorer onto this window.\nThe folder path will be added to the dataset.",
            WS_CHILD | WS_VISIBLE | SS_LEFT,
            24,
            30,
            540,
            70,
            hwnd,
            None,
            instance,
            None,
        )
        user32.CreateWindowExW(
            0,
            STATIC,
            "Only folders are accepted; cancel leaves the TOML unchanged.",
            WS_CHILD | WS_VISIBLE | SS_LEFT,
            24,
            112,
            540,
            28,
            hwnd,
            None,
            instance,
            None,
        )
        user32.CreateWindowExW(
            0,
            BUTTON,
            "Cancel",
            WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_DEFPUSHBUTTON,
            240,
            165,
            120,
            32,
            hwnd,
            HMENU(cancel_id),
            instance,
            None,
        )
        shell32.DragAcceptFiles(hwnd, True)
        user32.ShowWindow(hwnd, SW_SHOW)
        user32.UpdateWindow(hwnd)
        user32.SetForegroundWindow(hwnd)
        message = wintypes.MSG()
        while True:
            status = user32.GetMessageW(ctypes.byref(message), None, 0, 0)
            if status <= 0:
                break
            user32.TranslateMessage(ctypes.byref(message))
            user32.DispatchMessageW(ctypes.byref(message))
    finally:
        user32.UnregisterClassW(class_name, instance)
    return result


__all__ = ["choose_directories", "choose_directory"]
