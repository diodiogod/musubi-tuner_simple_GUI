"""Contain GUI-launched commands so their descendants cannot outlive a job."""

from __future__ import annotations

import os
import signal
import subprocess
import threading


class ProcessTreeScope:
    """Own a subprocess tree and reclaim every descendant when closed."""

    def __init__(self, process: subprocess.Popen):
        self.process = process
        self._lock = threading.Lock()
        self._closed = False
        self._job = self._create_windows_job(process) if os.name == "nt" else None
        self._pgid = os.getpgid(process.pid) if os.name != "nt" else None

    @staticmethod
    def _create_windows_job(process: subprocess.Popen):
        import ctypes
        from ctypes import wintypes

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "reads", "writes", "others", "read_bytes", "write_bytes", "other_bytes",
            )]

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                ("max_working_set", ctypes.c_size_t), ("active_processes", wintypes.DWORD),
                ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("basic", BasicLimits), ("io", IoCounters),
                ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        limits = ExtendedLimits()
        limits.basic.flags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        configured = kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits))
        assigned = configured and kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(process._handle))
        if not assigned:
            kernel32.CloseHandle(job)
            return None
        return job

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if os.name == "nt" and self._job:
                import ctypes
                from ctypes import wintypes
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
                kernel32.CloseHandle(self._job)
                self._job = None
            elif os.name != "nt":
                try:
                    os.killpg(self._pgid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    pass

    @property
    def contained(self) -> bool:
        return os.name != "nt" or self._job is not None
