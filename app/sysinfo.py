"""Lightweight CPU / memory / GPU probes without third-party dependencies.

Windows exposes everything through ``ctypes``; on other platforms the functions
degrade to ``None`` so the GUI can simply hide the value.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import time

__all__ = ["cpu_percent", "memory_usage", "process_memory_mb", "gpu_info"]


class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wt.DWORD), ("dwHighDateTime", wt.DWORD)]


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wt.DWORD),
        ("dwMemoryLoad", wt.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


_IS_WINDOWS = os.name == "nt"
_prev_cpu: tuple[int, int, int] | None = None
_last_cpu_value: float = 0.0


def _filetime_to_int(ft) -> int:
    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime


def cpu_percent() -> float | None:
    """System-wide CPU utilisation in percent (sampled between calls)."""
    global _prev_cpu, _last_cpu_value
    if not _IS_WINDOWS:
        return _last_cpu_value or None
    idle, kernel, user = _FILETIME(), _FILETIME(), _FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return None
    now = (_filetime_to_int(idle), _filetime_to_int(kernel), _filetime_to_int(user))
    if _prev_cpu is None:
        _prev_cpu = now
        return None
    d_idle = now[0] - _prev_cpu[0]
    d_kernel = now[1] - _prev_cpu[1]
    d_user = now[2] - _prev_cpu[2]
    _prev_cpu = now
    total = d_kernel + d_user
    if total <= 0:
        return _last_cpu_value
    busy = max(total - d_idle, 0)
    _last_cpu_value = round(100.0 * busy / total, 1)
    return _last_cpu_value


def memory_usage() -> tuple[float, float] | None:
    """Return (used_GB, total_GB) of physical memory."""
    if _IS_WINDOWS:
        status = _MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            total = status.ullTotalPhys / 1024 ** 3
            used = (status.ullTotalPhys - status.ullAvailPhys) / 1024 ** 3
            return round(used, 2), round(total, 2)
    try:
        import resource  # noqa: PLC0415  (Unix fallback)

        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
        return None, round(total, 2)
    except Exception:  # noqa: BLE001
        return None


def process_memory_mb() -> float | None:
    """Resident set size of the current process in MiB."""
    if not _IS_WINDOWS:
        return None
    try:
        class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wt.DWORD),
                ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return round(counters.WorkingSetSize / 1024 ** 2, 1)
    except Exception:  # noqa: BLE001
        return None
    return None


def gpu_info() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"CUDA {torch.cuda.get_device_name(0)}"
    except Exception:  # noqa: BLE001
        pass
    return "CPU"


class FPSMeter:
    """Tiny helper for sampling the probes on a fixed interval."""

    def __init__(self, interval: float = 0.8) -> None:
        self.interval = interval
        self.last = 0.0
        self.cpu: float | None = None

    def poll(self) -> float | None:
        now = time.perf_counter()
        if now - self.last >= self.interval:
            self.last = now
            self.cpu = cpu_percent()
        return self.cpu
