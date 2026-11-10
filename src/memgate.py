"""Start only when enough committed memory (page-file-backed virtual memory) is free.  This laptop runs several other projects; when the commit limit is reached every allocation fails
(MemoryError, OpenBLAS errors), which killed three of my jobs.  memgate.wait(gb) blocks (polling every 30 s with a random jitter) until at least `gb` GB of commit are available."""
import ctypes, random, time


class _MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_commit_gb():
    s = _MS(); s.dwLength = ctypes.sizeof(_MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)); return s.ullAvailPageFile / 2 ** 30


def wait(gb=2.0, max_hours=10.0):
    import os
    if os.environ.get('MEMGATE_OFF'):
        return
    t0 = time.time(); time.sleep(random.uniform(0, 20))
    while free_commit_gb() < gb and (time.time() - t0) < max_hours * 3600:
        time.sleep(30 + random.uniform(0, 15))
