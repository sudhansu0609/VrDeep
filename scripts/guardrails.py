"""
guardrails.py - Resource safety layer for the VrDeep pipeline.

WHY THIS EXISTS
---------------
Starting a conversion used to lock up the whole machine and force a hard
power-cycle. Three separate resource exhaustions caused that, and this
module contains all three:

  1. DISK.  FaceFusion extracts every frame to PNG by default, and the VR
     split step wrote near-lossless H.264 (-crf 6 -preset slow). A few
     minutes of 4K VR is hundreds of GB of temp data. Drive B: has ~68 GB
     free. When a Windows volume that holds temp/pagefile data fills up,
     the machine stops responding rather than failing cleanly.

  2. VRAM.  '--video-memory-strategy tolerant' keeps every model resident.
     Once the 16 GB card is full the WDDM driver spills GPU allocations
     into system RAM, which thrashes the pagefile and freezes the desktop.

  3. CPU.  x264 '-preset slow' / x265 '-preset medium' pin all 28 logical
     cores at 100% at Normal priority, so the shell never gets scheduled
     and the UI appears dead even when the run is progressing.

Everything here is defensive and layered:

  preflight()      refuses a run that obviously cannot fit on disk/VRAM
  clamp_config()   rewrites unsafe settings into safe ones
  ResourceGuard    puts every child process in a Windows Job Object with a
                   hard memory cap, a CPU rate cap and below-normal
                   priority, so children physically cannot take the machine
  Watchdog         polls free RAM / free disk during the run and aborts the
                   job if the machine gets close to the edge anyway

A failed conversion with a clear message is always better than a frozen PC.
"""

import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from ctypes import wintypes

IS_WINDOWS = sys.platform == "win32"

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
GUARD_CONFIG_PATH = os.path.join(SCRIPTS_DIR, "guardrails.json")

GB = 1024 ** 3
MB = 1024 ** 2


class GuardAbort(Exception):
    """Raised when the guard rails stop a run (watchdog trip or user cancel)."""


# ---------------------------------------------------------------------------
# Win32 plumbing
# ---------------------------------------------------------------------------

if IS_WINDOWS:
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
else:  # pragma: no cover - the app is Windows-only, this keeps imports safe
    _kernel32 = None

JobObjectExtendedLimitInformation = 9
JobObjectCpuRateControlInformation = 15
JobObjectBasicProcessIdList = 3

PROCESS_SUSPEND_RESUME = 0x0800
ERROR_MORE_DATA = 234

if IS_WINDOWS:
    # NtSuspendProcess / NtResumeProcess are what every "suspend process" tool
    # on Windows uses. There is no documented Win32 equivalent - SuspendThread
    # would mean enumerating threads and racing every thread the child creates
    # while we walk them.
    _ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
else:  # pragma: no cover
    _ntdll = None

JOB_OBJECT_LIMIT_AFFINITY = 0x00000010
JOB_OBJECT_LIMIT_PRIORITY_CLASS = 0x00000020
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

JOB_OBJECT_CPU_RATE_CONTROL_ENABLE = 0x00000001
JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP = 0x00000004

BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
IDLE_PRIORITY_CLASS = 0x00000040


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
        ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _CPU_RATE_UNION(ctypes.Union):
    _fields_ = [
        ("CpuRate", wintypes.DWORD),
        ("Weight", wintypes.DWORD),
    ]


class _JOBOBJECT_CPU_RATE_CONTROL_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("ControlFlags", wintypes.DWORD),
        ("u", _CPU_RATE_UNION),
    ]


TH32CS_SNAPPROCESS = 0x00000002
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

if IS_WINDOWS:
    _psapi = ctypes.WinDLL("psapi", use_last_error=True)
else:  # pragma: no cover
    _psapi = None


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", ctypes.c_wchar * 260),
    ]


class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def process_memory(pid):
    """(working set, private commit) for one process, or (0, 0) if unreadable."""
    if not IS_WINDOWS or not _psapi:
        return 0, 0
    handle = _kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return 0, 0
    try:
        counters = _PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        if not _psapi.GetProcessMemoryInfo(wintypes.HANDLE(handle),
                                           ctypes.byref(counters),
                                           ctypes.sizeof(counters)):
            return 0, 0
        return int(counters.WorkingSetSize), int(counters.PrivateUsage)
    finally:
        _kernel32.CloseHandle(wintypes.HANDLE(handle))


class _PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t),
        ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t),
        ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t),
        ("HandleCount", wintypes.DWORD),
        ("ProcessCount", wintypes.DWORD),
        ("ThreadCount", wintypes.DWORD),
    ]


def system_memory_breakdown():
    """
    Where the machine's physical memory is, in bytes, beyond process totals.

    Adding up every process's working set routinely accounts for only half of
    what Windows calls "in use", and the missing half is what makes a free-RAM
    abort unexplainable: the kernel pools and the compression store hold real
    physical pages that no process shows. A driver leaking nonpaged pool looks
    exactly like a pipeline that is using too much memory, right up until this
    line is printed.

    Empty dict if the counters cannot be read.
    """
    if not IS_WINDOWS or not _psapi:
        return {}
    info = _PERFORMANCE_INFORMATION()
    info.cb = ctypes.sizeof(info)
    if not _psapi.GetPerformanceInfo(ctypes.byref(info), ctypes.sizeof(info)):
        return {}
    page = int(info.PageSize) or 4096
    return {
        "commit_total": int(info.CommitTotal) * page,
        "commit_limit": int(info.CommitLimit) * page,
        "commit_peak": int(info.CommitPeak) * page,
        "physical_total": int(info.PhysicalTotal) * page,
        "physical_available": int(info.PhysicalAvailable) * page,
        "system_cache": int(info.SystemCache) * page,
        "kernel_paged": int(info.KernelPaged) * page,
        "kernel_nonpaged": int(info.KernelNonpaged) * page,
        "processes": int(info.ProcessCount),
        "threads": int(info.ThreadCount),
        "handles": int(info.HandleCount),
    }


PROCESS_SET_QUOTA = 0x0100
PROCESS_QUERY_INFORMATION = 0x0400


def empty_working_set(pid):
    """
    Hand one process's resident pages back to Windows immediately.

    Suspending a process does not by itself free any physical memory: Windows
    trims a frozen process eventually, under pressure, on its own schedule.
    When the machine is seconds from an abort, "eventually" is not a plan.
    EmptyWorkingSet does it now - the pages go to the modified and standby
    lists and become available, and the process faults back the ones it still
    needs when it is resumed.

    Only ever called on a process that is already frozen. Doing it to a running
    process would make it fault its whole world back in immediately, which is
    all cost and no benefit.
    """
    if not IS_WINDOWS or not _psapi:
        return False
    handle = _kernel32.OpenProcess(
        PROCESS_QUERY_INFORMATION | PROCESS_SET_QUOTA, False, int(pid))
    if not handle:
        return False
    try:
        return bool(_psapi.EmptyWorkingSet(wintypes.HANDLE(handle)))
    except Exception:
        return False
    finally:
        _kernel32.CloseHandle(wintypes.HANDLE(handle))


def top_memory_processes(limit=5, exclude_pids=()):
    """
    The machine's biggest memory holders right now, largest first.

    The watchdog trips on free RAM, and free RAM is a property of the whole
    machine. The job's own peak already answers "did the pipeline do this?";
    when the answer is no - which on a machine that is also doing something
    else is the common case - nothing answered "then what did?", and the user
    was left to conclude they needed a smaller pipeline or more RAM when what
    they actually needed was to close a browser.

    Returns [(name, pid, working_set_bytes), ...].
    """
    if not IS_WINDOWS:
        return []
    snapshot = _kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snapshot in (0, INVALID_HANDLE_VALUE, -1):
        return []
    exclude = set(int(p) for p in exclude_pids)
    found = []
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        ok = _kernel32.Process32FirstW(wintypes.HANDLE(snapshot), ctypes.byref(entry))
        while ok:
            pid = int(entry.th32ProcessID)
            if pid > 4 and pid not in exclude:
                working_set, _ = process_memory(pid)
                if working_set:
                    found.append((entry.szExeFile, pid, working_set))
            ok = _kernel32.Process32NextW(wintypes.HANDLE(snapshot), ctypes.byref(entry))
    except Exception:
        return []
    finally:
        _kernel32.CloseHandle(wintypes.HANDLE(snapshot))
    found.sort(key=lambda row: row[2], reverse=True)
    return found[:limit]


def top_commit_processes(limit=6, exclude_pids=()):
    """
    The machine's biggest holders of COMMITTED memory right now, largest first.

    Not the same question as top_memory_processes(). That one sorts on working
    set, which is what the free-RAM watchdog trips on; this one sorts on
    PrivateUsage, which *is* a process's commit charge - the address space it
    has booked against RAM plus pagefile whether or not it ever makes it
    resident. A run is refused for available commit far more often than for
    free RAM, and on that refusal the working-set list names the wrong
    programs: a CUDA process or a browser can hold three times its working set
    in commit, and the thing worth closing never appears.

    Duplicates are grouped by exe name, because that is how they are closed:
    three Ollama runners of one model are a single thing to shut down, not
    three lines of noise that between them hide the next holder.

    Returns [(name, count, total_private_bytes), ...].
    """
    if not IS_WINDOWS:
        return []
    snapshot = _kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snapshot in (0, INVALID_HANDLE_VALUE, -1):
        return []
    exclude = set(int(p) for p in exclude_pids)
    grouped = {}
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        ok = _kernel32.Process32FirstW(wintypes.HANDLE(snapshot), ctypes.byref(entry))
        while ok:
            pid = int(entry.th32ProcessID)
            if pid > 4 and pid not in exclude:
                _, private = process_memory(pid)
                if private:
                    name = entry.szExeFile
                    count, total = grouped.get(name, (0, 0))
                    grouped[name] = (count + 1, total + private)
            ok = _kernel32.Process32NextW(wintypes.HANDLE(snapshot), ctypes.byref(entry))
    except Exception:
        return []
    finally:
        _kernel32.CloseHandle(wintypes.HANDLE(snapshot))
    found = [(name, count, total) for name, (count, total) in grouped.items()]
    found.sort(key=lambda row: row[2], reverse=True)
    return found[:limit]


def format_commit_holders(holders):
    """'llama-server.exe x3 8.4 GB, pythonw.exe 3.2 GB' from top_commit_processes()."""
    return ", ".join(
        f"{name}{f' x{count}' if count > 1 else ''} {fmt_gb(total)}"
        for name, count, total in holders
    )


def parent_map():
    """
    {pid: parent pid} for every process on the machine, from one snapshot.

    On its own this is not a safe basis for acting on a process: Windows
    recycles PIDs and does not clear a dead parent's ID from its children, so a
    walk down from a root can graft a complete stranger onto the tree. Every
    caller here either uses it as an *exclusion* list (own_pid_tree, where a
    stale PID at worst leaves one line out of a report) or intersects the
    result with the job object's own PID list (ResourceGuard.lane_pids), which
    is the kernel's answer to "who is in this run" and cannot contain anything
    this run did not start.
    """
    parents = {}
    if not IS_WINDOWS:
        return parents
    snapshot = _kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snapshot in (0, INVALID_HANDLE_VALUE, -1):
        return parents
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        ok = _kernel32.Process32FirstW(wintypes.HANDLE(snapshot), ctypes.byref(entry))
        while ok:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            ok = _kernel32.Process32NextW(wintypes.HANDLE(snapshot), ctypes.byref(entry))
    except Exception:
        return {}
    finally:
        _kernel32.CloseHandle(wintypes.HANDLE(snapshot))
    return parents


def own_pid_tree(root_pid=None):
    """
    This process and everything descended from it.

    Only ever used to take the app's own memory back out of a "what is holding
    the machine's memory" list: telling someone to close the program that is
    printing the message is not advice. A parent-PID walk is unsafe as a way to
    *act* on processes (a recycled PID can graft a stranger onto the tree), but
    as an exclusion list the worst a stale PID can do is leave one line out.
    """
    root_pid = int(root_pid or os.getpid())
    if not IS_WINDOWS:
        return {root_pid}
    parents = parent_map()
    if not parents:
        return {root_pid}

    tree = {root_pid}
    # Repeat until nothing new joins: the snapshot is in no particular order,
    # so one pass would miss a grandchild listed before its parent.
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in tree and pid not in tree:
                tree.add(pid)
                changed = True
    return tree


class _JOBOBJECT_BASIC_PROCESS_ID_LIST(ctypes.Structure):
    """Header of the job's live PID list; the list itself follows in the buffer."""
    _fields_ = [
        ("NumberOfAssignedProcesses", wintypes.DWORD),
        ("NumberOfProcessIdsInList", wintypes.DWORD),
        ("ProcessIdList", ctypes.c_size_t * 1),
    ]


def _job_process_ids(job, capacity=512):
    """
    Every process currently in `job`, asked of the kernel.

    Deliberately not a walk of the process tree: FaceFusion spawns ffmpeg,
    ffmpeg spawns nothing but could, and a tree walk keyed on parent PIDs both
    misses re-parented children and risks freezing an unrelated process that
    has inherited a recycled PID. The job knows exactly who is in it, and by
    construction that is exactly what this run started.

    That property is also what makes ResourceGuard.lane_pids() safe. A lane is
    a subset of the job, and finding it does need a parent-PID walk (the job
    object records membership, not lineage) - but the walk only ever *selects
    from* this list. So the worst a recycled PID can do there is include or
    exclude another process of this same run; it can never reach Ollama, a
    browser, or anything else on the machine, because nothing outside the job
    is a candidate in the first place. Every lane operation - suspend, resume,
    terminate - is therefore bounded by what this run started.
    """
    if not job or not IS_WINDOWS:
        return []
    offset = (ctypes.sizeof(_JOBOBJECT_BASIC_PROCESS_ID_LIST)
              - ctypes.sizeof(ctypes.c_size_t))
    while True:
        size = offset + ctypes.sizeof(ctypes.c_size_t) * capacity
        buf = ctypes.create_string_buffer(size)
        returned = wintypes.DWORD(0)
        ok = _kernel32.QueryInformationJobObject(
            job, JobObjectBasicProcessIdList, buf, size, ctypes.byref(returned))
        if not ok:
            if ctypes.get_last_error() == ERROR_MORE_DATA and capacity < 16384:
                capacity *= 4
                continue
            return []
        info = ctypes.cast(
            buf, ctypes.POINTER(_JOBOBJECT_BASIC_PROCESS_ID_LIST)).contents
        count = int(info.NumberOfProcessIdsInList)
        if count <= 0:
            return []
        pids = ctypes.cast(ctypes.byref(buf, offset),
                           ctypes.POINTER(ctypes.c_size_t * count)).contents
        return [int(pid) for pid in pids]


def _set_process_suspended(pid, suspend):
    """Freeze or thaw one process. False if it is gone or not ours to touch."""
    if not IS_WINDOWS or not _ntdll:
        return False
    handle = _kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, int(pid))
    if not handle:
        return False
    try:
        call = _ntdll.NtSuspendProcess if suspend else _ntdll.NtResumeProcess
        return call(wintypes.HANDLE(handle)) == 0
    except Exception:
        return False
    finally:
        _kernel32.CloseHandle(wintypes.HANDLE(handle))


PROCESS_TERMINATE = 0x0001


def _terminate_process(pid, exit_code=1):
    """
    Kill one process. False if it is gone or not ours to touch.

    Only ever called on a PID that came out of ResourceGuard.lane_pids(), i.e.
    one the job object says belongs to this run - see _job_process_ids for why
    that bound is the thing that makes it safe. The whole-job kill still goes
    through TerminateJobObject; this exists so a single stuck lane can be shot
    without taking the other eye's hour of work with it.
    """
    if not IS_WINDOWS:
        return False
    handle = _kernel32.OpenProcess(PROCESS_TERMINATE, False, int(pid))
    if not handle:
        return False
    try:
        return bool(_kernel32.TerminateProcess(wintypes.HANDLE(handle), int(exit_code)))
    except Exception:
        return False
    finally:
        _kernel32.CloseHandle(wintypes.HANDLE(handle))


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def memory_status():
    """(total_phys, avail_phys, avail_pagefile) in bytes. Zeros off-Windows."""
    if not IS_WINDOWS:
        return (0, 0, 0)
    stat = _MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    if not _kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return (0, 0, 0)
    return (stat.ullTotalPhys, stat.ullAvailPhys, stat.ullAvailPageFile)


def wait_for_commit_release(timeout=12.0, settle=0.4, quiet_samples=3, log_cb=None):
    """
    Block until the machine's available commit stops climbing, or `timeout`.

    Called between queued jobs. A finished run's children hand their committed
    memory back over a second or two after they exit - the kernel tears a CUDA
    process's address space down on its own schedule, and the ffmpeg children
    the job-close kills die a moment behind it - and the next job is sized
    against the system commit charge the instant it is launched. Waiting until
    the available-commit figure plateaus (rather than for a fixed sleep) keeps
    the gap short when the release is quick and safe when it is slow, so a job
    is never refused for memory the one before it has not finished returning.

    Returns the bytes of commit that came back while waiting (0 off-Windows).
    """
    if not IS_WINDOWS:
        return 0
    _, _, start = memory_status()
    best = start
    stable = 0
    deadline = time.time() + max(0.0, timeout)
    while time.time() < deadline:
        time.sleep(settle)
        _, _, avail = memory_status()
        if avail <= best:
            stable += 1
            if stable >= quiet_samples:
                break
        else:
            best = avail
            stable = 0
    recovered = max(0, best - start)
    if log_cb and recovered:
        try:
            log_cb(f"[Guard] Waited for memory to come back: {fmt_gb(recovered)} of "
                   f"commit released, {fmt_gb(best)} now free.")
        except Exception:
            pass
    return recovered


def commit_limit_bytes():
    """
    The system commit limit: physical RAM plus the current pagefile.

    This - not physical RAM - is what a Job Object's JobMemoryLimit is measured
    against, so it is what the job cap has to be sized from. See
    default_guard_config().
    """
    if not IS_WINDOWS:
        return 0
    stat = _MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    if not _kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return 0
    return stat.ullTotalPageFile


def free_disk_bytes(path):
    """Free bytes on the volume holding `path`, walking up to an existing dir."""
    probe = os.path.abspath(path)
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent

    if not IS_WINDOWS:
        try:
            import shutil
            return shutil.disk_usage(probe).free
        except Exception:
            return 0

    free = ctypes.c_ulonglong(0)
    total = ctypes.c_ulonglong(0)
    ok = _kernel32.GetDiskFreeSpaceExW(
        ctypes.c_wchar_p(probe), ctypes.byref(free), ctypes.byref(total), None
    )
    return free.value if ok else 0


def gpu_memory_mb():
    """(total_mb, used_mb) from nvidia-smi, or (0, 0) if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0 and result.stdout.strip():
            first = result.stdout.strip().splitlines()[0]
            total, used = [int(x.strip()) for x in first.split(",")[:2]]
            return (total, used)
    except Exception:
        pass
    return (0, 0)


# Filled once, by pagefiles(). A list holding one list, so the "already asked"
# state is distinguishable from "asked, and the answer was nothing".
_PAGEFILES_CACHE = []


def pagefiles():
    """
    Every pagefile, with the physical disk it sits on.

    Returns a list of dicts: path, drive, media ('HDD'/'SSD'/'Unspecified'),
    disk (friendly name), allocated_mb, peak_used_mb, configured_mb. Empty when
    the query fails, which is not an error - the caller degrades to saying
    nothing.

    allocated_mb and configured_mb are two different facts and the gap between
    them is the point: Win32_PageFileUsage reports the file that exists on disk
    right now, Win32_PageFileSetting reports the size Windows has been told to
    make it. Resizing a pagefile does not resize it - it schedules it for the
    next boot - and until then the commit limit is still the old file's.
    configured_mb is 0 when the size is system-managed or unreadable.

    Why this is worth a WMI round trip: a VR run commits far more than this
    machine has in RAM (~42 GB against 32 GB, measured), so Windows *will*
    page. Where it pages to is then the single biggest factor in whether the
    desktop stays usable, and Windows spreads paging across every pagefile it
    has - so one pagefile on a mechanical disk drags the whole system down to
    that disk's random-IO rate no matter how fast the others are.
    """
    if not IS_WINDOWS:
        return []
    if _PAGEFILES_CACHE:
        # Cached for the life of the process: this shells out to PowerShell and
        # takes a second or two, and the watchdog wants to be able to mention it
        # in an abort message without adding a second to the abort.
        return _PAGEFILES_CACHE[0]

    # The configured-size lookup is wrapped in its own try/catch: it is the
    # newer half of this query and a machine where Win32_PageFileSetting is
    # empty (a system-managed pagefile) or blocked must still get the
    # mechanical-disk warning, which is the half that prevents a freeze.
    script = (
        "$ErrorActionPreference='Stop';"
        "$m=@{};Get-PhysicalDisk|%{$m[[int]$_.DeviceId]=$_.MediaType};"
        "$n=@{};Get-PhysicalDisk|%{$n[[int]$_.DeviceId]=$_.FriendlyName};"
        "$s=@{};try{Get-CimInstance Win32_PageFileSetting|%{"
        "$v=[int]$_.MaximumSize;if(-not $v){$v=[int]$_.InitialSize};"
        "$s[$_.Name.Substring(0,1).ToUpper()]=$v}}catch{};"
        "Get-CimInstance Win32_PageFileUsage|%{"
        "$p=Get-Partition -DriveLetter $_.Name.Substring(0,1);"
        "$c=$s[$_.Name.Substring(0,1).ToUpper()];if(-not $c){$c=0};"
        "'{0}|{1}|{2}|{3}|{4}|{5}' -f $_.Name,$m[[int]$p.DiskNumber],"
        "$n[[int]$p.DiskNumber],$_.AllocatedBaseSize,$_.PeakUsage,$c}"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []

    found = []
    for line in (result.stdout or "").splitlines():
        parts = line.strip().split("|")
        if len(parts) < 5 or not parts[0]:
            continue
        try:
            allocated, peak = int(parts[3]), int(parts[4])
        except ValueError:
            allocated, peak = 0, 0
        try:
            configured = int(parts[5]) if len(parts) > 5 else 0
        except ValueError:
            configured = 0
        found.append({
            "path": parts[0], "drive": parts[0][:1].upper(),
            "media": parts[1] or "Unspecified", "disk": parts[2],
            "allocated_mb": allocated, "peak_used_mb": peak,
            "configured_mb": configured,
        })
    _PAGEFILES_CACHE.append(found)
    return found


def pagefile_reboot_pending():
    """
    One message if a resized pagefile has not been applied yet, else None.

    Windows does not resize a pagefile when you press OK; it writes the new
    size to the registry and grows the file at the next boot. Until that
    reboot the commit limit is still the old file's, and every refusal that
    comes out of the commit check is a refusal the user believes they have
    already fixed. Measured here on Sep 10: the setting says 72 GB, the live
    file is 32 GB, and the machine had been up for five days - so the commit
    limit was 63.8 GB when the user's mental model said 104 GB.

    pagefiles() is cached for the life of the process, which is right for this:
    the answer cannot change without a reboot, and a reboot ends the process.
    """
    found = pagefiles()
    pending = [pf for pf in found
               if pf.get("configured_mb")
               and abs(pf["configured_mb"] - pf["allocated_mb"]) > 512]
    if not pending:
        return None

    where = "; ".join(
        f"{pf['path']} is set to {pf['configured_mb'] / 1024:.0f} GB but the live file "
        f"is {pf['allocated_mb'] / 1024:.0f} GB"
        for pf in pending
    )
    # The limit after the reboot is physical RAM plus every pagefile at the
    # size it is configured for (a pagefile with no configured size keeps the
    # one it has).
    total_ram, _, _ = memory_status()
    configured_total_mb = sum(pf.get("configured_mb") or pf.get("allocated_mb", 0)
                              for pf in found)
    new_limit = total_ram + configured_total_mb * MB
    return (
        f"The pagefile {where} - Windows applies the new size at the next reboot. "
        f"Rebooting raises the commit limit from {fmt_gb(commit_limit_bytes())} to "
        f"about {new_limit / GB:.0f} GB, which is what decides whether a run is "
        f"refused for committed memory."
    )


def slow_pagefile_warning():
    """
    One warning string if any pagefile lives on a mechanical disk, else None.

    This is the difference between a run that finishes and one that locks the
    desktop for fifteen seconds and then gets killed by the watchdog: paging to
    NVMe costs microseconds, paging to a 7200 rpm disk costs a seek, and
    Windows will not prefer the fast one.
    """
    found = pagefiles()
    spinning = [pf for pf in found if pf["media"] == "HDD"]
    if not spinning:
        return None

    where = "; ".join(
        f"{pf['path']} on {pf['disk']} ({pf['allocated_mb'] / 1024:.0f} GB"
        + (f", {pf['peak_used_mb'] / 1024:.1f} GB used at peak" if pf["peak_used_mb"] else "")
        + ")"
        for pf in spinning
    )
    fast = [pf["path"] for pf in found if pf["media"] and pf["media"] != "HDD"]
    return (
        f"Windows is paging to a mechanical hard disk: {where}.\n"
        f"    A VR run commits more memory than this machine has RAM, so Windows pages "
        f"no matter what - and it spreads paging across every pagefile, so these ones "
        f"set the pace for all of it. That is what a freeze during a run actually is: "
        f"not a shortage of RAM, but the machine waiting on disk seeks to get pages back.\n"
        f"    Fix (System > About > Advanced system settings > Performance > Advanced > "
        f"Virtual memory): set the drives above to 'No paging file' and leave the pagefile "
        + (f"on {', '.join(fast)} only." if fast else "on an SSD only.")
    )


def fmt_gb(num_bytes):
    return f"{num_bytes / GB:.1f} GB"


def fmt_duration(seconds):
    """
    A wall-clock span written the way someone waiting on it would say it.

    Runs here range from twenty seconds to three days, so one format cannot
    serve the whole range: below a minute the seconds are the story, above an
    hour they are noise.
    """
    seconds = max(0.0, float(seconds or 0))
    if seconds < 1:
        return "0s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    hours, rest = divmod(int(seconds), 3600)
    if hours < 48:
        return f"{hours}h {rest // 60:02d}m"
    return f"{hours // 24}d {hours % 24}h"


def fmt_clock(seconds_from_now):
    """Time of day a run finishing in `seconds_from_now` would land on."""
    end = time.localtime(time.time() + max(0.0, float(seconds_from_now or 0)))
    now = time.localtime()
    stamp = time.strftime("%H:%M", end)
    days = (end.tm_yday - now.tm_yday) % 366
    if days == 1:
        return stamp + " tomorrow"
    if days > 1:
        return time.strftime("%a %H:%M", end)
    return stamp


# ---------------------------------------------------------------------------
# Guard configuration
# ---------------------------------------------------------------------------

def derive_job_cap_gb(commit_limit, total_ram):
    """
    The job's committed-memory cap for a machine with this commit limit, in GB.

    Split out from default_guard_config() so load_guard_config() can ask the
    second question that matters: is the cap sitting in guardrails.json still
    the one this machine deserves? A stored cap that matches what this returns
    for the commit limit stored beside it was derived rather than chosen, and
    can be re-derived freely when the machine changes.
    """
    job_commit_cap = min(commit_limit * 0.5, max(commit_limit - 16 * GB, 4 * GB))
    # Never below physical RAM: a cap under that is tighter than the machine.
    job_commit_cap = int(max(job_commit_cap, min(total_ram, commit_limit)))
    return round(job_commit_cap / GB, 1)


def default_guard_config():
    """Derive sane limits from the machine this is actually running on."""
    total_ram, _, _ = memory_status()
    total_ram = total_ram or (16 * GB)
    cores = os.cpu_count() or 8
    vram_total_mb, _ = gpu_memory_mb()

    # A Job Object's JobMemoryLimit caps COMMITTED memory, not resident memory,
    # and those are nowhere near the same number for a CUDA process: FaceFusion
    # measured 13.1 GB committed against a 2.3 GB working set, because the CUDA
    # runtime and the WDDM driver reserve address space they never make
    # resident. Sizing this cap against physical RAM (it used to be 62% of it,
    # 19.7 GB here) therefore killed runs that were not touching physical RAM
    # at all - a 4K Normal-mode run on the "Best" preset peaks at ~19.5 GB of
    # commit, i.e. 99% of that old cap, and anything longer went over and died
    # mid-run with an allocation failure.
    #
    # Size it against the commit limit instead - RAM plus pagefile, which is
    # what the limit is actually measured against - and keep half of that in
    # reserve for the rest of the system. Physical-memory pressure, the thing
    # that actually freezes Windows, is handled by the watchdog's free-RAM
    # trip point, not by this.
    commit_limit = commit_limit_bytes() or total_ram

    return {
        "enabled": True,

        # Hard ceiling on committed memory for ALL pipeline children combined.
        # If they try to exceed it their allocation fails and they die -
        # instead of the machine dying.
        "job_memory_cap_gb": derive_job_cap_gb(commit_limit, total_ram),

        # Hard CPU ceiling for the job, percent of total machine CPU.
        # Keeps the shell and UI schedulable during long x264/x265 encodes.
        "cpu_percent_cap": 75,

        # The same ceiling for a run nobody is sitting in front of - a queued
        # job overnight. 75% is there to keep the desktop responsive; at 03:00
        # there is no desktop to keep responsive, and a split, a merge or a
        # second lane running alongside a swap wants the headroom. After the
        # ORT thread-pool fix a single swap only draws ~20% of the machine, so
        # this rarely binds today; it is the cap that will matter once stages
        # overlap.
        "cpu_percent_cap_unattended": 90,

        # Children run below normal so nothing they do can starve the desktop.
        "below_normal_priority": True,

        # Watchdog trip points. Three consecutive strikes aborts the run.
        "watchdog_enabled": True,
        "watchdog_interval_sec": 3.0,
        "watchdog_strikes": 3,
        "min_free_ram_gb": 2.5,
        "min_free_disk_gb": 12.0,
        "min_free_pagefile_gb": 3.0,

        # Before any of those is reached, the run is FROZEN rather than killed.
        # Suspending every process in the job and handing its resident pages
        # back gives the machine several GB in a second or two, and costs the
        # run nothing but time - where the abort costs it the stage in flight,
        # which at 8K is the better part of an hour. This is the point where
        # the watchdog starts trying that; if free RAM falls past the abort
        # line between two samples - which is what happens when the machine
        # hits its commit ceiling - the freeze is still tried first.
        #
        # Free RAM only, deliberately. Freezing returns *resident* pages; it
        # does not return committed address space, so it cannot fix a commit
        # exhaustion and pretending otherwise would only spend the hold timer
        # before aborting anyway.
        "auto_pause_enabled": True,
        "auto_pause_free_ram_gb": 5.0,
        # Hysteresis: resume well clear of the pause point, or a run that is
        # itself the pressure oscillates between frozen and thawed.
        "auto_resume_free_ram_gb": 7.5,
        # If the memory never comes back, the run was not going to finish.
        "auto_pause_max_sec": 1800,

        # The same idea one level finer, for a run with more than one lane in
        # flight: a background split under a swap, or both eyes at once. These
        # trip points sit ABOVE the whole-job ones on purpose, so that the first
        # thing tried when the machine tightens is holding the newest lane -
        # which costs that lane its wall time and costs the run nothing, because
        # the other lane still has the GPU. Only if that is not enough does the
        # whole-job freeze below it fire, and only then the abort.
        #
        # VRAM is in here and is not in the whole-job set, because VRAM is the
        # resource two FaceFusion lanes actually run out of: one 4K eye holds
        # ~7 GB of a 16.3 GB card, and when WDDM has to page GPU memory into
        # host RAM the machine stops answering for tens of seconds. Freezing a
        # lane does not hand VRAM back by itself - only exiting does - but it
        # stops that lane allocating more while the other one finishes, which
        # is what keeps the card off the paging line.
        "lane_pause_free_ram_gb": 6.5,
        "lane_pause_free_commit_gb": 6.0,
        "lane_pause_free_vram_mb": 1200,
        # Hysteresis again: come back this far above the trip point, for three
        # consecutive samples, before thawing.
        "lane_resume_margin_gb": 2.0,
        # A lane held this long is not going to get its memory back. It is
        # killed - just that lane - and the caller re-runs it sequentially. The
        # run is never aborted for one lane's problem.
        "lane_pause_max_sec": 1800,

        # Run a chunk's split and the previous chunk's merge underneath the
        # swap instead of before and after it. Both are NVENC/NVDEC work with
        # little CPU, and after the onnxruntime thread-pool fix the swap leaves
        # ~80% of the machine and most of the encoder idle. Measured shares per
        # 60 s chunk at 8192x4096 90 fps: split ~3.6 min, both swaps ~72 min,
        # merge ~8 min - so hiding the split and the merge is ~14% of the chunk.
        # It costs one extra chunk of split intermediates on disk (~1.2 GB at
        # 4K60), which estimate_disk_need accounts for.
        "overlap_stages": True,

        # Both eyes swapped at once, in two FaceFusion processes. "auto" asks
        # parallel_eyes_verdict() whether the measured VRAM, commit and RAM
        # headroom can hold two lanes and runs sequentially when it cannot;
        # "off" never tries; "on" skips the commit and RAM margins but still
        # refuses when the VRAM cannot fit, because that failure is not a slow
        # run - it is WDDM paging GPU memory into host RAM and a machine that
        # stops answering. On this card at 4K with a face enhancer the verdict
        # is no (7 GB per lane x 2 + margin against 16.3 GB); it is yes with the
        # enhancer off or a smaller frame.
        "parallel_eyes": "auto",

        # Preflight refuses to start a run that cannot plausibly fit.
        "preflight_enabled": True,
        "disk_reserve_gb": 15.0,

        # A queued job refused for committed memory or free RAM is held rather
        # than failed: those are facts about what the machine is doing this
        # minute, not about the run. The job on Sep 10 01:16 was refused
        # because Ollama still had 8 GB of commit charged, which it gives back
        # by itself - and the queue failed the job and moved on, so the night
        # was spent idle. Retry every few minutes, and only call it a failure
        # once the memory has stayed gone for hours.
        "queue_retry_minutes": 5,
        "queue_retry_hours": 6,

        # Safe processing caps. The old ceiling of 4 was fitted while every
        # onnxruntime session was still spinning a thread pool per core per
        # model (see scripts/patch_facefusion.py): more execution threads only
        # bought more spinning, so they bought nothing. With the pools bounded,
        # 3 -> 6 threads took a 183-frame 4096x4096 pass from 98.8 s to 89.2 s
        # and cost 0.9 GB of VRAM and 1.1 GB of commit, so the cap is 8 now and
        # the estimates below decide what actually fits.
        "max_execution_threads": max(2, min(8, cores // 3)),

        # Intra-op threads for every onnxruntime session in the FaceFusion
        # child, passed as VRDEEP_ORT_INTRA_THREADS by engine._cuda_child_env
        # and read by the inference_manager patch. FaceFusion's default is
        # onnxruntime's, which is one thread per core per model and busy-spins
        # between GPU ops: 82% of this whole machine for one pass, of which 62
        # points were spin-waiting. Two threads with spinning off measured 20%
        # at the same frame rate.
        "ort_intra_op_threads": 2,
        "video_memory_strategy": "moderate",
        "temp_frame_format": "jpeg",

        # Prefer the dedicated NVENC chip over all-core software encoding.
        "prefer_nvenc": True,

        # Ceiling for every NVENC encode in the pipeline, in Mbit/s. This is
        # not really a size limit - under '-rc vbr' NVENC applies its own, much
        # lower, default maximum when none is given, and that default is what
        # ends up deciding quality instead of -cq. See engine._encoder_args.
        # 120 Mbit/s is generous for 4K per eye and keeps the disk estimate
        # (INTERMEDIATE_BPP, ~179 Mbit/s at 4K30) conservative.
        "nvenc_maxrate_mbps": 120,

        # Real-ESRGAN tile size. 0 means whole-frame inference, which blows
        # up VRAM on anything above 1080p. Never use 0.
        "esrgan_tile": 512,

        # Decode threads for FaceFusion's raw-frame ffmpeg readers. Must match
        # VRDEEP_FFMPEG_READER_THREADS in scripts/patch_facefusion.py - it is
        # only here so estimate_resident_mb() knows what it is sizing.
        "ffmpeg_reader_threads": 4,

        # Above this many megapixels per frame, a single in-flight frame is
        # large enough that thread count starts to decide whether the run fits
        # in RAM. One eye of an 8K VR clip is 16.8. See clamp_config().
        "large_frame_mpix": 8.0,

        "_machine": {
            "total_ram_gb": round(total_ram / GB, 1),
            "logical_cores": cores,
            "vram_total_mb": vram_total_mb,
            # The number job_memory_cap_gb above was derived from. Recorded so
            # load_guard_config() can tell a cap that still fits this machine
            # from one that was sized against a machine that no longer exists.
            "commit_limit_gb": round(commit_limit / GB, 1),
        },
    }


def load_guard_config(path=None):
    """
    Load guardrails.json, writing defaults on first run.

    job_memory_cap_gb is re-checked against the machine on every load rather
    than trusted, because it is derived from the commit limit and the commit
    limit moves - resizing the pagefile is enough to change it. This machine's
    config was written against a 102.1 GB commit limit and stored a 51 GB cap;
    the pagefile is now a fixed 32 GB, the commit limit is 63.8 GB, and roughly
    18 GB of that is already spoken for by the rest of Windows. A 51 GB cap on
    those numbers can never be reached.

    That is not a loose guard rail, it is no guard rail at all. The run dies on
    the *system* commit limit instead, and that arrives as a bare allocation
    failure from whichever thread happened to ask for memory next - the 48 MB
    numpy frame in face_helper.paste_back, say - with the job sitting at a
    third of its cap and nothing anywhere to say what actually ran out.

    So a cap that is simply what the stored commit limit derives gets
    re-derived for the machine as it is now, in either direction - enlarging
    the pagefile should hand the headroom back rather than leave the cap pinned
    at the smallest the machine ever was. A cap somebody chose by hand is only
    ever clamped DOWN, to what this machine can actually back.
    """
    path = path or GUARD_CONFIG_PATH
    config = default_guard_config()
    derived_cap = float(config.get("job_memory_cap_gb", 0) or 0)
    derived_machine = dict(config.get("_machine", {}))

    if os.path.exists(path):
        try:
            with open(path) as handle:
                config.update(json.load(handle))
        except Exception:
            pass

        stored_cap = float(config.get("job_memory_cap_gb", 0) or 0)
        stored_machine = config.get("_machine") or {}
        stored_limit_gb = float(stored_machine.get("commit_limit_gb", 0) or 0)
        total_ram = (memory_status()[0] or 0) or int(16 * GB)

        # A stored cap that is exactly what the stored commit limit derives is
        # a computed value nobody chose, so it can move in either direction
        # with the machine - which matters, because enlarging the pagefile
        # should give the cap its headroom back rather than leaving it pinned
        # at whatever the smallest pagefile the machine ever had allowed.
        was_derived = bool(
            stored_limit_gb
            and abs(stored_cap - derive_job_cap_gb(stored_limit_gb * GB, total_ram)) <= 0.2
        )

        if derived_cap and stored_cap and (was_derived or stored_cap > derived_cap):
            if abs(stored_cap - derived_cap) > 0.2:
                config["job_memory_cap_gb"] = derived_cap
                # Read by ResourceGuard.start(), which is where a human sees it.
                config["_cap_clamped_from_gb"] = stored_cap

        # The stored _machine block describes the machine as it was when the
        # file was written. Replace it with the live one so the drift is
        # visible in the log rather than preserved as fact.
        config["_machine"] = derived_machine
    else:
        try:
            with open(path, "w") as handle:
                json.dump(config, handle, indent=2)
        except Exception:
            pass
    return config


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

# Rough bytes-per-pixel for one stored frame, used only for sizing estimates.
BYTES_PER_PIXEL = {
    "jpeg": 0.10,
    "jpg": 0.10,
    "bmp": 3.0,
    "png": 1.30,
    "tiff": 2.4,
}

# Rough bytes-per-pixel-per-frame for the near-lossless intermediate encodes.
INTERMEDIATE_BPP = 0.09


def probe_media(path):
    """Width/height/fps/duration for `path`. Falls back to 1080p30/60s."""
    fallback = {"width": 1920, "height": 1080, "fps": 30.0, "duration": 60.0}
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate:format=duration",
             "-of", "json", path],
            capture_output=True, text=True, timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0 or not result.stdout:
            return fallback

        data = json.loads(result.stdout)
        stream = data["streams"][0]
        num, den = stream.get("r_frame_rate", "30/1").split("/")
        fps = float(num) / float(den) if float(den) else 30.0
        duration = float(data.get("format", {}).get("duration") or 0) or fallback["duration"]
        return {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": fps or 30.0,
            "duration": duration,
        }
    except Exception:
        return fallback


def estimate_disk_need(target_video, is_vr, guard_config, trim_start=None, trim_end=None,
                       pipeline_config=None, overlap=False):
    """
    Estimate PEAK temp-disk usage for a run - the most that is on disk at once,
    not the sum of everything the run ever writes.

    Two things this used to get badly wrong, both found by projecting a real
    50-minute 8K job and comparing against measured file sizes:

      1. It charged for FaceFusion extracting every frame to disk. That is the
         `disk` workflow; run_facefusion() pins `--workflow-strategy memory`,
         which decodes and pipes straight to the encoder and writes no frames
         at all. On a 50-minute 8K clip this alone invented ~2.2 TB.
      2. It summed every intermediate as though they all coexisted, and sized
         them by bits-per-pixel. But the intermediates are deleted as soon as
         the stage that consumes them finishes, and their size is set by the
         NVENC rate cap (`nvenc_maxrate_mbps`), not by resolution - which is
         why a 4096x4096 eye and an 8192x4096 stereo frame come out at
         similar bitrates.

    Measured on the 8192x4096 90 fps source, per second of video:

        split, one eye     9.4 MB/s     (cq 16, 120 Mbit/s cap)
        swapped, one eye  12.0 MB/s     (FaceFusion's encode at quality 95)
        merged stereo     14.9 MB/s     (cq 3, and the cap is what binds)
    """
    probe = probe_media(target_video)
    duration = probe["duration"]
    if trim_start is not None or trim_end is not None:
        # A one-sided range is a real range: "from 10:00" is the tail of the
        # file, not the whole of it, and estimating the whole file for it would
        # block runs that comfortably fit.
        start = max(0.0, trim_start or 0.0)
        end = duration if trim_end is None else min(duration, trim_end)
        duration = max(0.0, end - start)

    frames = max(1.0, duration * probe["fps"])
    fmt = guard_config.get("temp_frame_format", "jpeg")
    bpp = BYTES_PER_PIXEL.get(fmt, BYTES_PER_PIXEL["jpeg"])
    config = pipeline_config or {}

    full_pixels = probe["width"] * probe["height"]
    # In VR each pass works on one eye, i.e. half the source area.
    pass_pixels = full_pixels / (2.0 if is_vr else 1.0)

    # An encode is bounded by bits-per-pixel on small frames and by the rate
    # cap on large ones. Both regimes matter: 1080p never reaches the cap,
    # 4K-per-eye is pinned to it.
    cap_bytes_per_sec = float(guard_config.get("nvenc_maxrate_mbps") or 120) * 1e6 / 8.0

    def encoded(pixels, cap_fraction):
        by_pixels = frames * pixels * INTERMEDIATE_BPP
        by_rate = duration * cap_bytes_per_sec * cap_fraction
        return min(by_pixels, by_rate)

    split_bytes = encoded(pass_pixels, 0.65)
    swapped_bytes = encoded(pass_pixels, 0.80)
    merged_bytes = encoded(full_pixels, 1.00) if is_vr else encoded(full_pixels, 0.80)

    # Frame interpolation writes another per-eye file. It holds the same
    # duration at the same rate cap, so it is about the size of the swapped eye
    # it replaces rather than N times it - the extra frames buy detail, not
    # bitrate. What it does cost is one more file alive at once.
    interpolate_multi = int(config.get("interpolate_multi") or 1)
    interp_bytes = swapped_bytes if interpolate_multi > 1 else 0.0

    # Only the disk workflow stages frames on disk, and then it holds the
    # extracted and the processed copies at the same time.
    if str(config.get("workflow_strategy", "memory")).lower() == "disk":
        frames_bytes = frames * pass_pixels * bpp * 2
    else:
        frames_bytes = 0.0

    upscale_bytes = 0.0
    if config.get("upscale_method"):
        target_width = config.get("upscale_target_width")
        if target_width:
            factor = (float(target_width) / probe["width"]) ** 2
        else:
            factor = float(config.get("upscale_scale", 2) or 1) ** 2
        # The rate cap does not scale with the frame, so an upscale really can
        # be several times the merged file.
        upscale_bytes = min(frames * full_pixels * factor * INTERMEDIATE_BPP,
                            duration * cap_bytes_per_sec * factor)

    if is_vr:
        # Walk the run and take the worst moment. Both eyes are split up front;
        # each split is deleted as soon as its own swap finishes, so the peak
        # lands on the merge, holding both swapped eyes plus the file being
        # written from them.
        peaks = [
            2 * split_bytes + swapped_bytes + frames_bytes,   # left eye swapping
            split_bytes + 2 * swapped_bytes + frames_bytes,   # right eye swapping
            merged_bytes + upscale_bytes,                     # upscaling
        ]
        if interp_bytes:
            peaks += [
                2 * swapped_bytes + interp_bytes,      # interpolating the left eye
                swapped_bytes + 2 * interp_bytes,      # interpolating the right eye
                2 * interp_bytes + merged_bytes,       # merging the interpolated eyes
            ]
        else:
            peaks.append(2 * swapped_bytes + merged_bytes)    # merging
        if overlap:
            # A chunked run with the stages overlapped (B9) holds one more
            # chunk's worth of splits than a sequential one: the next chunk is
            # being split while this one swaps, and the previous chunk's two
            # swapped eyes are still on disk waiting for their merge. That is
            # two chunks of split intermediates, one chunk of swapped eyes and
            # one merged chunk alive at the same moment - about 1.2 GB more at
            # 4K60, and the reason this is charged for rather than discovered.
            peaks.append(4 * split_bytes + 2 * swapped_bytes + merged_bytes)
    else:
        peaks = [
            split_bytes + merged_bytes + frames_bytes,
            merged_bytes + upscale_bytes,
        ]

    total = max(peaks)
    return {
        "bytes": int(total),
        "upscale_bytes": int(upscale_bytes),
        "split_bytes": int(split_bytes),
        "interp_bytes": int(interp_bytes),
        "swapped_bytes": int(swapped_bytes),
        "merged_bytes": int(merged_bytes),
        "frames": int(frames),
        "duration": duration,
        "width": probe["width"],
        "height": probe["height"],
        "fps": probe["fps"],
        "frame_format": fmt,
    }


def estimate_vram_mb(pipeline_config):
    """
    Rough resident-VRAM estimate for the selected model stack, in MB.

    Refitted against measurement rather than guesswork: a "Good" pass at
    4096x4096 with 3 execution threads holds ~7.0 GB of VRAM (8.4 GB in use
    system-wide against a 1.4 GB idle baseline), and 6 threads holds ~7.9 GB.
    The old version returned 5397 MB for that case - a third low - because it
    under-counted the models and then multiplied the whole stack by 1.35 per
    extra thread, which is not what a thread costs: a thread carries the
    in-flight frame and its intermediate tensors, not another copy of every
    model. It is a flat ~300 MB each here, so that is what it is charged.

    Used by the preflight and, later, by the both-eyes-at-once gate, where
    being a third low is the difference between two lanes fitting on the card
    and WDDM paging VRAM into host RAM - which is what freezes the machine.
    """
    total = 2500  # CUDA context, detector, landmarker, recognizer, maskers

    swapper = (pipeline_config.get("swapper_model") or "").lower()
    if "512" in swapper:
        total += 3800
    elif "256" in swapper:
        total += 2100
    else:
        total += 1600

    boost = str(pipeline_config.get("pixel_boost") or "512x512")
    try:
        side = int(boost.split("x")[0])
    except (ValueError, IndexError):
        side = 512
    total += int((side / 512.0) ** 2 * 700)

    enhancer = (pipeline_config.get("enhancer_model") or "").lower()
    if "2048" in enhancer:
        total += 6000
    elif "1024" in enhancer:
        total += 2800
    elif enhancer and enhancer != "none":
        total += 1600

    frame_enhancer = (pipeline_config.get("frame_enhancer") or "").lower()
    if frame_enhancer and frame_enhancer != "none":
        # Frame enhancers run on the full frame, not a 512px crop.
        total += 4200

    # Each extra in-flight frame carries its own intermediates, not its own
    # copy of the models: measured 3 -> 6 threads on a 4096x4096 eye added
    # 0.9 GB in total, i.e. 300 MB per thread.
    threads = int(pipeline_config.get("threads") or 1)
    total += 300 * max(0, threads - 1)
    return total


def estimate_commit_mb(pipeline_config, frame_pixels):
    """
    Rough peak COMMITTED host memory for one processing pass, in MB.

    This is the number the Job Object's memory cap is measured against, so it
    is the one that decides whether a run survives - and it is dominated by
    address space the CUDA runtime reserves rather than by anything resident.
    The constants are fitted to measurements on this machine (RTX 5060 Ti,
    peak job commit as reported by ResourceGuard.peak_memory_bytes, with
    CUDA_MODULE_LOADING=LAZY set by engine._cuda_child_env):

        frame            preset  threads  measured  this model
        3840x2160  8.3 MP  Good      3     11.1 GB    11.0 GB
        4096x4096 16.8 MP  Good      3     13.7 GB    13.6 GB
        4096x4096 16.8 MP  Good      6     14.9 GB    14.8 GB
        3840x2160  8.3 MP  Best      3     19.5 GB    19.5 GB

    The previous fit charged 100 MB per thread-megapixel and 130 MB per
    ffmpeg-megapixel, which put the 16.8 MP "Good" run at 19.7 GB against a
    measured 13.7 - a 6 GB over-estimate, and the reason the Sep 10 VR job was
    refused outright on a machine that could have run it. What the two frame
    sizes together actually say is that a thread costs ~22 MB per megapixel and
    each of FaceFusion's three ffmpeg children ~80 MB per megapixel plus a flat
    250; the constants below are that fit, with 5% added at the end so the
    number errs high rather than low.

    It is an estimate, not a measurement: treat a near miss as "warn", not as
    "refuse".
    """
    mpix = max(0.1, frame_pixels / 1e6)

    # Interpreter, onnxruntime arenas, CUDA context and driver reservations.
    total = 5000

    # Pixel boost tiles the crop, so the swapper's batch is (side/256)^2.
    boost = str(pipeline_config.get("pixel_boost") or "256x256")
    try:
        side = int(boost.split("x")[0])
    except (ValueError, IndexError):
        side = 256
    total += int((side / 256.0) ** 2 * 1200)

    # gpen_bfr_1024 is pinned by the "Best" anchor: once the per-thread and
    # ffmpeg terms below came down to what the two frame sizes actually say,
    # the whole 8.4 GB gap between measured "Good" (11.1 GB) and measured
    # "Best" (19.5 GB) at the same resolution has to sit in the fixed terms,
    # and the enhancer is where it is. The 2048 tier is extrapolated from that
    # ratio and has never been measured - no preset offers it.
    enhancer = (pipeline_config.get("enhancer_model") or "").lower()
    if "2048" in enhancer:
        total += 11000
    elif "1024" in enhancer:
        total += 5900
    elif "512" in enhancer:
        total += 1200
    elif enhancer and enhancer != "none":
        total += 900

    # Frame enhancers work on the whole frame, so they scale with resolution.
    frame_enhancer = (pipeline_config.get("frame_enhancer") or "").lower()
    if frame_enhancer and frame_enhancer != "none":
        total += int(mpix * 900)

    # Each in-flight frame carries its own buffers through the processor chain.
    # 22 MB per megapixel per thread: measured 3 -> 6 threads at 16.8 MP cost
    # 1.1 GB of commit, which is 22 MB per thread-megapixel almost exactly.
    threads = int(pipeline_config.get("threads") or 1)
    total += int(threads * mpix * 22)

    # FaceFusion's own ffmpeg children: two readers and a writer, all inside
    # the job and all holding pools of raw frames (48 MB each at 4096x4096).
    # The slope comes from the pair of measured frame sizes: 8.3 MP and 16.8 MP
    # at the same thread count differ by 2.6 GB, which is 80 MB per megapixel
    # across the three of them, not the 130 the old fit assumed.
    total += int(3 * (mpix * 80 + 250))

    # The whole thing is a model of a run that has not started yet, and the
    # cost of guessing low (an allocation failure mid-run, an hour of work
    # lost) is far higher than the cost of guessing high (a warning). 5%.
    return int(total * 1.05)


def estimate_resident_mb(pipeline_config, frame_pixels, guard_config=None):
    """
    Rough peak RESIDENT host memory for one processing pass, in MB.

    Not the same question as estimate_commit_mb(). That one sizes the run
    against the Job Object's committed-memory cap, which is half the machine's
    commit limit - 51 GB here, on a box with 32 GB of RAM. This one sizes it
    against physical RAM that is actually free, which is what the watchdog
    aborts on and what makes Windows unusable when it runs out.

    Fitted against measured left-eye passes on this machine: 4096x4096 per eye
    (an 8192x4096 VR source), hyperswap_1a_256 at pixel boost 256 with
    gpen_bfr_512, FaceFusion's in-memory workflow, decode threads capped at 4.
    Peak resident for the whole process tree, by execution thread count:

        threads  1      6.61 GB    150 s
        threads  2      6.81 GB    101 s
        threads  3      6.99 GB     76 s
        threads  4      7.09 GB     73 s

    Two things in that table matter. First, the ffmpeg children dominate and
    are flat: ~1.45 GB each for the two readers and the writer, 4.4 GB of the
    total, because a raw frame at 4096x4096 is 48 MB and each process holds a
    pool of them. `reader_threads` here has to match VRDEEP_FFMPEG_READER_THREADS
    in scripts/patch_facefusion.py or this estimate describes a pipeline that is
    no longer running.

    Second, execution threads are worth about 160 MB each - not the ~350 MB an
    earlier version of this function assumed. Halving them saves a quarter of a
    gigabyte and costs 40% of the run's speed, which is why clamp_config() now
    treats thread reduction as a last resort rather than a first move.
    """
    guard_config = guard_config or {}
    mpix = max(0.1, frame_pixels / 1e6)
    raw_frame_mb = mpix * 3

    # Interpreter, onnxruntime arenas, host-side copies of the model weights.
    total = 1800

    # The swapper stages one tile per pixel-boost step.
    boost = str(pipeline_config.get("pixel_boost") or "256x256")
    try:
        side = int(boost.split("x")[0])
    except (ValueError, IndexError):
        side = 256
    total += int((side / 256.0) ** 2 * 300)

    enhancer = (pipeline_config.get("enhancer_model") or "").lower()
    if "2048" in enhancer:
        total += 1500
    elif "1024" in enhancer:
        total += 700
    elif enhancer and enhancer != "none":
        total += 300

    # Frame enhancers run on the whole frame rather than a face crop.
    frame_enhancer = (pipeline_config.get("frame_enhancer") or "").lower()
    if frame_enhancer and frame_enhancer != "none":
        total += int(mpix * 180)

    # In-flight frames. Measured at ~160 MB per execution thread on a 50 MB
    # raw frame, i.e. a little over three frames' worth once the frame has been
    # through the processor chain - not the seven the first version guessed.
    threads = int(pipeline_config.get("threads") or 1)
    total += int(threads * raw_frame_mb * 3.2)

    # FaceFusion's own ffmpeg children: two raw-frame readers and one writer,
    # measured at ~1.45 GB each on a 50 MB frame with decode threads at 4.
    # That is around 28 frames' worth per process, so the pool grows with the
    # thread cap but is mostly fixed.
    reader_threads = int(guard_config.get("ffmpeg_reader_threads", 4))
    buffered_frames = reader_threads * 4 + 12
    total += int(3 * (buffered_frames * raw_frame_mb + 150))

    return total


# ---------------------------------------------------------------------------
# Runtime estimation
# ---------------------------------------------------------------------------
#
# A VR run is measured in hours, and until now nothing said so before it
# started. The point of this section is that the log can state a number up
# front, and that the live clock in engine.py has something to calibrate
# against once real frames start going through.
#
# Every stage is modelled as
#
#     seconds per frame = fixed + per_megapixel * frame_megapixels
#
# where `fixed` is work that does not care how big the frame is (the per-face
# model passes, mostly) and `per_megapixel` is work proportional to frame area
# (decode, crop, resize, encode). The numbers are fitted to the measurements
# in PROGRESS.md, all taken on this machine against the 8192x4096 90 fps
# source, for which one eye is 16.78 MP:
#
#     swap          2.5 frame/s per eye        ->  0.400 s/frame at 16.78 MP
#     split          50 frame/s per eye        ->  0.020 s/frame at 16.78 MP
#     merge          11 frame/s stereo         ->  0.091 s/frame at 33.55 MP
#     smoothing     ~1 frame/s at 3072x3072    ->  1.000 s/frame at  9.44 MP
#
# One anchor cannot separate the two terms, so how each stage's cost is split
# between them is a judgement rather than a measurement: the swap's per-face
# work (a 640x640 detector, a 256 swapper, a 512 enhancer) does not grow with
# the frame, so a quarter of its cost is charged as fixed; the ffmpeg stages
# are almost entirely area and are charged almost entirely per megapixel.
# Getting that split wrong only matters at resolutions far from the anchor,
# and RunClock recalibrates from the run's own first minutes regardless.
STAGE_COST = {
    "split": (0.002, 0.001073),
    "swap": (0.100, 0.017878),
    "merge": (0.005, 0.002561),
    "smooth": (0.010, 0.104900),
    "upscale_ffmpeg": (0.005, 0.002561),
    # Real-ESRGAN is a full-frame generative pass at tile 512 and has never
    # been timed on this machine at VR resolution - this is an order of
    # magnitude, not a measurement, and the caller says so. It is charged
    # against the INPUT frame: the network runs at input resolution and only
    # upsamples in its last layers, so a 4x pass is not four times a 2x one.
    "upscale_realesrgan": (0.020, 0.200000),
}

# RIFE is modelled separately because its cost has two different frame sizes in
# it: the model runs at `infer_scale` of the frame, while the resizes, the
# 50 MB-per-frame pipes and the NVENC encode all run at full size. Fitted to
# the resolution table in PROGRESS.md (6.1 / 3.0 / 1.6 frame/s at 2048 / 3072 /
# 4096 square) plus the 0.96 frame/s measured end to end at scale 0.75.
RIFE_FIXED = 0.011
RIFE_PER_MODEL_MP = 0.0366
RIFE_PER_FRAME_MP = 0.0409

# A frame enhancer runs a generative model over the WHOLE frame, once per
# frame, on top of everything else. Rough, and only ever charged when one is
# actually selected.
FRAME_ENHANCER_PER_MP = 0.25

TIMING_PATH = os.path.join(os.path.dirname(SCRIPTS_DIR), "output", "timing.json")

# How far a learned factor is allowed to move the model. A run that was
# starved by other programs should not poison the next estimate.
CALIBRATION_BOUNDS = (0.25, 4.0)


def parallel_eyes_lane_threads(pipeline_config):
    """Execution threads for ONE lane when both eyes run at once.

    Half the configured count, never below two. The two processes share one
    GPU, so six threads each would only queue more work behind the same
    kernels - and the pair of 3-thread lanes is the shape that was measured.
    """
    threads = int((pipeline_config or {}).get("threads") or 3)
    return max(2, threads // 2)


def parallel_eyes_verdict(pipeline_config, guard_config, pass_pixels):
    """
    Whether this machine can hold two FaceFusion lanes at once. -> (ok, reason)

    Sequential eyes are the safe answer and stay the default whenever this is
    not clearly yes. What is being weighed is not speed against speed: it is
    ~25-35% off the swap stage against the machine freezing. Two lanes that do
    not fit in VRAM do not run slowly, they make WDDM page GPU memory into host
    RAM, and that is the failure that locks the desktop for minutes at a time
    (PROGRESS.md, "cuDNN algorithm search").

    Measured on this card (RTX 5060 Ti, 16.3 GB): one 4096x4096 "Good" pass
    holds ~7.0 GB of VRAM with 3 threads and ~7.9 GB with 6. Two of those plus
    the 1.4 GB desktop baseline is 15.4-17.2 GB of 16.3, so at 4K with a face
    enhancer the answer here is no and is meant to be. It is yes with the
    enhancer off (~5 GB a lane) or on a smaller frame.

    The `parallel_eyes` guard key selects the policy:
      "off"  - never; sequential, no questions.
      "auto" - VRAM, commit and RAM must all have room for two lanes.
      "on"   - the commit and RAM margins are skipped (the user has decided
               they would rather page than wait), but VRAM is still checked.
               Overcommitting VRAM is not a slow run, it is a frozen machine.
    """
    config = pipeline_config or {}
    guard_config = guard_config or {}
    mode = str(guard_config.get("parallel_eyes", "auto") or "auto").lower()
    if mode not in ("auto", "on", "off"):
        mode = "auto"
    if mode == "off":
        return False, "turned off in guardrails.json (parallel_eyes: off)"

    # Each lane runs half the threads, so each lane is smaller than the
    # sequential pass the rest of the preflight sized.
    lane_config = dict(config, threads=parallel_eyes_lane_threads(config))
    lane_vram_mb = estimate_vram_mb(lane_config)
    lane_commit_mb = estimate_commit_mb(lane_config, pass_pixels)
    lane_resident_mb = estimate_resident_mb(lane_config, pass_pixels, guard_config)

    vram_total_mb, vram_used_mb = gpu_memory_mb()
    if vram_total_mb:
        free_vram_mb = vram_total_mb - vram_used_mb
        need_vram_mb = 2 * lane_vram_mb + 1500
        if free_vram_mb < need_vram_mb:
            return False, (
                f"VRAM {need_vram_mb / 1024.0:.1f} GB needed for two lanes "
                f"({lane_vram_mb / 1024.0:.1f} GB each plus 1.5 GB spare), "
                f"{vram_total_mb / 1024.0:.1f} GB on the card with "
                f"{vram_used_mb / 1024.0:.1f} GB in use")
    elif mode == "auto":
        return False, "the GPU's free VRAM could not be read, and guessing it is not safe"

    if mode == "on":
        return True, (f"forced on (parallel_eyes: on); VRAM fits at "
                      f"{lane_vram_mb / 1024.0:.1f} GB a lane, and the commit and RAM "
                      f"margins were not checked")

    _total_ram, avail_ram, avail_commit = memory_status()
    commit_floor = float(guard_config.get("min_free_pagefile_gb", 3.0)) * GB
    commit_headroom = max(0, avail_commit - commit_floor)
    if avail_commit and 2 * lane_commit_mb * 1024 * 1024 > commit_headroom:
        return False, (
            f"committed memory {2 * lane_commit_mb / 1024.0:.1f} GB needed for two "
            f"lanes, {fmt_gb(commit_headroom)} available")

    ram_floor = float(guard_config.get("min_free_ram_gb", 2.5)) * GB
    ram_headroom = max(0, avail_ram - ram_floor)
    if avail_ram and 2 * lane_resident_mb * 1024 * 1024 > ram_headroom:
        return False, (
            f"RAM {2 * lane_resident_mb / 1024.0:.1f} GB needed for two lanes, "
            f"{fmt_gb(ram_headroom)} usable")

    return True, (
        f"{lane_vram_mb / 1024.0:.1f} GB of VRAM, "
        f"{lane_commit_mb / 1024.0:.1f} GB committed and "
        f"{lane_resident_mb / 1024.0:.1f} GB of RAM per lane, at "
        f"{parallel_eyes_lane_threads(config)} threads each - two of those fit")


def format_parallel_eyes_verdict(ok, reason):
    """The one line the preflight logs about it, either way."""
    return f"[Guard] Both eyes at once: {'yes' if ok else 'no'} - {reason}."


def load_calibration():
    """Per-stage speed factors learned from previous runs on this machine."""
    try:
        with open(TIMING_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        stages = data.get("stages")
        return stages if isinstance(stages, dict) else {}
    except Exception:
        return {}


def record_stage_timing(kind, predicted_seconds, actual_seconds):
    """
    Fold one finished stage into the learned factors.

    Kept deliberately dull: an exponential average of actual/predicted per
    stage kind, bounded, and ignoring anything short enough that process
    startup dominates it. Refitting the cost model itself from history would
    need far more samples than a machine doing a handful of runs a week will
    ever produce.
    """
    if not predicted_seconds or predicted_seconds < 8 or actual_seconds < 8:
        return
    ratio = float(actual_seconds) / float(predicted_seconds)
    ratio = max(CALIBRATION_BOUNDS[0], min(CALIBRATION_BOUNDS[1], ratio))
    try:
        stages = load_calibration()
        entry = stages.get(kind) or {}
        previous = float(entry.get("factor") or 1.0)
        samples = int(entry.get("samples") or 0)
        # The first real sample is worth more than later ones; after that it
        # settles down and stops chasing a single slow run.
        weight = 0.6 if samples == 0 else 0.35
        stages[kind] = {
            "factor": round(previous * (1 - weight) + ratio * weight, 4),
            "samples": samples + 1,
            "last_ratio": round(ratio, 4),
        }
        os.makedirs(os.path.dirname(TIMING_PATH), exist_ok=True)
        with open(TIMING_PATH, "w", encoding="utf-8") as handle:
            json.dump({"stages": stages}, handle, indent=2)
    except Exception:
        # Timing history is a convenience. It must never break a run.
        pass


def _calibration_factor(kind, calibration):
    if not calibration:
        return 1.0
    entry = calibration.get(kind) or {}
    try:
        factor = float(entry.get("factor") or 1.0)
    except (TypeError, ValueError):
        return 1.0
    return max(CALIBRATION_BOUNDS[0], min(CALIBRATION_BOUNDS[1], factor))


def _stage_seconds(kind, frames, megapixels, calibration=None):
    fixed, per_mp = STAGE_COST[kind]
    return frames * (fixed + per_mp * megapixels) * _calibration_factor(kind, calibration)


def swap_face_factor(pipeline_config):
    """
    How much dearer this quality stack is per face than the measured one.

    The anchor was taken with the defaults - hyperswap at 256, gpen_bfr_512,
    no expression restorer. These are ratios of model work rather than
    timings; they exist so that switching to Pixel Boost 512 does not leave
    the estimate insisting the run is still three hours.
    """
    config = pipeline_config or {}
    factor = 1.0

    # Pixel boost tiles the crop, so 512 on a 256 model is a batch of 4 and
    # 1024 a batch of 16 - dearer, though not by the whole tile count.
    boost = str(config.get("pixel_boost") or "256x256").lower()
    factor *= {"256x256": 1.0, "512x512": 2.2, "1024x1024": 5.0}.get(boost, 1.0)

    enhancer = str(config.get("enhancer_model") or "").lower()
    if not enhancer or enhancer == "none":
        factor *= 0.55
    elif "1024" in enhancer:
        factor *= 1.9

    # A second generative model per face.
    if config.get("expression_restorer"):
        factor *= 1.45

    return factor


def rife_seconds_per_frame(megapixels, infer_scale):
    """Seconds to synthesise one frame at this size. See RIFE_* above."""
    scale = max(0.1, min(1.0, float(infer_scale or 0.75)))
    return (RIFE_FIXED
            + RIFE_PER_MODEL_MP * megapixels * scale * scale
            + RIFE_PER_FRAME_MP * megapixels)


# What two lanes cost each other on one GPU. A single pass leaves the card
# 25-30% idle, so two are not twice the work - but they are not free either:
# they share the SM pool, the copy engines and NVENC. Until it has been
# measured on this machine, assume the pair takes 15% longer than one eye alone
# and let the run's own pace correct it (RunClock rescales from measured wall
# time within the first few percent).
PARALLEL_EYES_PENALTY = 1.15


def estimate_runtime(estimate, pipeline_config, is_vr, calibration=None,
                     parallel_eyes=False, overlapped=()):
    """
    Wall-clock estimate for a run, stage by stage.

    Returns {"stages": [...], "total_seconds": float, "rough": [...]}, where
    each stage carries the key the pipeline uses for it, a human label, its
    predicted seconds and the cost model it came from. The list is in
    execution order, which is what lets RunClock lay the progress bar out
    proportionally to time instead of to a set of hardcoded percentages.

    `parallel_eyes` replaces the two per-eye swap stages with one `swap_both`
    stage, because that is what the run will actually do and an estimate that
    reads "left eye 36m, right eye 36m" for a 41-minute run is worse than no
    estimate. `overlapped` names stages that will run underneath another one
    (B9: a chunk's split and the previous chunk's merge). Those keep their own
    predicted seconds - the calibration still wants them - but are marked so
    RunClock gives them no share of the bar and leaves them out of the budget:
    time that is hidden under another stage is not time the user waits.
    """
    config = pipeline_config or {}
    calibration = load_calibration() if calibration is None else calibration

    frames = max(1, int(estimate.get("frames") or 1))
    full_mp = (estimate.get("width", 1920) * estimate.get("height", 1080)) / 1e6
    # In VR every stage before the merge works on one eye.
    pass_mp = full_mp / (2.0 if is_vr else 1.0)

    stages = []
    rough = []

    overlapped = set(overlapped or ())

    def add(key, label, seconds, kind):
        if seconds > 0:
            stage = {"key": key, "label": label,
                     "seconds": float(seconds), "kind": kind}
            if key in overlapped:
                stage["overlapped"] = True
            stages.append(stage)

    swap_fixed, swap_per_mp = STAGE_COST["swap"]
    swap_seconds = frames * (swap_fixed * swap_face_factor(config)
                             + swap_per_mp * pass_mp)
    if config.get("frame_enhancer"):
        swap_seconds += frames * FRAME_ENHANCER_PER_MP * pass_mp
        rough.append("a frame enhancer is on, and its cost here is an order of "
                     "magnitude rather than a measurement")
    swap_seconds *= _calibration_factor("swap", calibration)

    if is_vr:
        add("split", "Split into left and right eyes",
            2 * _stage_seconds("split", frames, pass_mp, calibration), "split")
        if parallel_eyes:
            # One stage, not two: the eyes run side by side, so what the user
            # waits for is the slower of them plus the GPU-sharing penalty.
            add("swap_both", "Face swap, both eyes together",
                swap_seconds * PARALLEL_EYES_PENALTY, "swap")
        else:
            add("swap_left", "Face swap, left eye", swap_seconds, "swap")
            add("swap_right", "Face swap, right eye", swap_seconds, "swap")

        multi = int(config.get("interpolate_multi") or 1)
        if multi > 1:
            per_eye = (frames * (multi - 1)
                       * rife_seconds_per_frame(pass_mp, config.get("interpolate_scale"))
                       * _calibration_factor("interp", calibration))
            add("interp_left", "Frame interpolation %dx, left eye" % multi, per_eye, "interp")
            add("interp_right", "Frame interpolation %dx, right eye" % multi, per_eye, "interp")
            # Everything downstream of RIFE handles multi times the frames.
            frames = int(frames * multi)

        add("merge", "Merge the eyes back into one stereo video",
            _stage_seconds("merge", frames, full_mp, calibration), "merge")
    else:
        add("swap", "Face swap", swap_seconds, "swap")

    if config.get("temporal_smoothing"):
        add("smooth", "Temporal smoothing",
            _stage_seconds("smooth", frames, full_mp, calibration), "smooth")

    method = str(config.get("upscale_method") or "").lower()
    if method in ("ffmpeg", "realesrgan"):
        target_width = config.get("upscale_target_width")
        if target_width:
            grow = (float(target_width) / max(1, estimate.get("width", 1920))) ** 2
        else:
            grow = float(config.get("upscale_scale", 2) or 1) ** 2
        kind = "upscale_" + method
        # Lanczos plus the encode is bounded by what comes out; Real-ESRGAN by
        # what goes in. See STAGE_COST.
        sized = full_mp * grow if method == "ffmpeg" else full_mp
        add("upscale", "Upscale (%s)" % method,
            _stage_seconds(kind, frames, sized, calibration), kind)
        if method == "realesrgan":
            rough.append("Real-ESRGAN has never been timed at this resolution on "
                         "this machine, so its share is a guess")

    # An overlapped stage is real work and is still recorded and calibrated,
    # but it happens underneath another stage, so counting it in the total
    # would promise a wait that nobody has.
    total = sum(stage["seconds"] for stage in stages if not stage.get("overlapped"))
    return {"stages": stages, "total_seconds": total, "rough": rough,
            "calibrated": any(calibration.values()) if calibration else False}


def preflight(target_video, temp_dir, pipeline_config, guard_config,
              is_vr=False, trim_start=None, trim_end=None, overlap=False):
    """
    Decide whether this run is safe to start.

    Returns (blocking_problems, warnings, estimate). A non-empty
    blocking_problems list means: do not start.

    blocking_problems is a list of (kind, text), kind being one of "disk",
    "vram", "cap", "commit" or "ram". The kind is what lets a caller tell a
    refusal that will read exactly the same way at 04:00 (disk, VRAM, the job
    cap - all properties of the run itself) from one that is only about what
    the machine happens to be holding this minute (commit, RAM). engine's
    PipelineError.transient is that distinction, and app defers rather than
    fails a queued job on it.
    """
    blocking = []
    warnings = []

    estimate = estimate_disk_need(target_video, is_vr, guard_config, trim_start, trim_end,
                                  pipeline_config=pipeline_config, overlap=overlap)

    # Whether both eyes can be swapped at once, decided here rather than at the
    # moment of spawning, so that the time estimate below describes the run
    # that is actually going to happen and the log says why either way. It is
    # re-checked immediately before the lanes start - VRAM is a property of the
    # whole machine and half an hour of split can pass in between.
    eye_pixels = estimate["width"] * estimate["height"] / (2.0 if is_vr else 1.0)
    parallel_ok, parallel_reason = (False, "not a VR run")
    if is_vr:
        parallel_ok, parallel_reason = parallel_eyes_verdict(
            pipeline_config, guard_config, eye_pixels)
    estimate["parallel_eyes"] = {
        "ok": bool(parallel_ok),
        "reason": parallel_reason,
        "line": format_parallel_eyes_verdict(parallel_ok, parallel_reason),
        "lane_threads": parallel_eyes_lane_threads(pipeline_config),
    }

    # Attached before the enabled check, because how long a run will take is
    # not a safety question and the answer is wanted either way.
    estimate["runtime"] = estimate_runtime(estimate, pipeline_config, is_vr,
                                           parallel_eyes=parallel_ok,
                                           overlapped=("split", "merge") if overlap else ())
    if not guard_config.get("preflight_enabled", True):
        return ([], ["Preflight checks are disabled in guardrails.json."], estimate)

    free_disk = free_disk_bytes(temp_dir)
    reserve = guard_config.get("disk_reserve_gb", 15.0) * GB
    needed = estimate["bytes"] + reserve

    estimate["free_disk"] = free_disk
    estimate["needed"] = needed

    if free_disk and needed > free_disk:
        blocking.append((
            "disk",
            f"Not enough temp disk space on {os.path.splitdrive(os.path.abspath(temp_dir))[0] or 'the temp drive'}. "
            f"This run needs about {fmt_gb(estimate['bytes'])} of temp files "
            f"(+{guard_config.get('disk_reserve_gb', 15.0):.0f} GB reserved for Windows) "
            f"but only {fmt_gb(free_disk)} is free.\n"
            f"    {estimate['frames']} frames at {estimate['width']}x{estimate['height']} "
            f"({estimate['duration']:.0f}s @ {estimate['fps']:.0f} fps).\n"
            f"    Fix: trim the clip, point the temp folder at a bigger drive, or free up space."
        ))
    elif free_disk and needed > free_disk * 0.75:
        warnings.append(
            f"Temp disk will be tight: ~{fmt_gb(estimate['bytes'])} needed, "
            f"{fmt_gb(free_disk)} free."
        )

    vram_total_mb, vram_used_mb = gpu_memory_mb()
    vram_need_mb = estimate_vram_mb(pipeline_config)
    estimate["vram_total_mb"] = vram_total_mb
    estimate["vram_need_mb"] = vram_need_mb

    if vram_total_mb:
        vram_free_mb = vram_total_mb - vram_used_mb
        if vram_need_mb > vram_total_mb * 0.92:
            blocking.append((
                "vram",
                f"The selected models need roughly {vram_need_mb} MB of VRAM but the GPU has "
                f"{vram_total_mb} MB. Overcommitting VRAM makes Windows page GPU memory into "
                f"system RAM, which is what freezes the machine.\n"
                f"    Fix: lower Pixel Boost, use a smaller face enhancer, turn the frame "
                f"enhancer off, or reduce Threads."
            ))
        elif vram_need_mb > vram_free_mb * 0.85:
            warnings.append(
                f"VRAM headroom is small: ~{vram_need_mb} MB needed, {vram_free_mb} MB free "
                f"of {vram_total_mb} MB."
            )

    # Host commit vs the Job Object cap. This is the limit that actually stops
    # long runs: when the job's committed memory hits the cap, every further
    # allocation inside it fails and FaceFusion dies mid-run, typically an hour
    # in. Catch it here instead.
    # One pass, deliberately: every check from here down sizes the run as it is
    # written today, one eye at a time. Two lanes are only ever started when
    # parallel_eyes_verdict above has already found room for both of them on
    # top of this, so the sequential figures stay the floor.
    pass_pixels = eye_pixels
    commit_need_mb = estimate_commit_mb(pipeline_config, pass_pixels)
    cap_gb = float(guard_config.get("job_memory_cap_gb", 0) or 0)
    estimate["commit_need_mb"] = commit_need_mb
    estimate["commit_cap_mb"] = int(cap_gb * 1024)

    if cap_gb > 0:
        cap_mb = cap_gb * 1024
        if commit_need_mb > cap_mb * 0.95:
            blocking.append((
                "cap",
                f"These settings need roughly {commit_need_mb / 1024:.1f} GB of committed memory "
                f"but the guard rails cap the whole pipeline at {cap_gb:.1f} GB. The run would "
                f"start normally and then die part-way through with an allocation failure.\n"
                f"    Note this is *committed* memory, not RAM in use - a CUDA process reserves "
                f"far more than it makes resident.\n"
                f"    Fix: raise job_memory_cap_gb in scripts/guardrails.json (this machine's "
                f"commit limit is {fmt_gb(commit_limit_bytes())}), pick a lighter quality preset, "
                f"or reduce Threads."
            ))
        elif commit_need_mb > cap_mb * 0.8:
            warnings.append(
                f"Memory headroom is small: ~{commit_need_mb / 1024:.1f} GB of committed memory "
                f"needed against a {cap_gb:.1f} GB cap."
            )

    # Available system commit. The cap above governs the pipeline; this governs
    # the machine. They are different ceilings and on this box they are close
    # together - a 31.9 GB cap against a 63.8 GB commit limit that already has
    # ~18 GB of Windows in it - so which one binds depends on what else is
    # running, and it is usually this one. It has to be checked here because
    # there is no second chance: the job cap fails loudly through the guard,
    # while running out of system commit just fails whichever allocation came
    # next, in whichever process, with no way back to the cause.
    _, _, avail_commit = memory_status()
    commit_floor = float(guard_config.get("min_free_pagefile_gb", 3.0)) * GB
    commit_headroom = max(0, avail_commit - commit_floor)
    estimate["avail_commit"] = avail_commit

    if avail_commit:
        if commit_need_mb * 1024 * 1024 > commit_headroom:
            message = (
                f"This run needs roughly {commit_need_mb / 1024:.1f} GB of committed memory "
                f"but only {fmt_gb(commit_headroom)} is available on the machine right now "
                f"({fmt_gb(avail_commit)} free of a {fmt_gb(commit_limit_bytes())} commit "
                f"limit, minus the {float(guard_config.get('min_free_pagefile_gb', 3.0)):.1f} GB "
                f"the watchdog keeps in reserve).\n"
                f"    The commit limit is physical RAM plus the pagefile, so it is not the "
                f"same thing as free RAM and closing programs is not the only lever - "
                f"enlarging the pagefile raises it directly.\n"
            )
            # "Close other programs" is useless advice without a list: the
            # programs holding the commit are exactly the ones whose working
            # sets are small enough that Task Manager's default sort hides
            # them. Name them, and say whether closing them is actually enough
            # - if it is not, the honest answer is the pagefile, not a
            # scavenger hunt.
            holders = top_commit_processes(limit=6, exclude_pids=own_pid_tree())
            if holders:
                message += f"    The commit is held by: {format_commit_holders(holders)}.\n"
                shortfall = commit_need_mb * 1024 * 1024 - commit_headroom
                recoverable = sum(total for _name, _count, total in holders)
                if recoverable >= shortfall:
                    message += (f"    Closing those frees {fmt_gb(recoverable)} - enough "
                                f"for this run.\n")
            reboot = pagefile_reboot_pending()
            if reboot:
                message += f"    {reboot}\n"
            message += ("    Fix: enlarge the pagefile, close other programs, drop to a "
                        "lighter quality preset, or reduce Threads.")
            blocking.append(("commit", message))
        elif commit_need_mb * 1024 * 1024 > commit_headroom * 0.7:
            warnings.append(
                f"Committed memory will be tight: ~{commit_need_mb / 1024:.1f} GB needed "
                f"against {fmt_gb(commit_headroom)} available of a "
                f"{fmt_gb(commit_limit_bytes())} commit limit. A long run is the one most "
                f"likely to reach it."
            )

    # Physical RAM. This is the check that was missing: job_memory_cap_gb is
    # sized from the commit limit (RAM + pagefile, 102 GB here) and so says
    # nothing about whether a run fits in the ~20 GB the machine actually has
    # free. When it does not, the watchdog aborts part-way through - which is
    # exactly the "my RAM goes over even on a 1 minute clip" failure.
    total_ram, avail_ram, avail_page = memory_status()
    resident_need_mb = estimate_resident_mb(pipeline_config, pass_pixels, guard_config)
    estimate["resident_need_mb"] = resident_need_mb
    estimate["avail_ram"] = avail_ram

    floor = guard_config.get("min_free_ram_gb", 2.5) * GB
    headroom = max(0, avail_ram - floor)
    if avail_ram:
        in_use = total_ram - avail_ram
        if resident_need_mb * 1024 * 1024 > headroom:
            blocking.append((
                "ram",
                f"This run needs roughly {resident_need_mb / 1024:.1f} GB of RAM but only "
                f"{fmt_gb(headroom)} is usable right now "
                f"({fmt_gb(avail_ram)} free of {fmt_gb(total_ram)}, "
                f"{fmt_gb(in_use)} already in use by other programs, "
                f"minus the {guard_config.get('min_free_ram_gb', 2.5):.1f} GB the watchdog keeps "
                f"in reserve for Windows).\n"
                f"    Each eye of this clip is a {estimate['width'] // 2}x{estimate['height']} "
                f"frame, and raw frames that size are {pass_pixels * 3 / 1e6:.0f} MB each - which "
                f"is why resolution, not clip length, is what decides this.\n"
                f"    Fix: close other programs (that alone frees {fmt_gb(in_use)} here), "
                f"drop to a lighter quality preset, or reduce Threads."
            ))
        elif resident_need_mb * 1024 * 1024 > headroom * 0.7:
            warnings.append(
                f"RAM will be tight: ~{resident_need_mb / 1024:.1f} GB needed against "
                f"{fmt_gb(headroom)} usable ({fmt_gb(avail_ram)} free of {fmt_gb(total_ram)}). "
                f"Closing other programs would give this run room."
            )
        elif avail_ram < floor * 1.5:
            warnings.append(
                f"Only {fmt_gb(avail_ram)} of RAM is free right now. "
                f"Close other apps before starting."
            )

    # Frame interpolation is by far the most expensive thing this pipeline can
    # be asked to do, and unlike everything else its cost scales with the frame
    # COUNT rather than the clip's resolution alone. Saying so before the run
    # is the difference between a considered choice and a nasty surprise on
    # hour thirty.
    interp_multi = int(pipeline_config.get("interpolate_multi") or 1)
    if interp_multi > 1:
        interp_seconds = sum(stage["seconds"] for stage in estimate["runtime"]["stages"]
                             if stage["kind"] == "interp")
        new_frames = estimate["frames"] * (interp_multi - 1) * (2 if is_vr else 1)
        estimate["interpolate_hours"] = interp_seconds / 3600.0
        warnings.append(
            f"Frame interpolation is on ({interp_multi}x): "
            f"{new_frames:,} frames have to be synthesised"
            + (" (both eyes)" if is_vr else "")
            + f", which on this card takes about {fmt_duration(interp_seconds)} - on top "
            f"of the face swap.\n"
            f"    It raises {estimate['fps']:.0f} fps to "
            f"{estimate['fps'] * interp_multi:.0f}. Source that is already 90 fps has "
            f"very little to gain; footage at 24-30 fps has a lot."
        )

    # Where Windows pages to. Measured on this machine: the run itself is
    # steady at ~7 GB resident and ~42 GB committed against 32 GB of RAM, so
    # paging is not avoidable - and when a pagefile is on a mechanical disk,
    # the moment Windows needs those pages back the whole desktop stops for
    # ten or twenty seconds and the watchdog below kills the run. That looks
    # exactly like "out of RAM" and is not.
    slow_paging = slow_pagefile_warning()
    if slow_paging:
        warnings.append(slow_paging)
        estimate["slow_pagefile"] = True

    # A pagefile that has been enlarged but not yet applied is worth saying
    # even on a run that fits: the user has done the fix and is one reboot away
    # from the headroom, and nothing else on this machine will tell them.
    reboot_pending = pagefile_reboot_pending()
    if reboot_pending:
        warnings.append(reboot_pending)
        estimate["pagefile_reboot_pending"] = True

    return (blocking, warnings, estimate)


def clamp_config(pipeline_config, guard_config, log_cb=None, frame_pixels=None):
    """
    Rewrite unsafe pipeline settings into safe ones.

    Returns a new dict; the caller's config is not mutated. `frame_pixels` is
    the size of one frame of the pass that is about to run - for VR that is one
    eye, not the full stereo frame - and is what lets the thread cap take the
    clip's resolution into account.
    """
    config = dict(pipeline_config)
    changes = []

    max_threads = int(guard_config.get("max_execution_threads", 6))
    # Dropping execution threads to fit a run into RAM is a bad trade and is
    # only worth making when the alternative is not running at all. Measured on
    # a 4096x4096 eye: going from 4 threads to 1 saves 0.48 GB resident and
    # takes the pass from 73 s to 150 s. The ceiling is 8 now rather than 4 -
    # the old cap was fitted while onnxruntime's thread pools were spinning, so
    # extra threads bought nothing - and the trade only gets worse as the cap
    # rises: each thread is ~160 MB resident and buys real wall time (3 -> 6
    # threads is 98.8 s -> 89.2 s on a 183-frame pass). So this only fires when
    # the run does not otherwise fit, and it stops as soon as it does - it
    # never trims threads for headroom's sake.
    large_mpix = float(guard_config.get("large_frame_mpix", 8.0) or 8.0)
    traded_for_memory = False
    if frame_pixels and frame_pixels / 1e6 > large_mpix:
        _, avail_ram, _ = memory_status()
        floor = guard_config.get("min_free_ram_gb", 2.5) * GB
        headroom_mb = max(0, avail_ram - floor) / (1024 * 1024)
        while max_threads > 1 and headroom_mb:
            probe = dict(config, threads=max_threads)
            if estimate_resident_mb(probe, frame_pixels, guard_config) <= headroom_mb:
                break
            max_threads -= 1
            traded_for_memory = True

    threads = int(config.get("threads") or 1)
    if threads > max_threads:
        config["threads"] = max_threads
        if traded_for_memory:
            changes.append(
                f"threads {threads} -> {max_threads} - there is not enough free RAM "
                f"for more, and fewer threads is the only setting that buys any back "
                f"without changing the output. Expect this run to be roughly "
                f"{threads / max(1, max_threads):.0f}x slower; closing other programs "
                f"and starting again would avoid it"
            )
        else:
            changes.append(
                f"threads {threads} -> {max_threads} (max_execution_threads in "
                f"guardrails.json; each thread holds a full frame in VRAM and in RAM)"
            )

    # 'tolerant' never releases models from VRAM. On a 16 GB card with a
    # 512 swapper + 2048 enhancer that overflows into system RAM.
    strategy = guard_config.get("video_memory_strategy", "moderate")
    if config.get("video_memory_strategy") != strategy:
        config["video_memory_strategy"] = strategy
        changes.append(f"video memory strategy -> {strategy}")

    fmt = guard_config.get("temp_frame_format", "jpeg")
    if config.get("temp_frame_format") != fmt:
        config["temp_frame_format"] = fmt
        changes.append(f"temp frame format -> {fmt} (png temp frames are ~13x larger)")

    if guard_config.get("prefer_nvenc", True):
        vram_total_mb, _ = gpu_memory_mb()
        if vram_total_mb:
            software = {"libx264": "h264_nvenc", "libx265": "hevc_nvenc"}
            encoder = config.get("encoder")
            if encoder in software:
                config["encoder"] = software[encoder]
                changes.append(
                    f"encoder {encoder} -> {software[encoder]} "
                    f"(software encoding pins every CPU core)"
                )

    if log_cb and changes:
        log_cb("[Guard] Adjusted settings for safety:")
        for change in changes:
            log_cb(f"  - {change}")

    return config


# ---------------------------------------------------------------------------
# ResourceGuard
# ---------------------------------------------------------------------------

class ResourceGuard:
    """
    Owns a Windows Job Object that every pipeline child process is placed in.

    The job gives three hard guarantees:
      * total committed memory for all children is capped
      * total CPU for all children is capped
      * closing the job (app exit, Stop button, crash) kills every child,
        so nothing is ever left running in the background

    Plus a watchdog thread that aborts the job if free RAM or free disk gets
    dangerously low despite all of the above.
    """

    def __init__(self, temp_dir, guard_config=None, log_cb=None):
        self.temp_dir = temp_dir
        self.config = guard_config or load_guard_config()
        self.log_cb = log_cb

        self._job = None
        self._procs = []
        self._lock = threading.RLock()
        self._closed = False

        self.abort_reason = None
        self._watchdog = None
        self._watchdog_stop = threading.Event()

        self._paused = False
        self._paused_at = None
        self._paused_total = 0.0
        # "user" or "guard". Only a freeze the guard applied may be lifted
        # automatically - a run the user paused stays paused.
        self._paused_by = None

        # Low-water mark of system-wide available commit over the run. The job
        # cap only governs what the children commit; the commit limit governs
        # the machine, and it is the one that bites first here. Without this
        # there is no way to tell the two apart after the fact.
        self._min_avail_commit = 0

        # -- lanes ---------------------------------------------------------
        #
        # A lane is a named part of one run that can be frozen without freezing
        # the rest of it: the right eye while the left keeps swapping, the
        # background split while the swap has the GPU. Everything in a lane is
        # still in the job object, so the memory cap, the CPU cap and
        # kill-on-close all still cover it - a lane is a finer handle on the
        # same processes, not a second guard.
        #
        # This exists because the whole-job freeze is too blunt once a run has
        # more than one thing in flight. Two FaceFusion lanes at 4K hold ~7 GB
        # of VRAM each on a 16.3 GB card; when the card runs short the right
        # answer is to hold one of them and let the other finish, not to stop
        # both and wait for memory that only the run itself is using.
        # lane -> the Popen objects spawned into it. The processes rather than
        # their PIDs, because a PID is only unique while the process is alive:
        # Windows recycles them, and a run that spawns a few hundred ffmpegs
        # over eight hours will eventually see one come back. Holding the Popen
        # both keeps the handle open (so the number cannot be reused underneath
        # us) and gives poll() as an exact answer to "is this still the process
        # I meant?".
        self._lanes = {}
        self._lane_order = []     # registration order; the newest is held first
        self._lane_paused = {}    # lane -> (frozen since, who froze it)
        # Set when the watchdog gives up on a held lane and kills it. The
        # caller reads this and re-runs that lane sequentially; the rest of the
        # run is never aborted for one lane's failure.
        self.lane_failed = {}
        # Wall-clock during which EVERY live lane was frozen at once. That, and
        # not a single held lane, is what stops the run clock: while one lane is
        # still swapping the run is still making progress and the estimate must
        # be charged for it.
        self._lane_frozen_at = None
        self._lane_frozen_total = 0.0

    # -- logging ----------------------------------------------------------

    def _log(self, msg):
        if self.log_cb:
            try:
                self.log_cb(msg)
            except Exception:
                pass

    # -- lifecycle --------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def start(self, unattended=False):
        """
        Create the job object and apply its limits.

        `unattended` is whether anybody is at the keyboard. It only selects the
        CPU cap: 75% exists to keep the shell and the UI schedulable during a
        long encode, and at 03:00 on a queued job there is nothing to keep
        schedulable, while an overlapped split or merge (and later a second
        lane) can use the difference. Everything else about the job is the same
        either way.
        """
        if not self.config.get("enabled", True):
            self._log("[Guard] Guard rails are DISABLED in guardrails.json.")
            return self
        if not IS_WINDOWS:
            return self

        self._job = _kernel32.CreateJobObjectW(None, None)
        if not self._job:
            self._log("[Guard] Could not create job object; running unguarded.")
            return self

        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        cap_gb = float(self.config.get("job_memory_cap_gb", 0) or 0)
        if cap_gb > 0:
            info.JobMemoryLimit = int(cap_gb * GB)
            flags |= JOB_OBJECT_LIMIT_JOB_MEMORY

        if self.config.get("below_normal_priority", True):
            info.BasicLimitInformation.PriorityClass = BELOW_NORMAL_PRIORITY_CLASS
            flags |= JOB_OBJECT_LIMIT_PRIORITY_CLASS

        info.BasicLimitInformation.LimitFlags = flags

        if not _kernel32.SetInformationJobObject(
            self._job, JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info)
        ):
            self._log(
                f"[Guard] Could not apply job limits (error {ctypes.get_last_error()}); "
                f"continuing with reduced protection."
            )

        cap_key = "cpu_percent_cap_unattended" if unattended else "cpu_percent_cap"
        cpu_cap = int(self.config.get(cap_key, 0)
                      or self.config.get("cpu_percent_cap", 0) or 0)
        if 0 < cpu_cap < 100:
            rate = _JOBOBJECT_CPU_RATE_CONTROL_INFORMATION()
            rate.ControlFlags = (
                JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP
            )
            rate.u.CpuRate = cpu_cap * 100  # units of 1/100th of a percent
            if not _kernel32.SetInformationJobObject(
                self._job, JobObjectCpuRateControlInformation,
                ctypes.byref(rate), ctypes.sizeof(rate)
            ):
                self._log("[Guard] CPU rate cap unavailable on this system.")
                cpu_cap = 0

        redrawn_from = float(self.config.get("_cap_clamped_from_gb", 0) or 0)
        if redrawn_from > cap_gb:
            self._log(
                f"[Guard] Memory cap lowered from the stored {redrawn_from:.1f} GB to "
                f"{cap_gb:.1f} GB. This machine's commit limit is now "
                f"{fmt_gb(commit_limit_bytes())}, so the stored cap could never have been "
                f"reached - the run would have died on the system commit limit first, as a "
                f"bare allocation failure with nothing to explain it. Re-derived from the "
                f"machine as it is now; edit scripts/guardrails.json to go lower still."
            )
        elif redrawn_from:
            self._log(
                f"[Guard] Memory cap raised from the stored {redrawn_from:.1f} GB to "
                f"{cap_gb:.1f} GB: this machine's commit limit has grown to "
                f"{fmt_gb(commit_limit_bytes())} since that cap was written."
            )

        self._log(
            f"[Guard] Active - memory cap {cap_gb:.1f} GB, "
            f"CPU cap {cpu_cap or 100}% "
            f"({'unattended' if unattended else 'interactive'} limit), "
            f"priority {'below normal' if self.config.get('below_normal_priority', True) else 'normal'}"
        )

        self._start_watchdog()
        return self

    def close(self, wait_timeout=12.0):
        """
        Kill anything still running and release the job.

        `wait_timeout` is how long to wait for the run's own processes to
        actually exit before returning. That wait is the difference between a
        run that frees its memory and one that does not: closing the job tells
        Windows to terminate the children, but a CUDA process does not hand its
        committed memory back the instant TerminateProcess returns - the kernel
        tears the address space down on its own schedule, and a FaceFusion
        child reserves ~13 GB of commit it never makes resident. Until that
        teardown finishes the commit is still charged against the machine, so
        the next queued job's preflight - which reads the system commit charge
        the moment this returns - sees the finished job's memory as still in
        use and refuses to start. Waiting here is what makes "one video
        finished" actually give the memory back before the next one is sized.

        Pass 0 to skip the wait (app shutdown, where the process is exiting and
        the OS reclaims everything wholesale anyway).
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True

        self._watchdog_stop.set()
        if self._watchdog and self._watchdog.is_alive():
            self._watchdog.join(timeout=5)

        if self._job:
            peak = self.peak_memory_bytes()
            if peak:
                self._log(f"[Guard] Peak memory used by pipeline: {fmt_gb(peak)}")
            if self._paused_total:
                self._log(f"[Guard] Total time frozen to stay under the memory "
                          f"limits: {fmt_duration(self._paused_total)} - none of it "
                          f"charged to the estimate, and no stage was lost to it.")
            # KILL_ON_JOB_CLOSE means this also reaps every surviving child.
            _kernel32.CloseHandle(self._job)
            self._job = None

        self._reap_stragglers(wait_timeout)
        self._report_released()

    def _report_released(self):
        """
        Say what came back, so 'did that run leave anything behind?' is
        answerable from the log.

        Every model in this pipeline lives in a child process - FaceFusion, the
        CUDA probe, the Real-ESRGAN worker - so closing the job is what frees
        both host and device memory: the OS reclaims it wholesale when the
        children exit. Nothing is cached in this process to release.

        This runs on its own thread: close() is called from the UI thread at
        the end of every run, and this polls nvidia-smi up to five times with
        quarter-second waits between them. Doing that inline froze the window
        for a second or so each time a run finished.
        """
        threading.Thread(
            target=self._report_released_blocking, name="vrdeep-release", daemon=True
        ).start()

    def _report_released_blocking(self):
        total_mb, used_mb = gpu_memory_mb()
        if not total_mb:
            return

        # The driver can lag the process exit by a moment, so give it a chance,
        # but only while the number is still falling and for about a second
        # either way.
        for _ in range(4):
            if used_mb < total_mb * 0.25:
                break
            time.sleep(0.25)
            _, now_mb = gpu_memory_mb()
            if now_mb >= used_mb:
                used_mb = now_mb
                break
            used_mb = now_mb

        self._log(
            f"[Guard] Memory released. VRAM now in use across the whole system: "
            f"{used_mb / 1024:.1f} GB of {total_mb / 1024:.1f} GB."
        )

    # -- process management ----------------------------------------------

    def spawn(self, cmd, lane=None, **kwargs):
        """
        subprocess.Popen, with the child adopted into the guarded job.

        `lane` names the part of the run this process belongs to ("left",
        "right", "split", "merge"). It is optional and every existing caller
        omits it, in which case the process belongs to no lane and is only ever
        touched by the whole-job pause. A lane is registered the first time
        something is spawned into it, and the registration order is what
        decides which lane the watchdog holds first: the newest, because the
        oldest is the one closest to finishing.
        """
        if self.abort_reason:
            raise GuardAbort(self.abort_reason)

        flags = kwargs.pop("creationflags", 0)
        if IS_WINDOWS:
            # Own process group so we can signal the tree, and no console popup.
            flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
            flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        proc = subprocess.Popen(cmd, creationflags=flags, **kwargs)
        self._adopt(proc)

        with self._lock:
            self._procs.append(proc)
            paused = self._paused
            lane_held = False
            if lane:
                if lane not in self._lanes:
                    self._lanes[lane] = []
                    self._lane_order.append(lane)
                self._lanes[lane].append(proc)
                lane_held = lane in self._lane_paused
            self._recount_lane_freeze()
        if paused or lane_held:
            # Closes the one race a freeze has: a stage that spawns between
            # pause() reading the job's PID list and the freeze taking hold
            # would otherwise be the only thing still running. The same race
            # exists per lane, for a lane that respawns while it is held.
            _set_process_suspended(proc.pid, True)
        return proc

    def _adopt(self, proc):
        """Assign a freshly spawned process to the job object."""
        if not self._job or not IS_WINDOWS:
            return
        try:
            handle = int(proc._handle)
        except Exception:
            return
        if not _kernel32.AssignProcessToJobObject(self._job, wintypes.HANDLE(handle)):
            err = ctypes.get_last_error()
            self._log(f"[Guard] Could not place PID {proc.pid} under the job (error {err}).")

    def run(self, cmd, timeout=None, lane=None, **kwargs):
        """subprocess.run equivalent that goes through the job object."""
        capture = kwargs.pop("capture_output", False)
        if capture:
            kwargs.setdefault("stdout", subprocess.PIPE)
            kwargs.setdefault("stderr", subprocess.PIPE)

        proc = self.spawn(cmd, lane=lane, **kwargs)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.kill_all()
            raise
        finally:
            self._forget(proc)

        if self.abort_reason and proc.returncode not in (0,):
            raise GuardAbort(self.abort_reason)

        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)

    def _forget(self, proc):
        with self._lock:
            if proc in self._procs:
                self._procs.remove(proc)

    def kill_all(self):
        """Terminate every process in the job, immediately."""
        if self._job and IS_WINDOWS:
            _kernel32.TerminateJobObject(self._job, 1)
        self._reap_stragglers()

    def _reap_stragglers(self, wait_timeout=12.0):
        """
        Belt-and-braces kill for anything the job did not cover, then wait for
        the run's own processes to be gone so their committed memory is back.

        See close() for why the wait matters: a terminated CUDA process keeps
        its commit charged for a moment after it is killed, and the next run is
        sized against the system commit charge the instant this returns. Each
        wait() returns at once for a process that has already exited - which,
        on a clean finish, is all of them - so the timeout only ever bites on a
        child that is genuinely stuck mid-teardown.
        """
        with self._lock:
            procs = list(self._procs)
            self._procs.clear()
        for proc in procs:
            if proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
        deadline = time.time() + max(0.0, wait_timeout)
        for proc in procs:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                proc.wait(timeout=remaining)
            except Exception:
                # Timed out, or not waitable; the job handle is already closed,
                # so there is nothing more we can do to hurry it along.
                pass

    # -- pause ------------------------------------------------------------
    #
    # A VR run is hours of GPU work and 28 cores, and until now the only way to
    # get the machine back was to stop it. Stopping keeps the finished stages
    # since the checkpoints landed, but it still throws away whatever the
    # current eye had rendered - up to an hour of it.
    #
    # Pausing freezes every process in the job instead. Nothing is lost, no
    # stage restarts, and one click puts it back exactly where it was.
    #
    # It is a freeze, not an unload: a paused run keeps its RAM and its VRAM.
    # This gives back the CPU and the GPU's *time*, not the card's memory.
    # Stop is still what hands the VRAM back.

    def pause(self, by="user"):
        """Suspend every process in the job. True if it took effect."""
        with self._lock:
            if self._closed or self._paused or self.abort_reason:
                return False
            self._paused = True
            self._paused_at = time.time()
            self._paused_by = by
        # A lane the watchdog is already holding is left exactly as it is:
        # suspend counts nest, so freezing it a second time here would need two
        # resumes and resume() only ever issues one. The lanes it is holding are
        # remembered by _lane_paused, and resume() reads the same set.
        frozen = self._set_suspended(True, skip_pids=self._held_lane_pids())
        if not frozen and not self._lane_paused:
            with self._lock:
                self._paused = False
                self._paused_at = None
                self._recount_lane_freeze()
            self._log("[Guard] Nothing to pause - no pipeline process is running.")
            return False
        if not frozen:
            # Everything that is left was already frozen by the watchdog, lane
            # by lane. The pause still has to take: otherwise the moment the
            # watchdog released those lanes the run would start again, and the
            # user had pressed Pause.
            self._log("[Guard] PAUSED - every remaining process was already frozen by "
                      "the guard rails; the run stays down until you resume it.")
            return True
        self._log(f"[Guard] PAUSED - {frozen} process(es) frozen exactly where they "
                  f"were. They keep their RAM and VRAM; Stop is what gives those back.")
        return True

    def resume(self):
        """Thaw a paused run. True if it was paused."""
        with self._lock:
            if not self._paused:
                return False
            self._paused = False
            held = time.time() - (self._paused_at or time.time())
            self._paused_total += held
            self._paused_at = None
            self._paused_by = None
        # Whatever the watchdog is holding per lane stays held. Thawing it here
        # would undo a freeze that is still keeping the machine alive, and the
        # user's Resume means "put the run back", not "override the guard".
        thawed = self._set_suspended(False, skip_pids=self._held_lane_pids())
        with self._lock:
            self._recount_lane_freeze()
            still_held = sorted(self._lane_paused)
        self._log(f"[Guard] RESUMED after {fmt_duration(held)} - {thawed} process(es) "
                  f"running again. Paused time is not charged to the estimate."
                  + (f" The {', '.join(still_held)} lane(s) stay frozen - the guard "
                     f"rails are still holding them." if still_held else ""))
        return True

    @property
    def paused(self):
        return self._paused

    @property
    def paused_by(self):
        """"user", "guard", or None - who is holding the freeze."""
        return self._paused_by

    def free_memory_now(self):
        """
        Freeze-and-trim, returning how much free RAM that recovered.

        The freeze is what makes the trim safe and the trim is what makes the
        freeze useful; neither is much good alone.
        """
        # Measured as the drop in THIS run's resident memory, not as the rise
        # in the machine's free memory: the second is what we care about but it
        # moves for everyone's reasons, and a figure that reads "handed back
        # 0.0 GB" because something else allocated at the same moment is worse
        # than no figure at all.
        before = self.job_memory_now()
        trimmed = sum(1 for pid in _job_process_ids(self._job) if empty_working_set(pid))
        # The pages move to the standby list as the trim walks them, so give it
        # a moment before claiming a figure.
        time.sleep(0.5)
        return trimmed, max(0, before - self.job_memory_now())

    def paused_seconds(self):
        """
        Wall-clock this run has spent frozen, including the current pause.

        Two ways to be frozen and only one of them stops the clock. A whole-job
        pause does: nothing is running. A single held lane does not: the other
        eye is still swapping, the run is still making progress, and charging
        the estimate for that time would report a swap that took an hour as
        having taken two. Only a moment when *every* live lane is frozen at
        once counts, and _recount_lane_freeze is what notices those.
        """
        with self._lock:
            total = self._paused_total + self._lane_frozen_total
            if self._paused and self._paused_at:
                total += time.time() - self._paused_at
            if self._lane_frozen_at:
                total += time.time() - self._lane_frozen_at
        return total

    def _set_suspended(self, suspend, skip_pids=()):
        """
        Freeze or thaw the job. Parents first on the way down, so that nothing
        can spawn behind the freeze; children first on the way up.

        `skip_pids` are left exactly as they are - the lanes the watchdog is
        holding, which must not be thawed by a user Resume nor frozen a second
        time by a user Pause (NtSuspendProcess nests, and one NtResumeProcess
        would then leave them stopped).
        """
        skip = set(int(pid) for pid in skip_pids or ())
        pids = _job_process_ids(self._job)
        if not pids:
            # No job object - guard rails off, or the assignment failed. The
            # processes this app started directly are all that can be reached,
            # and their own children keep running. Do not claim otherwise.
            with self._lock:
                pids = [p.pid for p in self._procs if p.poll() is None]
            if pids and suspend:
                self._log("[Guard] No job object, so only the processes started "
                          "directly by this app can be frozen - anything they "
                          "spawned keeps running.")
        pids = [pid for pid in pids if pid not in skip]
        if not suspend:
            pids = list(reversed(pids))
        return sum(1 for pid in pids if _set_process_suspended(pid, suspend))

    # -- lanes -------------------------------------------------------------
    #
    # A lane is a named part of one run that can be frozen without freezing the
    # rest of it. Everything in it is still in the job object, so the memory
    # cap, the CPU cap and kill-on-close all still cover it: a lane is a finer
    # handle on the same processes, not a second guard.

    def lanes(self):
        """Every lane that has been spawned into, in registration order."""
        with self._lock:
            return list(self._lane_order)

    def live_lanes(self):
        """
        Lanes that still have a process running.

        Deliberately answered from the Popen objects rather than from
        lane_pids(): this is read on every watchdog sample and a process
        snapshot per lane per three seconds, for the whole length of a run, is
        a cost with nothing to show for it.
        """
        return [lane for lane in self.lanes() if self._lane_roots(lane)]

    def _lane_roots(self, lane):
        """PIDs of the lane's own processes that are still running."""
        with self._lock:
            procs = list(self._lanes.get(lane) or ())
        return [proc.pid for proc in procs if proc.poll() is None]

    def lane_pids(self, lane):
        """
        Every process of `lane` that is alive right now: the roots spawned into
        it plus their descendants.

        The descendant walk is a parent-PID walk, which is unsafe in general -
        Windows recycles PIDs and leaves a dead parent's ID on its children - so
        it is intersected with _job_process_ids(self._job), the kernel's own
        list of who is in this run's job object. A stale parent PID can then
        only ever pull in another process of this same run; it can never reach
        anything else on the machine. That intersection is the entire safety
        argument for suspending, trimming and killing lanes, and it is why the
        lane roots are recorded at spawn time rather than looked up later.
        """
        roots = set(self._lane_roots(lane))
        if not roots:
            return []
        allowed = set(_job_process_ids(self._job))
        if not allowed:
            # No job object (guard rails off, or the assignment failed). Only
            # the roots this app started itself can be reached; claiming their
            # children too would be a lie, and acting on them unsafe.
            with self._lock:
                live = set(proc.pid for proc in self._procs if proc.poll() is None)
            return sorted(roots & live)

        found = set(root for root in roots if root in allowed)
        parents = parent_map()
        # The snapshot is in no particular order, so one pass would miss a
        # grandchild listed before its parent. Repeat until nothing new joins.
        changed = True
        while changed:
            changed = False
            for pid, parent in parents.items():
                if pid in allowed and pid not in found and parent in found:
                    found.add(pid)
                    changed = True
        return sorted(found)

    def _held_lane_pids(self):
        """Every PID belonging to a lane that is frozen right now."""
        pids = set()
        for lane in list(self._lane_paused):
            pids.update(self.lane_pids(lane))
        return pids

    def lane_memory_now(self, lane):
        """Working set held by one lane's processes at this moment."""
        return sum(process_memory(pid)[0] for pid in self.lane_pids(lane))

    def lane_paused(self, lane):
        """True if `lane` is frozen right now."""
        return lane in self._lane_paused

    def pause_lane(self, lane, by="guard"):
        """
        Freeze one lane and leave the rest of the run running.

        Parents first, exactly as the whole-job freeze does, so nothing in the
        lane can spawn behind the freeze. Returns how many processes actually
        stopped; 0 means the lane is already gone or already held, and nothing
        is recorded in that case.
        """
        with self._lock:
            if self._closed or self.abort_reason or lane in self._lane_paused:
                return 0
        pids = self.lane_pids(lane)
        frozen = sum(1 for pid in pids if _set_process_suspended(pid, True))
        if not frozen:
            return 0
        with self._lock:
            self._lane_paused[lane] = (time.time(), by)
            self._recount_lane_freeze()
        return frozen

    def resume_lane(self, lane):
        """Thaw one held lane. Children first, so nothing runs into a frozen parent."""
        with self._lock:
            if lane not in self._lane_paused:
                return 0
            self._lane_paused.pop(lane, None)
            self._recount_lane_freeze()
            whole_job_paused = self._paused
        pids = list(reversed(self.lane_pids(lane)))
        thawed = sum(1 for pid in pids if _set_process_suspended(pid, False))
        # The user may have paused the whole run while this lane was held, in
        # which case that freeze skipped these processes and thawing them here
        # would start them running behind the user's back.
        if whole_job_paused:
            for pid in pids:
                _set_process_suspended(pid, True)
        return thawed

    def lane_held_seconds(self, lane):
        """How long `lane` has been frozen, or 0 if it is running."""
        held = self._lane_paused.get(lane)
        return max(0.0, time.time() - held[0]) if held else 0.0

    def kill_lane(self, lane, reason=""):
        """
        Terminate one lane and record why, without touching the rest of the run.

        The caller (the both-eyes-at-once path in engine) reads `lane_failed`
        and re-runs that eye sequentially. A lane that could not be made to fit
        is a reason to stop trying to run two at once, never a reason to throw
        away the eye that did fit.
        """
        pids = self.lane_pids(lane)
        with self._lock:
            self._lane_paused.pop(lane, None)
            self.lane_failed[lane] = reason or "the guard rails stopped this lane"
            self._lanes.pop(lane, None)
            self._recount_lane_freeze()
        # Children first: killing the parent first can leave an ffmpeg holding
        # a pipe nobody will ever read.
        return sum(1 for pid in reversed(pids) if _terminate_process(pid))

    def _recount_lane_freeze(self):
        """
        Start or stop the all-lanes-frozen clock. Call with the lock held.

        The condition is "there are live lanes and every one of them is held",
        and it is deliberately false while the whole job is paused: that time is
        already counted by _paused_total, and counting it twice would tell
        RunClock the run had been frozen for longer than it has been running.
        """
        live = [lane for lane in self._lane_order
                if any(proc.poll() is None for proc in self._lanes.get(lane) or ())]
        all_held = (bool(live) and not self._paused
                    and all(lane in self._lane_paused for lane in live))
        now = time.time()
        if all_held and self._lane_frozen_at is None:
            self._lane_frozen_at = now
        elif not all_held and self._lane_frozen_at is not None:
            self._lane_frozen_total += now - self._lane_frozen_at
            self._lane_frozen_at = None

    def cancel(self, reason="Cancelled by user"):
        # A frozen process still dies on TerminateJobObject, but thawing first
        # means the pipeline thread's blocked communicate() returns at once
        # rather than after the kill has propagated.
        if self._paused:
            self.resume()
        self.abort(reason)

    def abort(self, reason):
        if self.abort_reason:
            return
        self.abort_reason = reason
        self._log(f"[Guard] ABORTING RUN: {reason}")
        self.kill_all()

    def check(self):
        """Raise GuardAbort if the run has been stopped. Call between stages."""
        if self.abort_reason:
            raise GuardAbort(self.abort_reason)

    def peak_memory_bytes(self):
        if not self._job or not IS_WINDOWS:
            return 0
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        returned = wintypes.DWORD(0)
        ok = _kernel32.QueryInformationJobObject(
            self._job, JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(returned)
        )
        return info.PeakJobMemoryUsed if ok else 0

    def min_available_commit_bytes(self):
        """
        Least system-wide available commit seen during the run, in bytes.

        Zero when the watchdog is off or never got a sample. Read it next to
        the job's own peak: if this fell to near nothing while the job peak sat
        well under its cap, the machine ran out of commit and the pipeline did
        not - which is a completely different problem with completely different
        fixes, and the two are indistinguishable from the traceback alone.
        """
        return self._min_avail_commit

    def memory_cap_bytes(self):
        """The job's committed-memory ceiling, or 0 if none is set."""
        return int(float(self.config.get("job_memory_cap_gb", 0) or 0) * GB)

    def hit_memory_cap(self):
        """
        True if the job got close enough to its committed-memory ceiling that a
        failure inside it is very likely to be an allocation failure.

        Worth checking whenever a child dies unexpectedly: once the cap is
        reached every allocation in the job fails, and what surfaces is an
        onnxruntime or numpy memory error with nothing in it that points back
        here.
        """
        cap = self.memory_cap_bytes()
        peak = self.peak_memory_bytes()
        return bool(cap and peak and peak >= cap * 0.97)

    def job_memory_now(self):
        """Working set held by this run's own processes at this moment."""
        return sum(process_memory(pid)[0] for pid in _job_process_ids(self._job))

    def job_pids(self):
        return _job_process_ids(self._job)

    def explain_memory_pressure(self):
        """
        Lines attributing the machine's memory to this run and to everything
        else, logged when the watchdog trips.

        peak_memory_bytes() already answers "did the pipeline do this?". When
        the answer is no - and on a machine that is also doing something else
        that is the usual answer - nothing answered "then what did?", and an
        abort reading "free RAM down to 0.0 GB" left the user to conclude they
        needed a smaller pipeline or more RAM when what they needed was to
        close something.
        """
        mine = self.job_memory_now()
        total, avail, _ = memory_status()
        in_use = max(0, total - avail)
        lines = [f"[Guard] Of the {fmt_gb(in_use)} in use, this run's own processes "
                 f"hold {fmt_gb(mine)}."]
        mine_pids = self.job_pids()
        others = top_memory_processes(limit=5, exclude_pids=mine_pids)
        if others:
            listed = ", ".join(f"{name} {fmt_gb(size)}" for name, _pid, size in others)
            lines.append(f"[Guard] Largest processes outside this run: {listed}")
        # Working set and commit charge name different programs, and half the
        # watchdog's trips are commit trips. Without this line an abort on
        # commit lists the processes holding RAM, which on this machine is a
        # different set entirely from the ones holding the commit limit.
        holders = top_commit_processes(limit=5,
                                       exclude_pids=set(mine_pids) | own_pid_tree())
        if holders:
            lines.append(f"[Guard] Largest holders of committed memory outside this "
                         f"run: {format_commit_holders(holders)}")
        # Every process's working set added together leaves a large
        # remainder on most machines, and that remainder is where an
        # unexplainable free-RAM abort usually lives.
        info = system_memory_breakdown()
        if info:
            pools = info["kernel_paged"] + info["kernel_nonpaged"]
            lines.append(
                f"[Guard] Kernel pools hold {fmt_gb(pools)} "
                f"(nonpaged {fmt_gb(info['kernel_nonpaged'])}); commit charge is "
                f"{fmt_gb(info['commit_total'])} of {fmt_gb(info['commit_limit'])}, "
                f"peak {fmt_gb(info['commit_peak'])}.")
            if info["commit_peak"] >= info["commit_limit"] * 0.98:
                lines.append(
                    "[Guard] Commit charge has already reached this machine's ceiling "
                    f"of {fmt_gb(info['commit_limit'])} at some point since boot. At "
                    "that ceiling Windows trims every working set looking for pages "
                    "and free RAM collapses - which is what a 'free RAM down to 0.0 "
                    "GB' abort looks like from the inside. The ceiling is RAM plus "
                    "pagefile, so it is the pagefile that raises it.")
            if info["kernel_nonpaged"] > 4 * GB:
                lines.append(
                    "[Guard] That nonpaged pool figure is abnormal - it is kernel "
                    "memory that can never be paged out, and no application can "
                    "free it. A leaking driver is the usual cause; a reboot is the "
                    "usual fix, and nothing in this app will help.")
        if mine and in_use and mine < in_use * 0.5:
            lines.append("[Guard] The pipeline is NOT what filled the memory - closing "
                         "whatever is listed above will do more than lowering any "
                         "setting in this app.")
        return lines

    # -- watchdog ---------------------------------------------------------

    def _start_watchdog(self):
        if not self.config.get("watchdog_enabled", True):
            return
        self._watchdog = threading.Thread(
            target=self._watchdog_loop, name="vrdeep-watchdog", daemon=True
        )
        self._watchdog.start()

    def _watchdog_loop(self):
        interval = float(self.config.get("watchdog_interval_sec", 3.0))
        max_strikes = int(self.config.get("watchdog_strikes", 3))
        min_ram = self.config.get("min_free_ram_gb", 2.5) * GB
        min_disk = self.config.get("min_free_disk_gb", 12.0) * GB
        min_page = self.config.get("min_free_pagefile_gb", 3.0) * GB

        auto_pause = bool(self.config.get("auto_pause_enabled", True))
        pause_ram = float(self.config.get("auto_pause_free_ram_gb", 5.0)) * GB
        resume_ram = float(self.config.get("auto_resume_free_ram_gb", 7.5)) * GB
        max_hold = float(self.config.get("auto_pause_max_sec", 1800))
        # A pause point at or below the abort point would never fire, and one
        # that resumes below where it paused would oscillate.
        pause_ram = max(pause_ram, min_ram + 1.0 * GB)
        resume_ram = max(resume_ram, pause_ram + 1.0 * GB)

        # Lane trip points. Deliberately above the whole-job ones: with two
        # lanes in flight the first thing to try is holding one of them, which
        # costs the run nothing because the other lane still has the GPU. These
        # are read fresh on every sample rather than captured here, so a test
        # (or a user) can move them at runtime and see the watchdog react - the
        # config dict is the same object the caller holds.
        lane_resume_streak = 0
        last_lane_report = 0.0

        strikes = 0
        warned = set()
        holds = 0
        last_hold_report = 0.0
        last_freeze_fail = 0.0

        while not self._watchdog_stop.wait(interval):
            if self.abort_reason:
                return

            _, avail_ram, avail_page = memory_status()
            free_disk = free_disk_bytes(self.temp_dir)

            if avail_page:
                self._min_avail_commit = (
                    min(self._min_avail_commit, avail_page)
                    if self._min_avail_commit else avail_page
                )

            problems = []
            if avail_ram and avail_ram < min_ram:
                problems.append(f"free RAM down to {fmt_gb(avail_ram)}")
            if free_disk and free_disk < min_disk:
                problems.append(f"free temp disk down to {fmt_gb(free_disk)}")
            if avail_page and avail_page < min_page:
                problems.append(f"free commit charge down to {fmt_gb(avail_page)}")

            if self._paused:
                # A frozen run is not what is eating the machine: it touches
                # nothing, and its pages have been handed back. Killing it here
                # would destroy exactly the work the freeze was protecting.
                # Keep reporting, do not act.
                strikes = 0
                if self._paused_by != "guard":
                    if problems and "paused" not in warned:
                        warned.add("paused")
                        self._log("[Guard] " + ", ".join(problems)
                                  + " - but the run is paused, so the watchdog is "
                                    "only watching. It arms again on resume.")
                    continue

                held = time.time() - (self._paused_at or time.time())
                if avail_ram and avail_ram >= resume_ram:
                    self._log(f"[Guard] Free RAM is back to {fmt_gb(avail_ram)} - "
                              f"thawing the run after {fmt_duration(held)}. Nothing "
                              f"was lost; the estimate is not charged for the wait.")
                    self.resume()
                elif held >= max_hold:
                    self.abort(
                        f"The run was frozen for {fmt_duration(held)} waiting for "
                        f"memory to come back and it never did (free RAM "
                        f"{fmt_gb(avail_ram)}, needs {fmt_gb(resume_ram)}). Stopping "
                        f"rather than holding the work hostage indefinitely - every "
                        f"finished stage is kept and the next run resumes from them."
                    )
                    return
                elif time.time() - last_hold_report > 60:
                    last_hold_report = time.time()
                    self._log(f"[Guard] Still frozen after {fmt_duration(held)} - free "
                              f"RAM {fmt_gb(avail_ram)}, waiting for "
                              f"{fmt_gb(resume_ram)}. Closing something releases it.")
                continue

            # Lanes first. A run with two lanes in flight has something the
            # whole-job freeze does not: a part it can stop that is not the
            # whole. Holding the newest lane leaves the older one - the one
            # closest to finishing - with the whole GPU, and costs the run
            # nothing but that lane's wall time. Only when this cannot help
            # does the whole-job freeze below fire, and only then the abort.
            #
            # Everything in here is skipped entirely on a single-lane run,
            # which is every run that existed before this: no extra samples, no
            # nvidia-smi, no behaviour change.
            live_lanes = self.live_lanes() if len(self._lane_order) >= 2 else []
            if len(live_lanes) >= 2:
                lane_ram = float(self.config.get("lane_pause_free_ram_gb", 6.5)) * GB
                lane_commit = float(self.config.get("lane_pause_free_commit_gb", 6.0)) * GB
                lane_vram = float(self.config.get("lane_pause_free_vram_mb", 1200))
                margin = float(self.config.get("lane_resume_margin_gb", 2.0)) * GB
                lane_max_hold = float(self.config.get("lane_pause_max_sec", 1800))

                # ~50 ms of subprocess, so only on a multi-lane run and only
                # once per watchdog interval. VRAM is the resource two swap
                # lanes actually run out of, and nothing else here can see it.
                vram_total_mb, vram_used_mb = gpu_memory_mb()
                free_vram_mb = (vram_total_mb - vram_used_mb) if vram_total_mb else None

                trips = []
                if avail_ram and avail_ram < lane_ram:
                    trips.append(f"free RAM down to {fmt_gb(avail_ram)}")
                if avail_page and avail_page < lane_commit:
                    trips.append(f"free commit down to {fmt_gb(avail_page)}")
                if free_vram_mb is not None and free_vram_mb < lane_vram:
                    trips.append(f"free VRAM down to {free_vram_mb / 1024.0:.1f} GB")

                held = [lane for lane in live_lanes if self.lane_paused(lane)]

                if held:
                    # Every metric has to be back above its trip point plus the
                    # margin, for three consecutive samples, before a held lane
                    # is thawed. One good sample is noise - the other lane
                    # between two allocations.
                    clear = (
                        (not avail_ram or avail_ram >= lane_ram + margin)
                        and (not avail_page or avail_page >= lane_commit + margin)
                        and (free_vram_mb is None or free_vram_mb >= lane_vram + 800)
                    )
                    lane_resume_streak = lane_resume_streak + 1 if clear else 0
                    lane = held[-1]
                    waited = self.lane_held_seconds(lane)
                    if lane_resume_streak >= 3:
                        lane_resume_streak = 0
                        thawed = self.resume_lane(lane)
                        self._log(
                            f"[Guard] Memory is back (RAM {fmt_gb(avail_ram)}, commit "
                            f"{fmt_gb(avail_page)}"
                            + (f", VRAM {free_vram_mb / 1024.0:.1f} GB free"
                               if free_vram_mb is not None else "")
                            + f") - releasing the {lane} lane after "
                              f"{fmt_duration(waited)}; {thawed} process(es) running "
                              f"again. Nothing was lost and the other lane never "
                              f"stopped.")
                        continue
                    if waited >= lane_max_hold:
                        reason = (f"held for {fmt_duration(waited)} waiting for memory "
                                  f"that never came back ({', '.join(trips) or 'still short'})")
                        killed = self.kill_lane(lane, reason)
                        self._log(
                            f"[Guard] The {lane} lane has been frozen for "
                            f"{fmt_duration(waited)} and the memory has not come back. "
                            f"Killing that lane only ({killed} process(es)) - the rest of "
                            f"the run keeps going, and the {lane} eye is re-run on its own "
                            f"once the other one finishes. This machine could not hold two "
                            f"lanes at these settings.")
                        continue
                    if time.time() - last_lane_report > 60:
                        last_lane_report = time.time()
                        self._log(f"[Guard] The {lane} lane is still frozen after "
                                  f"{fmt_duration(waited)} ({', '.join(trips) or 'watching'}); "
                                  f"the other lane is still running.")
                    if not trips:
                        continue
                    # Still tripped with a lane already held: nothing more to
                    # take here, so fall through to the whole-job path below.
                elif trips:
                    # Hold the lane registered last. The older lane is the one
                    # closest to finishing, and finishing is what actually gives
                    # its memory back.
                    lane = live_lanes[-1]
                    lane_resume_streak = 0
                    before = self.lane_memory_now(lane)
                    frozen = self.pause_lane(lane, by="guard")
                    if frozen:
                        trimmed = sum(1 for pid in self.lane_pids(lane)
                                      if empty_working_set(pid))
                        time.sleep(0.5)
                        recovered = max(0, before - self.lane_memory_now(lane))
                        self._log(
                            f"[Guard] {', '.join(trips)} - holding the {lane} lane; the "
                            f"other lane keeps running. Froze {frozen} process(es), "
                            f"trimmed {trimmed} and handed back {fmt_gb(recovered)}. It "
                            f"resumes by itself once there is "
                            f"{fmt_gb(margin)} of headroom again.")
                        last_lane_report = time.time()
                        continue
                    # Nothing froze, so the lane is already gone. Fall through.
                else:
                    lane_resume_streak = 0

            # Freeze before killing. Low free RAM is the one trip point a
            # freeze can actually fix - it hands this run's resident pages
            # back - and it fixes it just as well from below the abort line as
            # from above. So the freeze is tried whenever RAM is short, not
            # only while RAM is still inside the pause band. Free RAM does not
            # drift down through that band when the machine hits its commit
            # ceiling: Windows trims every working set at once and the next
            # sample is already past the abort line. Gating the freeze on "no
            # problems yet" made it unreachable in exactly the case it exists
            # for.
            #
            # A commit or disk shortage is different - a freeze returns neither
            # committed address space nor disk - so if either of those is also
            # tripped there is nothing to try and the strike path below is the
            # honest answer.
            other = [p for p in problems if not p.startswith("free RAM")]
            if (auto_pause and avail_ram and avail_ram < pause_ram and not other
                    and time.time() - last_freeze_fail > 60):
                holds += 1
                if problems:
                    self._log(f"[Guard] Free RAM down to {fmt_gb(avail_ram)}, already "
                              f"past the {fmt_gb(min_ram)} abort line - FREEZING the "
                              f"run rather than killing it.")
                else:
                    self._log(f"[Guard] Free RAM down to {fmt_gb(avail_ram)} - FREEZING "
                              f"the run instead of letting it reach the "
                              f"{fmt_gb(min_ram)} abort line.")
                for line in self.explain_memory_pressure():
                    self._log(line)
                if self.pause(by="guard"):
                    trimmed, recovered = self.free_memory_now()
                    self._log(f"[Guard] Froze {trimmed} process(es) and handed back "
                              f"{fmt_gb(recovered)}; free RAM is now "
                              f"{fmt_gb(memory_status()[1])}. The run resumes by "
                              f"itself once {fmt_gb(resume_ram)} is free.")
                    last_hold_report = time.time()
                    continue
                # Nothing froze, so there is no live process holding pages to
                # give back and the freeze cannot help. Fall through to the
                # abort line, and do not re-announce this every three seconds.
                last_freeze_fail = time.time()
                self._log("[Guard] Nothing could be frozen, so freezing cannot help "
                          "here - the watchdog is back to watching the abort line.")

            if problems:
                strikes += 1
                key = tuple(p.split(" down")[0] for p in problems)
                if key not in warned:
                    warned.add(key)
                    self._log(f"[Guard] WARNING: {', '.join(problems)}")
                    if any(p.startswith("free RAM") for p in problems):
                        for line in self.explain_memory_pressure():
                            self._log(line)
                if strikes >= max_strikes:
                    # Worked out before the kill, because afterwards every
                    # number that would explain it is gone.
                    culprit = ""
                    if any(p.startswith("free RAM") for p in problems):
                        for line in self.explain_memory_pressure():
                            self._log(line)
                        mine = self.job_memory_now()
                        total, avail, _ = memory_status()
                        in_use = max(0, total - avail)
                        if mine and in_use and mine < in_use * 0.5:
                            culprit = (f" This run's own processes were only using "
                                       f"{fmt_gb(mine)} of the {fmt_gb(in_use)} in "
                                       f"use, so the memory went somewhere else on "
                                       f"the machine - see the log for what.")
                    reason = (
                        "The machine was running out of resources ("
                        + ", ".join(problems)
                        + "). The run was stopped to keep Windows responsive."
                        + culprit
                    )
                    # If free RAM is what tripped this and a pagefile is on a
                    # mechanical disk, that is almost certainly the actual
                    # cause: the run's own footprint is steady, and what
                    # collapses is Windows' ability to get paged-out memory
                    # back fast enough. Say so here rather than leaving the
                    # user to conclude they need more RAM.
                    if any(p.startswith("free RAM") for p in problems):
                        paging = slow_pagefile_warning()
                        if paging:
                            reason += "\n\n" + paging
                    self.abort(reason)
                    return
            else:
                strikes = 0

            # Same idea for the job's own committed-memory ceiling. Nothing can
            # be done about it mid-run, but it turns a later "FaceFusion failed"
            # into something the log already explains.
            cap = self.memory_cap_bytes()
            if cap:
                peak = self.peak_memory_bytes()
                if peak > cap * 0.85 and "job-memory" not in warned:
                    warned.add("job-memory")
                    self._log(
                        f"[Guard] Pipeline memory is close to its ceiling: "
                        f"{fmt_gb(peak)} committed of a {fmt_gb(cap)} cap. If the run dies "
                        f"from here, raise job_memory_cap_gb in scripts/guardrails.json "
                        f"or use a lighter quality preset."
                    )

            # Early warning well before the trip point.
            if free_disk and free_disk < min_disk * 2.5 and "disk-soon" not in warned:
                warned.add("disk-soon")
                self._log(
                    f"[Guard] Temp disk getting low: {fmt_gb(free_disk)} free. "
                    f"Run will stop automatically at {fmt_gb(min_disk)}."
                )


# ---------------------------------------------------------------------------
# Fallback for calls made outside a guard
# ---------------------------------------------------------------------------

class _NullGuard:
    """Used when no guard is active, so engine code has one call shape."""

    abort_reason = None

    def spawn(self, cmd, lane=None, **kwargs):
        # `lane` is accepted and ignored: without a job object there is no safe
        # way to find a lane's processes, so there is nothing to freeze.
        flags = kwargs.pop("creationflags", 0)
        if IS_WINDOWS:
            flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return subprocess.Popen(cmd, creationflags=flags, **kwargs)

    def run(self, cmd, timeout=None, lane=None, **kwargs):
        capture = kwargs.pop("capture_output", False)
        if capture:
            kwargs.setdefault("stdout", subprocess.PIPE)
            kwargs.setdefault("stderr", subprocess.PIPE)
        proc = self.spawn(cmd, lane=lane, **kwargs)
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)

    def check(self):
        return None

    def cancel(self, reason="Cancelled"):
        return None

    def kill_all(self):
        return None

    def min_available_commit_bytes(self):
        return 0

    # The pause API, so engine code never has to ask whether it has a real
    # guard before consulting the pause clock.
    paused = False
    paused_by = None

    def pause(self, by="user"):
        return False

    def resume(self):
        return False

    def free_memory_now(self):
        return 0, 0

    def paused_seconds(self):
        return 0.0

    def job_memory_now(self):
        return 0

    def job_pids(self):
        return []

    def explain_memory_pressure(self):
        return []

    def memory_cap_bytes(self):
        return 0

    def hit_memory_cap(self):
        return False

    def peak_memory_bytes(self):
        return 0

    # The lane API, so engine code never has to ask whether it has a real guard
    # before naming a lane. Nothing here can freeze anything - see spawn().
    lane_failed = {}

    def lanes(self):
        return []

    def live_lanes(self):
        return []

    def lane_pids(self, lane):
        return []

    def lane_paused(self, lane):
        return False

    def lane_memory_now(self, lane):
        return 0

    def lane_held_seconds(self, lane):
        return 0.0

    def pause_lane(self, lane, by="guard"):
        return 0

    def resume_lane(self, lane):
        return 0

    def kill_lane(self, lane, reason=""):
        return 0


NULL_GUARD = _NullGuard()
