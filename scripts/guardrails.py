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


def fmt_gb(num_bytes):
    return f"{num_bytes / GB:.1f} GB"


# ---------------------------------------------------------------------------
# Guard configuration
# ---------------------------------------------------------------------------

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
    job_commit_cap = min(commit_limit * 0.5, max(commit_limit - 16 * GB, 4 * GB))
    # Never below physical RAM: a cap under that is tighter than the machine.
    job_commit_cap = int(max(job_commit_cap, min(total_ram, commit_limit)))

    return {
        "enabled": True,

        # Hard ceiling on committed memory for ALL pipeline children combined.
        # If they try to exceed it their allocation fails and they die -
        # instead of the machine dying.
        "job_memory_cap_gb": round(job_commit_cap / GB, 1),

        # Hard CPU ceiling for the job, percent of total machine CPU.
        # Keeps the shell and UI schedulable during long x264/x265 encodes.
        "cpu_percent_cap": 75,

        # Children run below normal so nothing they do can starve the desktop.
        "below_normal_priority": True,

        # Watchdog trip points. Three consecutive strikes aborts the run.
        "watchdog_enabled": True,
        "watchdog_interval_sec": 3.0,
        "watchdog_strikes": 3,
        "min_free_ram_gb": 2.5,
        "min_free_disk_gb": 12.0,
        "min_free_pagefile_gb": 3.0,

        # Preflight refuses to start a run that cannot plausibly fit.
        "preflight_enabled": True,
        "disk_reserve_gb": 15.0,

        # Safe processing caps.
        "max_execution_threads": max(2, min(4, cores // 4)),
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

        "_machine": {
            "total_ram_gb": round(total_ram / GB, 1),
            "logical_cores": cores,
            "vram_total_mb": vram_total_mb,
        },
    }


def load_guard_config(path=None):
    """Load guardrails.json, writing defaults on first run."""
    path = path or GUARD_CONFIG_PATH
    config = default_guard_config()
    if os.path.exists(path):
        try:
            with open(path) as handle:
                config.update(json.load(handle))
        except Exception:
            pass
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
                       pipeline_config=None):
    """
    Estimate peak temp-disk usage for a run.

    This is what actually kills the machine, so it is worth being explicit:
    FaceFusion extracts every frame to disk, keeps the processed frames
    alongside the extracted ones, and VR mode does that twice (one pass per
    eye) on top of two intermediate split files.
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

    # In VR each pass works on one eye, i.e. half the source area.
    pass_pixels = probe["width"] * probe["height"] / (2.0 if is_vr else 1.0)

    # Extracted frames + processed frames live on disk at the same time.
    frames_bytes = frames * pass_pixels * bpp * 2

    # Intermediate encodes: 2 split files for VR, plus the merged result.
    intermediate = frames * pass_pixels * INTERMEDIATE_BPP
    if is_vr:
        intermediate *= 2
        intermediate += frames * probe["width"] * probe["height"] * INTERMEDIATE_BPP

    # The upscaled result. It is the largest single file the run produces and
    # was previously left out entirely, which under-estimated exactly the runs
    # most able to fill the drive: at scale 2 it is 4x the source area, at 4 it
    # is 16x. It lands in output/final rather than the temp folder, so this is
    # only the same drive in the usual layout - erring high is the safe side.
    upscale_bytes = 0.0
    upscale_config = pipeline_config or {}
    if upscale_config.get("upscale_method"):
        target_width = upscale_config.get("upscale_target_width")
        if target_width:
            factor = (float(target_width) / probe["width"]) ** 2
        else:
            factor = float(upscale_config.get("upscale_scale", 2) or 1) ** 2
        upscale_bytes = frames * probe["width"] * probe["height"] * factor * INTERMEDIATE_BPP

    total = frames_bytes + intermediate + upscale_bytes
    return {
        "bytes": int(total),
        "upscale_bytes": int(upscale_bytes),
        "frames": int(frames),
        "duration": duration,
        "width": probe["width"],
        "height": probe["height"],
        "fps": probe["fps"],
        "frame_format": fmt,
    }


def estimate_vram_mb(pipeline_config):
    """Very rough resident-VRAM estimate for the selected model stack."""
    total = 900  # CUDA context + detector + recognizer baseline

    swapper = (pipeline_config.get("swapper_model") or "").lower()
    if "512" in swapper:
        total += 2200
    elif "256" in swapper:
        total += 1200
    else:
        total += 900

    boost = str(pipeline_config.get("pixel_boost") or "512x512")
    try:
        side = int(boost.split("x")[0])
    except (ValueError, IndexError):
        side = 512
    total += int((side / 512.0) ** 2 * 700)

    enhancer = (pipeline_config.get("enhancer_model") or "").lower()
    if "2048" in enhancer:
        total += 3400
    elif "1024" in enhancer:
        total += 1600
    elif enhancer and enhancer != "none":
        total += 900

    frame_enhancer = (pipeline_config.get("frame_enhancer") or "").lower()
    if frame_enhancer and frame_enhancer != "none":
        # Frame enhancers run on the full frame, not a 512px crop.
        total += 4200

    threads = int(pipeline_config.get("threads") or 1)
    # Each extra in-flight frame carries its own working set.
    total = int(total * (1.0 + 0.35 * max(0, threads - 1)))
    return total


def estimate_commit_mb(pipeline_config, frame_pixels):
    """
    Rough peak COMMITTED host memory for one processing pass, in MB.

    This is the number the Job Object's memory cap is measured against, so it
    is the one that decides whether a run survives - and it is dominated by
    address space the CUDA runtime reserves rather than by anything resident.
    The constants are fitted to measurements on this machine (RTX 5060 Ti,
    Normal mode, 4K, 3 execution threads, peak job commit as reported by
    ResourceGuard.peak_memory_bytes):

        pb256 + gpen_bfr_512  ("Good")   11.1 GB short run, model says 13.5
        pb512 + gpen_bfr_1024 ("Best")   19.5 GB on a 60s clip, model says 19.4

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

    enhancer = (pipeline_config.get("enhancer_model") or "").lower()
    if "2048" in enhancer:
        total += 7000
    elif "1024" in enhancer:
        total += 3600
    elif "512" in enhancer:
        total += 1200
    elif enhancer and enhancer != "none":
        total += 900

    # Frame enhancers work on the whole frame, so they scale with resolution.
    frame_enhancer = (pipeline_config.get("frame_enhancer") or "").lower()
    if frame_enhancer and frame_enhancer != "none":
        total += int(mpix * 900)

    # Each in-flight frame carries its own buffers through the processor chain.
    threads = int(pipeline_config.get("threads") or 1)
    total += int(threads * mpix * 100)

    # FaceFusion's own ffmpeg children: two readers and a writer, all in the
    # job and all holding raw frames.
    total += int(3 * (mpix * 130 + 250))

    return total


def preflight(target_video, temp_dir, pipeline_config, guard_config,
              is_vr=False, trim_start=None, trim_end=None):
    """
    Decide whether this run is safe to start.

    Returns (blocking_problems, warnings, estimate). A non-empty
    blocking_problems list means: do not start.
    """
    blocking = []
    warnings = []

    estimate = estimate_disk_need(target_video, is_vr, guard_config, trim_start, trim_end,
                                  pipeline_config=pipeline_config)
    if not guard_config.get("preflight_enabled", True):
        return ([], ["Preflight checks are disabled in guardrails.json."], estimate)

    free_disk = free_disk_bytes(temp_dir)
    reserve = guard_config.get("disk_reserve_gb", 15.0) * GB
    needed = estimate["bytes"] + reserve

    estimate["free_disk"] = free_disk
    estimate["needed"] = needed

    if free_disk and needed > free_disk:
        blocking.append(
            f"Not enough temp disk space on {os.path.splitdrive(os.path.abspath(temp_dir))[0] or 'the temp drive'}. "
            f"This run needs about {fmt_gb(estimate['bytes'])} of temp files "
            f"(+{guard_config.get('disk_reserve_gb', 15.0):.0f} GB reserved for Windows) "
            f"but only {fmt_gb(free_disk)} is free.\n"
            f"    {estimate['frames']} frames at {estimate['width']}x{estimate['height']} "
            f"({estimate['duration']:.0f}s @ {estimate['fps']:.0f} fps).\n"
            f"    Fix: trim the clip, point the temp folder at a bigger drive, or free up space."
        )
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
            blocking.append(
                f"The selected models need roughly {vram_need_mb} MB of VRAM but the GPU has "
                f"{vram_total_mb} MB. Overcommitting VRAM makes Windows page GPU memory into "
                f"system RAM, which is what freezes the machine.\n"
                f"    Fix: lower Pixel Boost, use a smaller face enhancer, turn the frame "
                f"enhancer off, or reduce Threads."
            )
        elif vram_need_mb > vram_free_mb * 0.85:
            warnings.append(
                f"VRAM headroom is small: ~{vram_need_mb} MB needed, {vram_free_mb} MB free "
                f"of {vram_total_mb} MB."
            )

    # Host commit vs the Job Object cap. This is the limit that actually stops
    # long runs: when the job's committed memory hits the cap, every further
    # allocation inside it fails and FaceFusion dies mid-run, typically an hour
    # in. Catch it here instead.
    pass_pixels = estimate["width"] * estimate["height"] / (2.0 if is_vr else 1.0)
    commit_need_mb = estimate_commit_mb(pipeline_config, pass_pixels)
    cap_gb = float(guard_config.get("job_memory_cap_gb", 0) or 0)
    estimate["commit_need_mb"] = commit_need_mb
    estimate["commit_cap_mb"] = int(cap_gb * 1024)

    if cap_gb > 0:
        cap_mb = cap_gb * 1024
        if commit_need_mb > cap_mb * 0.95:
            blocking.append(
                f"These settings need roughly {commit_need_mb / 1024:.1f} GB of committed memory "
                f"but the guard rails cap the whole pipeline at {cap_gb:.1f} GB. The run would "
                f"start normally and then die part-way through with an allocation failure.\n"
                f"    Note this is *committed* memory, not RAM in use - a CUDA process reserves "
                f"far more than it makes resident.\n"
                f"    Fix: raise job_memory_cap_gb in scripts/guardrails.json (this machine's "
                f"commit limit is {fmt_gb(commit_limit_bytes())}), pick a lighter quality preset, "
                f"or reduce Threads."
            )
        elif commit_need_mb > cap_mb * 0.8:
            warnings.append(
                f"Memory headroom is small: ~{commit_need_mb / 1024:.1f} GB of committed memory "
                f"needed against a {cap_gb:.1f} GB cap."
            )

    total_ram, avail_ram, avail_page = memory_status()
    if avail_ram and avail_ram < guard_config.get("min_free_ram_gb", 2.5) * GB * 1.5:
        warnings.append(
            f"Only {fmt_gb(avail_ram)} of RAM is free right now. Close other apps before starting."
        )

    return (blocking, warnings, estimate)


def clamp_config(pipeline_config, guard_config, log_cb=None):
    """
    Rewrite unsafe pipeline settings into safe ones.

    Returns a new dict; the caller's config is not mutated.
    """
    config = dict(pipeline_config)
    changes = []

    max_threads = int(guard_config.get("max_execution_threads", 4))
    threads = int(config.get("threads") or 1)
    if threads > max_threads:
        config["threads"] = max_threads
        changes.append(f"threads {threads} -> {max_threads} (each thread holds a full frame in VRAM)")

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

    def start(self):
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

        cpu_cap = int(self.config.get("cpu_percent_cap", 0) or 0)
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

        self._log(
            f"[Guard] Active - memory cap {cap_gb:.1f} GB, "
            f"CPU cap {cpu_cap or 100}%, "
            f"priority {'below normal' if self.config.get('below_normal_priority', True) else 'normal'}"
        )

        self._start_watchdog()
        return self

    def close(self):
        """Kill anything still running and release the job."""
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
            # KILL_ON_JOB_CLOSE means this also reaps every surviving child.
            _kernel32.CloseHandle(self._job)
            self._job = None

        self._reap_stragglers()
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

    def spawn(self, cmd, **kwargs):
        """subprocess.Popen, with the child adopted into the guarded job."""
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

    def run(self, cmd, timeout=None, **kwargs):
        """subprocess.run equivalent that goes through the job object."""
        capture = kwargs.pop("capture_output", False)
        if capture:
            kwargs.setdefault("stdout", subprocess.PIPE)
            kwargs.setdefault("stderr", subprocess.PIPE)

        proc = self.spawn(cmd, **kwargs)
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

    def _reap_stragglers(self):
        """Belt-and-braces kill for anything the job did not cover."""
        with self._lock:
            procs = list(self._procs)
            self._procs.clear()
        for proc in procs:
            if proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass

    def cancel(self, reason="Cancelled by user"):
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

        strikes = 0
        warned = set()

        while not self._watchdog_stop.wait(interval):
            if self.abort_reason:
                return

            _, avail_ram, avail_page = memory_status()
            free_disk = free_disk_bytes(self.temp_dir)

            problems = []
            if avail_ram and avail_ram < min_ram:
                problems.append(f"free RAM down to {fmt_gb(avail_ram)}")
            if free_disk and free_disk < min_disk:
                problems.append(f"free temp disk down to {fmt_gb(free_disk)}")
            if avail_page and avail_page < min_page:
                problems.append(f"free commit charge down to {fmt_gb(avail_page)}")

            if problems:
                strikes += 1
                key = tuple(p.split(" down")[0] for p in problems)
                if key not in warned:
                    warned.add(key)
                    self._log(f"[Guard] WARNING: {', '.join(problems)}")
                if strikes >= max_strikes:
                    self.abort(
                        "The machine was running out of resources ("
                        + ", ".join(problems)
                        + "). The run was stopped to keep Windows responsive."
                    )
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

    def spawn(self, cmd, **kwargs):
        flags = kwargs.pop("creationflags", 0)
        if IS_WINDOWS:
            flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return subprocess.Popen(cmd, creationflags=flags, **kwargs)

    def run(self, cmd, timeout=None, **kwargs):
        capture = kwargs.pop("capture_output", False)
        if capture:
            kwargs.setdefault("stdout", subprocess.PIPE)
            kwargs.setdefault("stderr", subprocess.PIPE)
        proc = self.spawn(cmd, **kwargs)
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)

    def check(self):
        return None

    def cancel(self, reason="Cancelled"):
        return None

    def kill_all(self):
        return None

    def memory_cap_bytes(self):
        return 0

    def hit_memory_cap(self):
        return False

    def peak_memory_bytes(self):
        return 0


NULL_GUARD = _NullGuard()
