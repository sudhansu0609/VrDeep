import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
import os
import sys
import gc
import copy
import json
import time
import threading
from datetime import datetime, timedelta
from types import SimpleNamespace


def attach_console_log():
    """
    Give the app somewhere to write when it was started without a console.

    The launchers use pythonw.exe so that the desktop app is the only window on
    screen, and pythonw leaves sys.stdout and sys.stderr as None. print() is a
    no-op then, which is harmless, but anything reaching for sys.stderr.write -
    Tk's own exception reporter, a traceback from a worker thread, a library
    warning - raises there instead, and the failure it was trying to report is
    lost with it. A file keeps them, and keeps them off the user's screen,
    which was the point of the windowless launcher.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        os.makedirs(os.path.join(project_dir, "output"), exist_ok=True)
        stream = open(os.path.join(project_dir, "output", "app.log"), "a",
                      encoding="utf-8", errors="replace", buffering=1)
        stream.write("\n=== VrDeep started "
                     + time.strftime("%Y-%m-%d %H:%M:%S") + " ===\n")
    except Exception:
        # A read-only project folder is not a reason to refuse to start.
        stream = open(os.devnull, "w")
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


attach_console_log()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guardrails
import vr_metadata
import upscale_vr
import clip_selector
from guardrails import ResourceGuard
from engine import (
    run_chunked_pipeline, load_config, prepare_run, chunk_plan, chunk_dir_for, probe_video,
    split_sources, is_video_path, is_image_path, describe_run_state,
    vr_stage_keys, quality_to_cq,
    PROJECT_DIR, DEFAULT_CONFIG, PipelineError
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

APP_WIDTH = 1200
APP_HEIGHT = 850

# ---------------------------------------------------------------------------
# Session memory
#
# Every field on this window used to be empty again at the next launch, so
# picking a stopped job back up meant re-browsing to the source, re-browsing to
# the target and retyping the in and out points. That is not merely tedious:
# the target file and the clip range are part of the resume key, so a start
# time typed back as 61 where it had been 60.5 quietly invalidated a finished
# eye and re-swapped it for an hour. Remembering the fields is what makes
# "Resume from where the last attempt stopped" actually resume.
#
# It lives in output/ beside run_state.json, which is the other half of the
# same memory: this file is what the user chose, that one is what was built.
# ---------------------------------------------------------------------------

SESSION_PATH = os.path.join(PROJECT_DIR, "output", "session.json")

# ---------------------------------------------------------------------------
# Job queue
#
# One run at a time is all this machine can do - a second FaceFusion would
# fight the first for the same 16 GB of VRAM - but one run at a time is not
# the same as one run per sitting. The queue is a list of jobs, each a frozen
# copy of a tab's fields plus the settings in force when it was added, run in
# order, each one no earlier than the time written against it. It lives in
# output/queue.json so that a night's worth of jobs survives a restart.
# ---------------------------------------------------------------------------

QUEUE_PATH = os.path.join(PROJECT_DIR, "output", "queue.json")

# Which tab variable each job field is read from and written back to. The
# upscale quality slider is not a variable and is handled by hand.
JOB_FIELD_VARS = {
    "vr": {
        "preset": "vr_preset_var", "source": "vr_source_var", "target": "vr_target_var",
        "output": "vr_output_var", "layout": "vr_layout_var",
        "trim_start": "vr_trim_start_var", "trim_end": "vr_trim_end_var",
        "fps": "vr_fps_var", "keep_temp": "vr_keep_temp_var", "resume": "vr_resume_var",
        "refresh_faces": "vr_refresh_faces_var",
    },
    "normal": {
        "preset": "nm_preset_var", "source": "nm_source_var", "target": "nm_target_var",
        "output": "nm_output_var",
        "trim_start": "nm_trim_start_var", "trim_end": "nm_trim_end_var",
        "fps": "nm_fps_var", "keep_temp": "nm_keep_temp_var",
        "refresh_faces": "nm_refresh_faces_var",
    },
    "upscale": {
        "input": "up_input_var", "output": "up_output_var",
        "trim_start": "up_trim_start_var", "trim_end": "up_trim_end_var",
        "target": "up_target_var", "layout": "up_layout_var", "method": "up_method_var",
        "headset": "up_headset_var", "encoder": "up_encoder_var",
        "split_eyes": "up_split_eyes_var", "allow_sw": "up_allow_sw_var",
    },
}

JOB_SCHEDULE_VARS = {"vr": "vr_schedule_var", "normal": "nm_schedule_var",
                     "upscale": "up_schedule_var"}
JOB_TABS = {"vr": "VR Mode", "normal": "Normal Mode", "upscale": "Upscale"}
JOB_HEADINGS = {"vr": "VR PIPELINE", "normal": "NORMAL PIPELINE", "upscale": "UPSCALE"}
JOB_DONE_TEXT = {"vr": "VR Pipeline complete!", "normal": "Normal Pipeline complete!",
                 "upscale": "Upscale complete!"}

JOB_STATUS_COLORS = {
    "waiting": "#60a5fa", "running": "#4ade80", "done": "#22c55e",
    "failed": "#ef4444", "stopped": "#f59e0b", "skipped": "#888888",
    # Amber, like "stopped": the job has not failed and has not been abandoned,
    # it is holding for memory somebody else's program is using.
    "deferred": "#f59e0b",
}

# Statuses that still have a run ahead of them. A deferred job is a waiting job
# that has already been refused once for memory, so everywhere the queue asks
# "is there anything left to do" the answer has to include it - otherwise the
# queue reports itself finished with jobs still on the list.
PENDING_STATUSES = ("waiting", "deferred")
NOT_BEFORE_FMT = "%Y-%m-%d %H:%M:%S"


def job_title(job):
    """One line that says what a job is, for the queue list and the log."""
    f = job["fields"]
    if job["kind"] == "upscale":
        return (f"Upscale to {f.get('target', '')}: "
                f"{os.path.basename(f.get('input', '')) or '(no input)'}")
    what = "VR swap" if job["kind"] == "vr" else "2D swap"
    return f"{what}: {os.path.basename(f.get('target', '')) or '(no target)'}"


def job_detail(job):
    """The second line: the portion, the output name, and what happened to it."""
    f = job["fields"]
    start = clip_selector.parse_timecode(f.get("trim_start") or "")
    end = clip_selector.parse_timecode(f.get("trim_end") or "")
    if start is None and end is None:
        portion = "whole video"
    else:
        portion = (f"{clip_selector.format_timecode(start) if start is not None else 'start'}"
                   f" to {clip_selector.format_timecode(end) if end is not None else 'end'}")
    bits = [portion]
    if job["kind"] != "upscale" and f.get("preset"):
        bits.append(f.get("preset").split(" (")[0])
    bits.append("-> " + (os.path.basename(f.get("output", "")) or "(no output)"))
    if job.get("elapsed"):
        bits.append(guardrails.fmt_duration(job["elapsed"]))
    if job.get("note"):
        bits.append(job["note"])
    return "  |  ".join(bits)


def job_not_before(job):
    text = job.get("not_before")
    if not text:
        return None
    try:
        return datetime.strptime(text, NOT_BEFORE_FMT)
    except ValueError:
        return None


def _job_time(job, key):
    """One of the job's datetime strings, or None when absent or unreadable.

    Absent is the normal case for a queue.json written before deferral
    existed, so it is not worth a complaint - a job with no retry time is a
    job that has never been deferred.
    """
    text = job.get(key)
    if not text:
        return None
    try:
        return datetime.strptime(text, NOT_BEFORE_FMT)
    except (TypeError, ValueError):
        return None


def job_retry_after(job):
    """When a deferred job may be tried again, or None."""
    return _job_time(job, "retry_after")


def job_ready_at(job):
    """
    The earliest this job may start: its scheduled time, or its retry time.

    A deferred job keeps whatever start time it was given - a 02:00 job that
    defers at 02:00 must not become eligible at 01:55 on the next tick - so the
    later of the two is what the queue waits for.
    """
    when = job_not_before(job)
    if job.get("status") == "deferred":
        retry = job_retry_after(job)
        if retry is not None and (when is None or retry > when):
            return retry
    return when

# What the "Start At" boxes accept. Written out rather than left to a docstring
# because it is also the hint shown under the box.
SCHEDULE_HINT = "empty = start now  |  02:30  |  tomorrow 02:30  |  +90m  |  +2h30m"


def parse_schedule_time(text, now=None):
    """
    "02:30" / "tomorrow 02:30" / "+90m" / "+2h30m" / "2026-09-06 02:30" -> datetime.

    None for an empty box. Raises ValueError, with a sentence worth showing, for
    anything else. A bare clock time that has already gone today means tomorrow:
    someone typing 02:30 at midnight means tonight, and someone typing it at
    03:00 means the next one.
    """
    text = (text or "").strip()
    if not text:
        return None
    now = now or datetime.now()

    if text.startswith("+"):
        rest = text[1:].strip().lower().replace(" ", "")
        if not rest:
            raise ValueError("'+' needs a length after it, like +90m or +2h30m.")
        minutes, number, seen = 0.0, "", False
        for char in rest:
            if char.isdigit() or char == ".":
                number += char
            elif char in "hm":
                if not number:
                    raise ValueError(f"'{text}' has a '{char}' with no number before it.")
                minutes += float(number) * (60 if char == "h" else 1)
                number, seen = "", True
            else:
                raise ValueError(f"'{text}' is not a length - use +90m or +2h30m.")
        if number:
            # A bare "+45" is minutes, which is what anyone typing it means.
            minutes += float(number)
        elif not seen:
            raise ValueError(f"'{text}' is not a length - use +90m or +2h30m.")
        if minutes <= 0:
            raise ValueError("A scheduled start has to be in the future.")
        return now + timedelta(minutes=minutes)

    tomorrow = False
    lowered = text.lower()
    for word in ("tomorrow", "tmr"):
        if lowered.startswith(word):
            tomorrow = True
            text = text[len(word):].strip()
            break

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass

    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            clock = datetime.strptime(text, fmt).time()
        except ValueError:
            continue
        when = now.replace(hour=clock.hour, minute=clock.minute,
                           second=clock.second, microsecond=0)
        if tomorrow or when <= now:
            when += timedelta(days=1)
        return when

    raise ValueError(f"'{text}' is not a time. Use {SCHEDULE_HINT}.")

# Deliberately excludes vr/nm_refresh_faces_var. Re-extract is a one-shot that
# the end of every run already clears, and remembering it would re-run the
# source analysis on every launch - exactly the cost the extraction cache
# exists to avoid.
SESSION_VARS = (
    "vr_preset_var", "vr_source_var", "vr_target_var", "vr_layout_var",
    "vr_output_var", "vr_trim_start_var", "vr_trim_end_var", "vr_fps_var",
    "vr_keep_temp_var", "vr_resume_var", "vr_schedule_var",
    "nm_preset_var", "nm_source_var", "nm_target_var", "nm_output_var",
    "nm_trim_start_var", "nm_trim_end_var", "nm_fps_var", "nm_keep_temp_var",
    "nm_schedule_var",
    "up_input_var", "up_output_var", "up_trim_start_var", "up_trim_end_var",
    "up_target_var", "up_layout_var", "up_method_var", "up_headset_var",
    "up_encoder_var", "up_split_eyes_var", "up_allow_sw_var", "up_schedule_var",
)

# The log pane and its backing list both grew for the lifetime of the session,
# and a Tk text widget holds every line it has ever been given. Nothing here is
# large per line, but the app is meant to sit open across many runs.
MAX_LOG_LINES = 2000
LOG_TRIM_TO = 1500

# Slider-backed settings that must reach the FaceFusion CLI as integers.
INT_SETTINGS = {"enhancer_blend", "frame_enhancer_blend", "quality", "threads", "upscale_scale",
                "target_frame_amount", "reference_frame_number", "expression_restorer_factor",
                "source_video_faces", "source_video_max_samples", "source_video_min_face_px",
                # A frame multiplier of 2.0 is not a thing; it is 2 or it is 3.
                "interpolate_multi",
                # Whole seconds; 0 turns chunking off.
                "chunk_seconds"}

# Encoders offered on the Upscale tab. NVENC first because it is 10x the speed
# of software at these sizes; libx265/libx264 are there because NVENC refuses
# anything over 8192px, and a software encoder is the only way past that.
UPSCALE_ENCODERS = ["hevc_nvenc", "av1_nvenc", "h264_nvenc", "libx265", "libx264"]

# Entry-backed settings that must reach the worker as floats. Without this an
# entry falls through to the "store the string" branch, and "2.0" in the config
# file is not the same thing as 2.0 to the code that reads it back.
FLOAT_SETTINGS = {"source_video_sample_fps"}

# Settings shown on the Settings tab that belong to the GUARD RAILS, not to the
# pipeline: they live in scripts/guardrails.json and are read by the guard, not
# passed to FaceFusion. They are on the same tab because that is where a user
# looks for them, but they must be saved to the other file - writing
# parallel_eyes into pipeline_config.json would put it somewhere nothing reads.
GUARD_SETTINGS = {"parallel_eyes"}

# Slider ranges, and how many decimals the read-out beside them shows. Kept in
# one place so a range and its label cannot drift apart - a 0-1 weight rendered
# to one decimal reads 0.6 for both 0.55 and 0.64.
SLIDER_RANGES = {
    "enhancer_blend": (0, 100, 0),
    "expression_restorer_factor": (0, 100, 0),
    "enhancer_weight": (0.0, 1.0, 2),
    "frame_enhancer_blend": (0, 100, 0),
    "swapper_weight": (0.0, 1.0, 2),
    "detector_score": (0.1, 0.9, 2),
    "landmarker_score": (0.0, 1.0, 2),
    # FaceFusion's own accepted range is 0.0-0.5; 0 disables tracking entirely.
    "face_tracker_score": (0.0, 0.5, 2),
    "target_frame_amount": (0, 5, 0),
    "reference_face_distance": (0.0, 1.0, 2),
    "quality": (50, 100, 0),
    # Each in-flight frame carries its own model working set, so this
    # multiplies VRAM. The guard clamps it further.
    "threads": (1, 8, 0),
    "mask_blur": (0.1, 1.5, 2),
    "upscale_scale": (1, 4, 0),
    # 1 is off. Every step above 1 adds a whole synthesised frame per source
    # frame per eye, at roughly a second each on a 4096x4096 eye.
    "interpolate_multi": (1, 4, 0),
    # Fraction of the frame the RIFE model runs on. Below ~0.5 the synthesised
    # frames get visibly soft; at 1.0 the model needs 15.8 GB of a 16.3 GB card
    # and any other GPU user tips it into a ~50x slowdown.
    "interpolate_scale": (0.5, 1.0, 2),
    # The identity average converges after a handful of genuinely different
    # views; past that every extra still is a near-duplicate re-weighting a
    # view already in the average.
    "source_video_faces": (1, 12, 0),
    "source_video_identity_threshold": (0.2, 0.9, 2),
}

# Times are measured on this machine (RTX 5060 Ti 16 GB) against a deliberately
# hard case: 4K per eye, ~6 faces in every frame, and both eyes processed. They
# are an upper bound - single-subject footage runs several times faster, because
# almost all of the cost is per-face, not per-frame.
# The names used to carry a time - "Fast (~17 min/1min video)" and so on -
# and those numbers were a single face-swap pass. A VR run swaps each eye
# separately and then splits and merges 8192x4096 frames around them, so a
# minute of VR source really costs the better part of an hour and a half at
# 'Good', not thirty-one minutes. Rather than keep four numbers that are only
# right for the 2D tab, the names now say what the preset DOES; how long a
# particular run will take is worked out from the actual clip, resolution and
# settings when Run is pressed, and printed before anything starts.
QUALITY_PRESETS = {
    'Fast (swap only)': {
        'tooltip': 'Face swap only, no enhancement. Measured 3.5 frame/s.',
        'swapper_model': 'hyperswap_1a_256',
        'pixel_boost': '256x256',
        'enhancer_model': 'none',
        'frame_enhancer': 'none',
    },
    'Good (swap + 512 enhancer)': {
        'tooltip': 'Face swap + 512px enhancement. Best speed/quality balance; '
                   'measured 1.95 frame/s.',
        'swapper_model': 'hyperswap_1a_256',
        'pixel_boost': '256x256',
        'enhancer_model': 'gpen_bfr_512',
        'frame_enhancer': 'none',
    },
    'Better (512 pixel boost)': {
        'tooltip': 'Adds 512px pixel boost for more detail per face. '
                   'Measured 1.6 frame/s, so about 2x the run time of Good.',
        'swapper_model': 'hyperswap_1a_256',
        'pixel_boost': '512x512',
        'enhancer_model': 'gpen_bfr_512',
        'frame_enhancer': 'none',
    },
    'Best (512 boost + 1024 enhancer)': {
        'tooltip': '512px pixel boost + 1024px enhancement, and about 4x the '
                   'run time of Good. gpen_bfr_2048 is deliberately not used: '
                   'it costs ~2x again for detail a face a few hundred pixels '
                   'wide cannot carry.',
        'swapper_model': 'hyperswap_1a_256',
        'pixel_boost': '512x512',
        'enhancer_model': 'gpen_bfr_1024',
        'frame_enhancer': 'none',
    },
    'Custom (Settings Tab)': {
        'tooltip': 'Use Settings tab to configure manually.',
        'swapper_model': None,
        'pixel_boost': None,
        'enhancer_model': None,
        'frame_enhancer': None,
    },
}

# Referenced by both tabs; keep it out of the widget code so renaming a preset
# cannot leave a dangling key behind.
DEFAULT_PRESET = 'Good (swap + 512 enhancer)'
CUSTOM_PRESET = 'Custom (Settings Tab)'

# The settings a preset writes. Editing any of them in the Settings tab flips
# both preset dropdowns to Custom - without that, run_vr/run_normal re-apply
# the selected preset just before starting and the edit is silently discarded.
PRESET_KEYS = ('swapper_model', 'pixel_boost', 'enhancer_model', 'frame_enhancer')


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VrDeep - VR & Normal Deepfake Pipeline")
        self.geometry(f"{APP_WIDTH}x{APP_HEIGHT}+100+50")
        self.minsize(APP_WIDTH, APP_HEIGHT)

        self.config = DEFAULT_CONFIG.copy()
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_config.json")
        if os.path.exists(self.config_path):
            self.config = load_config(self.config_path)

        # The handful of Settings-tab controls that belong to the guard rails
        # (see GUARD_SETTINGS). Read from guardrails.json here so the widget
        # opens on the value that is actually in force, and written back there
        # by save_settings. They are deliberately NOT copied into a queued
        # job's frozen config: the guard rails describe the machine, and the
        # machine at 03:00 is the one that gets a say, not the machine as it
        # was when the job was added.
        self.guard_settings = {
            key: value for key, value in guardrails.load_guard_config().items()
            if key in GUARD_SETTINGS
        }

        self.is_running = False
        self.current_thread = None
        self._pause_baseline = 0.0
        # The job list, whether it is being worked through, the job in flight,
        # and whether the user asked for that job to be killed (so its failure
        # is reported as "stopped", not as a pipeline error).
        self.queue = []
        self.queue_active = False
        self._queue_after = None
        self._active_job = None
        self._active_interactive = True
        self._stop_requested = False
        # The Pause/Resume button on the running job's queue row, when one is
        # shown. Kept so _sync_pause_button can follow a guard-driven freeze.
        self._queue_row_pause_btn = None
        self.log_messages = []
        self.guard = None

        # gpu_memory_mb() shells out to nvidia-smi, which under a loaded machine
        # regularly takes hundreds of milliseconds. The resource read-out used to
        # call it from the UI timer, i.e. on the Tk thread, every 3 seconds - so
        # during a heavy run the window stopped repainting and Windows greyed it
        # out as "not responding". Sample it on a background thread instead and
        # let the UI read the last value.
        self._vram_used_mb = 0
        self._vram_stop = threading.Event()
        self._vram_thread = None

        self.setup_ui()
        # After the widgets exist and before anything is logged: the fields it
        # restores are what the resume report below is about.
        self.start_session_memory()
        self.load_queue()

        # Nothing must outlive the window. Closing used to leave FaceFusion and
        # ffmpeg running, still eating the machine with no UI left to stop them.
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.start_vram_sampler()
        self.log_machine_limits()
        self.log_resume_state()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        top_frame = ctk.CTkFrame(self, height=60)
        top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        top_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(top_frame, text="VrDeep", font=ctk.CTkFont(size=22, weight="bold")).grid(
            row=0, column=0, padx=(15, 0), sticky="w", pady=10
        )
        ctk.CTkLabel(top_frame, text="VR & Normal Deepfake Pipeline", font=ctk.CTkFont(size=13),
                     text_color="gray").grid(row=0, column=1, padx=(10, 0), sticky="w", pady=10)

        self.status_label = ctk.CTkLabel(top_frame, text="Ready", font=ctk.CTkFont(size=12),
                                         text_color="#4ade80")
        self.status_label.grid(row=0, column=2, padx=(0, 15), sticky="e", pady=10)

        main_container = ctk.CTkFrame(self)
        main_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        main_container.grid_columnconfigure(0, weight=1)
        main_container.grid_rowconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(main_container)
        self.tabview.add("VR Mode")
        self.tabview.add("Normal Mode")
        self.tabview.add("Upscale")
        self.tabview.add("Queue")
        self.tabview.add("Settings")
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.build_vr_tab()
        self.build_normal_tab()
        self.build_upscale_tab()
        self.build_queue_tab()
        self.build_settings_tab()

        bottom_frame = ctk.CTkFrame(self, height=200)
        bottom_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        bottom_frame.grid_columnconfigure(0, weight=1)
        bottom_frame.grid_rowconfigure(1, weight=1)

        self.progress_bar = ctk.CTkProgressBar(bottom_frame, mode="determinate")
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=10, pady=(5, 0))
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(bottom_frame, text="Progress: --", font=ctk.CTkFont(size=11))
        self.progress_label.grid(row=0, column=0, sticky="ew", padx=10, pady=(5, 0))

        header_row = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        header_row.grid(row=1, column=0, sticky="ew", padx=10, pady=(2, 0))
        header_row.grid_columnconfigure(0, weight=1)

        log_header = ctk.CTkLabel(header_row, text="Log Console", font=ctk.CTkFont(size=11, weight="bold"))
        log_header.grid(row=0, column=0, sticky="w")

        self.resource_label = ctk.CTkLabel(header_row, text="", font=ctk.CTkFont(size=10),
                                           text_color="#888888")
        self.resource_label.grid(row=0, column=1, sticky="e", padx=(0, 10))

        # Freezes the run where it stands instead of throwing away the stage
        # in flight. Next to Stop because they are the two answers to "I need
        # this machine back", and they cost very different things.
        self.pause_btn = ctk.CTkButton(
            header_row, text="PAUSE", width=90, height=26,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#a16207", hover_color="#ca8a04",
            command=self.toggle_pause, state="disabled"
        )
        self.pause_btn.grid(row=0, column=2, sticky="e", padx=(0, 8))

        # Always available, never disabled: if a run is misbehaving this is the
        # way out, and it kills the whole process tree rather than just asking.
        self.stop_btn = ctk.CTkButton(
            header_row, text="STOP", width=90, height=26,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#b91c1c", hover_color="#dc2626",
            command=self.stop_run, state="disabled"
        )
        self.stop_btn.grid(row=0, column=3, sticky="e")

        self.log_text = ctk.CTkTextbox(bottom_frame, font=ctk.CTkFont(family="Consolas", size=11),
                                       wrap="word", activate_scrollbars=True)
        self.log_text.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        bottom_frame.grid_rowconfigure(2, weight=1)

    def build_vr_tab(self):
        tab = self.tabview.tab("VR Mode")
        tab.grid_columnconfigure(1, weight=1)

        row = 0
        pad_y = 8
        pad_x = 15

        ctk.CTkLabel(tab, text="Quality Preset", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y
        )
        self.vr_preset_var = ctk.StringVar(value=DEFAULT_PRESET)
        self.vr_preset_combo = ctk.CTkComboBox(
            tab, values=list(QUALITY_PRESETS.keys()), variable=self.vr_preset_var,
            width=250, state="readonly", command=self.on_vr_preset_change
        )
        self.vr_preset_combo.grid(row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)
        self.vr_preset_tooltip = ctk.CTkLabel(
            tab, text=QUALITY_PRESETS[DEFAULT_PRESET]['tooltip'],
            font=ctk.CTkFont(size=10), text_color="#888888"
        )
        self.vr_preset_tooltip.grid(row=row, column=2, sticky="w", padx=(5, 0), pady=pad_y)

        row += 1
        ctk.CTkLabel(tab, text="Source Face", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y
        )
        self.vr_source_var = ctk.StringVar()
        self.vr_source_entry = ctk.CTkEntry(tab, textvariable=self.vr_source_var, width=350)
        self.vr_source_entry.grid(row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)
        ctk.CTkButton(tab, text="Browse", width=70, command=self.browse_vr_source).grid(
            row=row, column=2, padx=(5, 0), pady=pad_y
        )

        row += 1
        ctk.CTkLabel(
            tab, text="Photos, or one short video clip of the face.",
            font=ctk.CTkFont(size=10), text_color="#888888"
        ).grid(row=row, column=1, sticky="w", padx=pad_x)
        # Extractions are cached per clip and per settings, so the second run
        # against the same source is instant. This is the way to redo one after
        # editing the clip in place, where the cache key cannot tell.
        self.vr_refresh_faces_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(tab, text="Re-extract", variable=self.vr_refresh_faces_var,
                        font=ctk.CTkFont(size=10), checkbox_width=16, checkbox_height=16,
                        width=90).grid(row=row, column=2, sticky="w", padx=(5, 0))

        row += 1
        ctk.CTkLabel(tab, text="Target VR Video", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y
        )
        self.vr_target_var = ctk.StringVar()
        self.vr_target_entry = ctk.CTkEntry(tab, textvariable=self.vr_target_var, width=350)
        self.vr_target_entry.grid(row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)
        ctk.CTkButton(tab, text="Browse", width=70, command=self.browse_vr_target).grid(
            row=row, column=2, padx=(5, 0), pady=pad_y
        )

        row += 1
        ctk.CTkLabel(tab, text="Stereo Layout", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y
        )
        self.vr_layout_var = ctk.StringVar(value="sbs")
        layout_frame = ctk.CTkFrame(tab, width=200, height=36)
        layout_frame.grid(row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)
        layout_frame.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkRadioButton(layout_frame, text="Side-by-Side (SBS)", variable=self.vr_layout_var,
                           value="sbs").grid(row=0, column=0, padx=(10, 0), pady=5)
        ctk.CTkRadioButton(layout_frame, text="Top-Bottom (TB)", variable=self.vr_layout_var,
                           value="tb").grid(row=0, column=1, padx=(5, 0), pady=5)

        row += 1
        ctk.CTkLabel(tab, text="Output Path", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y
        )
        # The _LR_180 suffix is not decoration. DeoVR, Skybox and Pigasus pick
        # the layout off the filename before they read any metadata - which is
        # the only reason the untagged SLR source plays correctly in the first
        # place, since it carries no VR metadata either. The pipeline now also
        # writes proper st3d/sv3d boxes, but the suffix costs nothing and
        # covers the players that never look.
        self._vr_output_stem = os.path.join(
            PROJECT_DIR, "output", "final",
            f"vr_deepfake_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self._vr_output_auto = self._vr_output_stem + "_LR_180.mp4"
        self.vr_output_var = ctk.StringVar(value=self._vr_output_auto)
        # Remembered so that a restored name this app generated keeps following
        # the layout radio, and so a finished run can hand the next job a fresh
        # timestamp instead of quietly overwriting the render just produced.
        # Only rewrites the name while it is still the one we generated, so a
        # path the user typed or browsed to is never moved under them.
        self.vr_layout_var.trace_add("write", self._sync_vr_output_suffix)
        self.vr_output_entry = ctk.CTkEntry(tab, textvariable=self.vr_output_var, width=350)
        self.vr_output_entry.grid(row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)
        ctk.CTkButton(tab, text="Browse", width=70, command=self.browse_vr_output).grid(
            row=row, column=2, padx=(5, 0), pady=pad_y
        )

        row += 1
        ctk.CTkLabel(tab, text="Clip Start", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y
        )
        self.vr_trim_start_var = ctk.StringVar(value="")
        ctk.CTkEntry(tab, textvariable=self.vr_trim_start_var, width=100).grid(
            row=row, column=1, sticky="w", padx=pad_x, pady=pad_y
        )
        ctk.CTkButton(tab, text="Select Clip", width=90,
                      command=self.select_vr_clip).grid(
            row=row, column=2, padx=(5, 0), pady=pad_y
        )
        ctk.CTkLabel(tab, text="Seconds or m:ss - empty means the whole video.",
                     font=ctk.CTkFont(size=10),
                     text_color="gray").grid(row=row, column=3, padx=(5, 0), pady=pad_y)

        row += 1
        ctk.CTkLabel(tab, text="Clip End", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y
        )
        self.vr_trim_end_var = ctk.StringVar(value="")
        ctk.CTkEntry(tab, textvariable=self.vr_trim_end_var, width=100).grid(
            row=row, column=1, sticky="w", padx=pad_x, pady=pad_y
        )
        ctk.CTkButton(tab, text="Whole Video", width=90, fg_color="#444444",
                      hover_color="#555555", command=self.clear_vr_clip).grid(
            row=row, column=2, padx=(5, 0), pady=pad_y
        )
        self.vr_clip_label = ctk.CTkLabel(tab, text="", font=ctk.CTkFont(size=10),
                                          text_color="#5a9bd5")
        self.vr_clip_label.grid(row=row, column=3, sticky="w", padx=(5, 0), pady=pad_y)

        row += 1
        ctk.CTkLabel(tab, text="Output FPS", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y
        )
        self.vr_fps_var = ctk.StringVar(value="")
        ctk.CTkEntry(tab, textvariable=self.vr_fps_var, width=100).grid(
            row=row, column=1, sticky="w", padx=pad_x, pady=pad_y
        )
        ctk.CTkLabel(tab, text="(leave empty for default)", font=ctk.CTkFont(size=10),
                     text_color="gray").grid(row=row, column=3, padx=(5, 0), pady=pad_y)

        row += 1
        self.vr_keep_temp_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(tab, text="Keep temp files (debugging)", variable=self.vr_keep_temp_var).grid(
            row=row, column=1, sticky="w", padx=pad_x, pady=pad_y
        )

        row += 1
        # On by default: a failed run leaves its finished stages in output/temp,
        # and re-swapping an eye that already rendered is the better part of an
        # hour of GPU time. Untick it to force everything from scratch.
        self.vr_resume_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(tab, text="Resume from where the last attempt stopped",
                        variable=self.vr_resume_var).grid(
            row=row, column=1, sticky="w", padx=pad_x, pady=pad_y
        )
        ctk.CTkLabel(tab, text="(reuses finished eyes; ignored if settings changed)",
                     font=ctk.CTkFont(size=10), text_color="gray").grid(
            row=row, column=2, columnspan=2, sticky="w", padx=(5, 0), pady=pad_y)

        row += 1
        ctk.CTkLabel(tab, text="Start At", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y
        )
        self.vr_schedule_var = ctk.StringVar(value="")
        ctk.CTkEntry(tab, textvariable=self.vr_schedule_var, width=140).grid(
            row=row, column=1, sticky="w", padx=pad_x, pady=pad_y
        )
        ctk.CTkLabel(tab, text=SCHEDULE_HINT, font=ctk.CTkFont(size=10),
                     text_color="gray").grid(row=row, column=2, columnspan=2,
                                             sticky="w", padx=(5, 0), pady=pad_y)

        row += 1
        ctk.CTkLabel(tab, text="A time here makes Run wait for it. Add to Queue lines "
                              "jobs up to run one after another (see the Queue tab). "
                              "The app has to stay open either way.",
                     font=ctk.CTkFont(size=10), text_color="#888888").grid(
            row=row, column=1, columnspan=3, sticky="w", padx=pad_x)

        row += 2
        self.vr_run_btn = ctk.CTkButton(
            tab, text="Run VR Pipeline", height=45, font=ctk.CTkFont(size=14, weight="bold"),
            command=self.run_vr, corner_radius=10
        )
        self.vr_run_btn.grid(row=row, column=1, padx=pad_x, pady=(10, 5))
        ctk.CTkButton(
            tab, text="Add to Queue", height=45, width=130,
            font=ctk.CTkFont(size=13, weight="bold"), fg_color="#374151",
            hover_color="#4b5563", corner_radius=10,
            command=lambda: self.add_to_queue("vr")
        ).grid(row=row, column=2, columnspan=2, sticky="w", padx=(5, 0), pady=(10, 5))

        row += 1
        ctk.CTkLabel(tab, text="VR Pipeline: Splits stereoscopic video, runs FaceFusion on each eye, merges back.",
                     font=ctk.CTkFont(size=10), text_color="gray").grid(
            row=row, column=1, sticky="w", padx=pad_x, pady=(0, 5)
        )

    def build_normal_tab(self):
        tab = self.tabview.tab("Normal Mode")
        tab.grid_columnconfigure(1, weight=1)

        row = 0
        pad_y = 8
        pad_x = 15

        ctk.CTkLabel(tab, text="Quality Preset", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y
        )
        self.nm_preset_var = ctk.StringVar(value=DEFAULT_PRESET)
        self.nm_preset_combo = ctk.CTkComboBox(
            tab, values=list(QUALITY_PRESETS.keys()), variable=self.nm_preset_var,
            width=250, state="readonly", command=self.on_nm_preset_change
        )
        self.nm_preset_combo.grid(row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)
        self.nm_preset_tooltip = ctk.CTkLabel(
            tab, text=QUALITY_PRESETS[DEFAULT_PRESET]['tooltip'],
            font=ctk.CTkFont(size=10), text_color="#888888"
        )
        self.nm_preset_tooltip.grid(row=row, column=2, sticky="w", padx=(5, 0), pady=pad_y)

        row += 1
        ctk.CTkLabel(tab, text="Source Face", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y
        )
        self.nm_source_var = ctk.StringVar()
        self.nm_source_entry = ctk.CTkEntry(tab, textvariable=self.nm_source_var, width=350)
        self.nm_source_entry.grid(row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)
        ctk.CTkButton(tab, text="Browse", width=70, command=self.browse_nm_source).grid(
            row=row, column=2, padx=(5, 0), pady=pad_y
        )

        row += 1
        ctk.CTkLabel(
            tab, text="Photos, or one short video clip of the face.",
            font=ctk.CTkFont(size=10), text_color="#888888"
        ).grid(row=row, column=1, sticky="w", padx=pad_x)
        self.nm_refresh_faces_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(tab, text="Re-extract", variable=self.nm_refresh_faces_var,
                        font=ctk.CTkFont(size=10), checkbox_width=16, checkbox_height=16,
                        width=90).grid(row=row, column=2, sticky="w", padx=(5, 0))

        row += 1
        ctk.CTkLabel(tab, text="Target Video", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y
        )
        self.nm_target_var = ctk.StringVar()
        self.nm_target_entry = ctk.CTkEntry(tab, textvariable=self.nm_target_var, width=350)
        self.nm_target_entry.grid(row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)
        ctk.CTkButton(tab, text="Browse", width=70, command=self.browse_nm_target).grid(
            row=row, column=2, padx=(5, 0), pady=pad_y
        )

        row += 1
        ctk.CTkLabel(tab, text="Output Path", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y
        )
        self._nm_output_auto = os.path.join(
            PROJECT_DIR, "output", "final",
            f"deepfake_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        self.nm_output_var = ctk.StringVar(value=self._nm_output_auto)
        self.nm_output_entry = ctk.CTkEntry(tab, textvariable=self.nm_output_var, width=350)
        self.nm_output_entry.grid(row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)
        ctk.CTkButton(tab, text="Browse", width=70, command=self.browse_nm_output).grid(
            row=row, column=2, padx=(5, 0), pady=pad_y
        )

        row += 1
        ctk.CTkLabel(tab, text="Clip Start", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y
        )
        self.nm_trim_start_var = ctk.StringVar(value="")
        ctk.CTkEntry(tab, textvariable=self.nm_trim_start_var, width=100).grid(
            row=row, column=1, sticky="w", padx=pad_x, pady=pad_y
        )
        ctk.CTkButton(tab, text="Select Clip", width=90,
                      command=self.select_nm_clip).grid(
            row=row, column=2, padx=(5, 0), pady=pad_y
        )
        ctk.CTkLabel(tab, text="Seconds or m:ss - empty means the whole video.",
                     font=ctk.CTkFont(size=10),
                     text_color="gray").grid(row=row, column=3, padx=(5, 0), pady=pad_y)

        row += 1
        ctk.CTkLabel(tab, text="Clip End", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y
        )
        self.nm_trim_end_var = ctk.StringVar(value="")
        ctk.CTkEntry(tab, textvariable=self.nm_trim_end_var, width=100).grid(
            row=row, column=1, sticky="w", padx=pad_x, pady=pad_y
        )
        ctk.CTkButton(tab, text="Whole Video", width=90, fg_color="#444444",
                      hover_color="#555555", command=self.clear_nm_clip).grid(
            row=row, column=2, padx=(5, 0), pady=pad_y
        )
        self.nm_clip_label = ctk.CTkLabel(tab, text="", font=ctk.CTkFont(size=10),
                                          text_color="#5a9bd5")
        self.nm_clip_label.grid(row=row, column=3, sticky="w", padx=(5, 0), pady=pad_y)

        row += 1
        ctk.CTkLabel(tab, text="Output FPS", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y
        )
        self.nm_fps_var = ctk.StringVar(value="")
        ctk.CTkEntry(tab, textvariable=self.nm_fps_var, width=100).grid(
            row=row, column=1, sticky="w", padx=pad_x, pady=pad_y
        )
        ctk.CTkLabel(tab, text="(leave empty for default)", font=ctk.CTkFont(size=10),
                     text_color="gray").grid(row=row, column=3, padx=(5, 0), pady=pad_y)

        row += 1
        self.nm_schedule_var = ctk.StringVar(value="")
        ctk.CTkLabel(tab, text="Start At", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y
        )
        ctk.CTkEntry(tab, textvariable=self.nm_schedule_var, width=140).grid(
            row=row, column=1, sticky="w", padx=pad_x, pady=pad_y
        )
        ctk.CTkLabel(tab, text=SCHEDULE_HINT, font=ctk.CTkFont(size=10),
                     text_color="gray").grid(row=row, column=2, columnspan=2,
                                             sticky="w", padx=(5, 0), pady=pad_y)

        row += 1
        self.nm_keep_temp_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(tab, text="Keep temp files (debugging)", variable=self.nm_keep_temp_var).grid(
            row=row, column=1, sticky="w", padx=pad_x, pady=pad_y
        )

        row += 2
        self.nm_run_btn = ctk.CTkButton(
            tab, text="Run Normal Pipeline", height=45, font=ctk.CTkFont(size=14, weight="bold"),
            command=self.run_normal, corner_radius=10
        )
        self.nm_run_btn.grid(row=row, column=1, padx=pad_x, pady=(10, 5))
        ctk.CTkButton(
            tab, text="Add to Queue", height=45, width=130,
            font=ctk.CTkFont(size=13, weight="bold"), fg_color="#374151",
            hover_color="#4b5563", corner_radius=10,
            command=lambda: self.add_to_queue("normal")
        ).grid(row=row, column=2, columnspan=2, sticky="w", padx=(5, 0), pady=(10, 5))

        row += 1
        ctk.CTkLabel(tab, text="Normal Pipeline: Runs FaceFusion directly on the video (no split/merge).",
                     font=ctk.CTkFont(size=10), text_color="gray").grid(
            row=row, column=1, sticky="w", padx=pad_x, pady=(0, 5)
        )

    def build_upscale_tab(self):
        """
        Enlarge an existing video. Nothing to do with face swapping - it is a
        separate tool that happens to live in the same window, so it takes a
        finished file (or any file at all) rather than a source face.
        """
        tab = self.tabview.tab("Upscale")
        tab.grid_columnconfigure(1, weight=1)
        pad_x, pad_y = 10, 6
        row = 0

        ctk.CTkLabel(tab, text="Upscale any video - VR or ordinary 2D. Independent of "
                               "the deepfake pipeline.",
                     font=ctk.CTkFont(size=11), text_color="gray").grid(
            row=row, column=0, columnspan=4, sticky="w", padx=pad_x, pady=(8, 10))

        row += 1
        ctk.CTkLabel(tab, text="Input Video", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y)
        self.up_input_var = ctk.StringVar()
        ctk.CTkEntry(tab, textvariable=self.up_input_var, width=350).grid(
            row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)
        ctk.CTkButton(tab, text="Browse", width=70, command=self.browse_up_input).grid(
            row=row, column=2, padx=(5, 0), pady=pad_y)

        row += 1
        ctk.CTkLabel(tab, text="Output", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y)
        self.up_output_var = ctk.StringVar()
        ctk.CTkEntry(tab, textvariable=self.up_output_var, width=350).grid(
            row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)
        ctk.CTkButton(tab, text="Browse", width=70, command=self.browse_up_output).grid(
            row=row, column=2, padx=(5, 0), pady=pad_y)

        row += 1
        ctk.CTkLabel(tab, text="Portion", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y)
        self.up_trim_start_var = ctk.StringVar(value="")
        self.up_trim_end_var = ctk.StringVar(value="")
        portion_frame = ctk.CTkFrame(tab, fg_color="transparent")
        portion_frame.grid(row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)
        ctk.CTkEntry(portion_frame, textvariable=self.up_trim_start_var, width=90,
                     placeholder_text="start").pack(side="left")
        ctk.CTkLabel(portion_frame, text="to", width=24).pack(side="left")
        ctk.CTkEntry(portion_frame, textvariable=self.up_trim_end_var, width=90,
                     placeholder_text="end").pack(side="left")
        ctk.CTkButton(tab, text="Select Clip", width=70,
                      command=self.select_up_clip).grid(
            row=row, column=2, padx=(5, 0), pady=pad_y)
        ctk.CTkLabel(tab, text="Upscale only part of the file - empty means all of it",
                     font=ctk.CTkFont(size=10), text_color="gray").grid(
            row=row, column=3, sticky="w", padx=(5, 0), pady=pad_y)

        row += 1
        ctk.CTkLabel(tab, text="Target Size", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y)
        self.up_target_var = ctk.StringVar(value="8k")
        ctk.CTkComboBox(tab, values=list(upscale_vr.TARGETS), width=140, state="readonly",
                        variable=self.up_target_var,
                        command=self.refresh_upscale_preview).grid(
            row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)

        row += 1
        ctk.CTkLabel(tab, text="Layout", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y)
        self.up_layout_var = ctk.StringVar(value="auto")
        ctk.CTkComboBox(tab, values=["auto", "sbs", "tb", "mono"], width=140,
                        state="readonly", variable=self.up_layout_var,
                        command=self.refresh_upscale_preview).grid(
            row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)
        ctk.CTkLabel(tab, text="auto reads the file's stereo tag / name; falls back to 2D",
                     font=ctk.CTkFont(size=10), text_color="gray").grid(
            row=row, column=3, sticky="w", padx=(5, 0), pady=pad_y)

        row += 1
        ctk.CTkLabel(tab, text="Method", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y)
        self.up_method_var = ctk.StringVar(value="lanczos")
        ctk.CTkComboBox(tab, values=["lanczos", "bicubic", "spline", "esrgan"], width=140,
                        state="readonly", variable=self.up_method_var,
                        command=self.refresh_upscale_preview).grid(
            row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)
        ctk.CTkLabel(tab, text="esrgan is real super-resolution and is ~300x slower",
                     font=ctk.CTkFont(size=10), text_color="gray").grid(
            row=row, column=3, sticky="w", padx=(5, 0), pady=pad_y)

        row += 1
        ctk.CTkLabel(tab, text="Tune For", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y)
        self.up_headset_var = ctk.StringVar(value="quest3")
        ctk.CTkComboBox(tab, values=["none"] + sorted(upscale_vr.HEADSETS), width=140,
                        state="readonly", variable=self.up_headset_var,
                        command=self.refresh_upscale_preview).grid(
            row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)

        row += 1
        ctk.CTkLabel(tab, text="Encoder", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y)
        self.up_encoder_var = ctk.StringVar(
            value=self.config.get("encoder") or "hevc_nvenc")
        ctk.CTkComboBox(tab, values=UPSCALE_ENCODERS, width=140, state="readonly",
                        variable=self.up_encoder_var,
                        command=self.refresh_upscale_preview).grid(
            row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)
        ctk.CTkLabel(tab, text="av1_nvenc is ~25% smaller at equal quality; libx265 is "
                               "software and the only route past 8192px",
                     font=ctk.CTkFont(size=10), text_color="gray").grid(
            row=row, column=3, sticky="w", padx=(5, 0), pady=pad_y)

        row += 1
        ctk.CTkLabel(tab, text="Quality", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y)
        quality_frame = ctk.CTkFrame(tab, width=300, height=36)
        quality_frame.grid(row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)
        quality_frame.grid_columnconfigure(0, weight=1)
        low, high, _ = SLIDER_RANGES["quality"]
        initial = int(self.config.get("quality", 95))
        self.up_quality_slider = ctk.CTkSlider(
            quality_frame, from_=low, to=high, number_of_steps=int(high - low),
            command=self.on_up_quality_change, width=200)
        self.up_quality_slider.set(initial)
        self.up_quality_label = ctk.CTkLabel(quality_frame, text=str(initial), width=40)
        self.up_quality_slider.grid(row=0, column=0, padx=(5, 0), pady=3, sticky="ew")
        self.up_quality_label.grid(row=0, column=1, padx=(5, 5), pady=3, sticky="e")

        row += 1
        self.up_split_eyes_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(tab, text="One file per eye (the only route past 8K that still "
                                  "hardware-decodes)",
                        variable=self.up_split_eyes_var,
                        command=self.refresh_upscale_preview).grid(
            row=row, column=1, columnspan=3, sticky="w", padx=pad_x, pady=pad_y)

        row += 1
        self.up_allow_sw_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(tab, text="Allow software encoder (slow; will not hardware-decode)",
                        variable=self.up_allow_sw_var,
                        command=self.refresh_upscale_preview).grid(
            row=row, column=1, columnspan=3, sticky="w", padx=pad_x, pady=pad_y)

        # The preflight exists to answer "will this work, and will it even be
        # visible" before anything is encoded. Showing it live as the settings
        # change is far more useful than printing it afterwards.
        row += 1
        self.up_preview = ctk.CTkTextbox(tab, height=120, wrap="word",
                                         font=ctk.CTkFont(family="Consolas", size=11))
        self.up_preview.grid(row=row, column=0, columnspan=4, sticky="ew",
                             padx=pad_x, pady=(12, 4))

        row += 1
        ctk.CTkLabel(tab, text="Start At", font=ctk.CTkFont(weight="bold")).grid(
            row=row, column=0, sticky="w", padx=pad_x, pady=pad_y)
        self.up_schedule_var = ctk.StringVar(value="")
        ctk.CTkEntry(tab, textvariable=self.up_schedule_var, width=140).grid(
            row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)
        ctk.CTkLabel(tab, text=SCHEDULE_HINT, font=ctk.CTkFont(size=10),
                     text_color="gray").grid(row=row, column=2, columnspan=2,
                                             sticky="w", padx=(5, 0), pady=pad_y)

        row += 1
        self.up_run_btn = ctk.CTkButton(
            tab, text="Run Upscale", height=45, font=ctk.CTkFont(size=14, weight="bold"),
            command=self.run_upscale_job, corner_radius=10)
        self.up_run_btn.grid(row=row, column=1, padx=pad_x, pady=(6, 5))
        ctk.CTkButton(
            tab, text="Add to Queue", height=45, width=130,
            font=ctk.CTkFont(size=13, weight="bold"), fg_color="#374151",
            hover_color="#4b5563", corner_radius=10,
            command=lambda: self.add_to_queue("upscale")
        ).grid(row=row, column=2, columnspan=2, sticky="w", padx=(5, 0), pady=(6, 5))

        self.refresh_upscale_preview()

    def on_up_quality_change(self, value):
        self.up_quality_label.configure(text=str(int(round(value))))
        self.refresh_upscale_preview()

    def browse_up_input(self):
        path = filedialog.askopenfilename(
            title="Select a video to upscale",
            filetypes=[("Videos", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v *.ts")])
        if path:
            self.up_input_var.set(path)
            if not self.up_output_var.get().strip():
                stem, ext = os.path.splitext(path)
                self.up_output_var.set(stem + "_upscaled" + (ext or ".mp4"))
            self.refresh_upscale_preview()

    def browse_up_output(self):
        path = filedialog.asksaveasfilename(
            title="Save upscaled video as", defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("MKV Video", "*.mkv")])
        if path:
            self.up_output_var.set(path)

    def upscale_args(self):
        """
        Build the same options object the CLI's argparse produces.

        The GUI calls upscale_vr.run_upscale() directly rather than shelling out
        or keeping a second copy of the logic, so the two front ends cannot
        drift apart.
        """
        return self._upscale_args_from(self._tab_fields("upscale"))

    def _upscale_args_from(self, fields):
        """The options object for an upscale job's frozen fields."""
        headset = fields.get("headset") or "none"
        return SimpleNamespace(
            input=(fields.get("input") or "").strip(),
            output=(fields.get("output") or "").strip(),
            target=fields.get("target"),
            scale=None,
            layout=fields.get("layout"),
            method=fields.get("method"),
            content=None,
            headset=None if headset == "none" else headset,
            # The tab owns these rather than reading the Settings tab: that one
            # configures the deepfake pipeline, and an upscale is a different
            # job with different trade-offs - a software encoder here is a
            # deliberate choice to exceed 8192px, not a fallback.
            encoder=fields.get("encoder"),
            quality=int(fields.get("quality") or 95),
            allow_software=bool(fields.get("allow_sw")),
            split_eyes=bool(fields.get("split_eyes")),
            trim_start=self.parse_optional_float(fields.get("trim_start") or ""),
            trim_end=self.parse_optional_float(fields.get("trim_end") or ""),
            keep_temp=False,
            yes=True,
        )

    def refresh_upscale_preview(self, _value=None):
        """Show what this run would produce, and whether it can happen at all."""
        self.up_preview.delete("1.0", "end")
        args = self.upscale_args()
        if not args.input or not os.path.exists(args.input):
            self.up_preview.insert("end", "Pick an input video to see what it "
                                          "would produce.")
            return
        try:
            lines = upscale_vr.describe(args)
        except Exception as exc:
            lines = ["Could not read that file: " + str(exc)]
        self.up_preview.insert("end", "\n".join(lines))

    def run_upscale_job(self):
        self._run_now_or_queue("upscale")

    def build_settings_tab(self):
        tab = self.tabview.tab("Settings")
        tab.grid_columnconfigure(1, weight=1)

        row = 0
        pad_y = 6
        pad_x = 15

        self.settings_widgets = {}

        sections = [
            # These lists are the model names FaceFusion 3.8.2 actually accepts
            # (facefusion/processors/modules/*/types.py and facefusion/choices.py).
            # Anything not in them is rejected by argparse and the run dies at
            # launch, so do not add names by hand without checking there first.
            ("Face Swapper", [
                ("Model", "swapper_model", "combobox",
                 ["hyperswap_1a_256", "hyperswap_1b_256", "hyperswap_1c_256",
                  "inswapper_128_fp16", "inswapper_128", "simswap_unofficial_512",
                  "simswap_256", "ghost_3_256", "ghost_2_256", "ghost_1_256",
                  "hififace_unofficial_256", "blendswap_256", "uniface_256"]),
                # Valid boosts depend on the model: the 256 models start at
                # 256x256, simswap_unofficial_512 at 512x512, inswapper at 128x128.
                ("Pixel Boost", "pixel_boost", "combobox",
                 ["256x256", "512x512", "768x768", "1024x1024", "384x384", "128x128"]),
                # FaceFusion blends the source and target ArcFace embeddings.
                # 0.50 is neutral - the swap uses the source identity as-is.
                # Below it the original face is mixed back in; above it the
                # result is pushed away from the original, which sharpens the
                # resemblance to the source photo. Past ~0.70 the face starts
                # to stop fitting the head it is sitting on.
                ("Identity Strength", "swapper_weight", "slider", None),
            ]),
            # A frame the detector loses is otherwise left unswapped, so the
            # original face flashes back for a single frame - which is very
            # obvious in a headset. The tracker interpolates that frame from
            # its neighbours instead. It does NOT smooth landmarks: where
            # detection never fails it changes nothing at all.
            ("Detection Dropout Recovery", [
                ("Face Tracker", "face_tracker_score", "slider", None),
                ("Tracker Window (frames)", "target_frame_amount", "slider", None),
            ]),
            # 'one' replaces a single face per frame - the first one left after
            # Order sorts them and Gender filters them. 'many' replaces every
            # face in shot, bystanders and detector mistakes included.
            ("Face Selection", [
                ("Mode", "selector_mode", "combobox",
                 ["one", "many", "reference"]),
                ("Which Face", "selector_order", "combobox",
                 ["large-small", "best-worst", "left-right", "right-left",
                  "top-bottom", "bottom-top", "small-large", "worst-best"]),
                # 'auto' takes the gender of the source photo, so a female source
                # only ever replaces female faces. Pin it when the source is
                # ambiguous or the classifier gets it wrong.
                ("Gender", "selector_gender", "combobox",
                 ["auto", "female", "male"]),
                # 'reference' only: how close a face has to be to the reference
                # embedding to count as the same person, and which frame the
                # reference is taken from. Ignored by 'one' and 'many'.
                ("Reference Distance", "reference_face_distance", "slider", None),
                ("Reference Frame", "reference_frame_number", "entry", None),
            ]),
            ("Face Enhancer", [
                ("Model", "enhancer_model", "combobox",
                 ["gpen_bfr_512", "gpen_bfr_1024", "codeformer", "gfpgan_1.4",
                  "gfpgan_1.3", "gfpgan_1.2", "restoreformer_plus_plus",
                  "gpen_bfr_256", "gpen_bfr_2048", "none"]),
                ("Blend (%)", "enhancer_blend", "slider", None),
                # Only enhancers whose ONNX graph has a 'weight' input read
                # this, which in 3.8.2 is codeformer alone (its fidelity knob).
                ("Weight (codeformer only)", "enhancer_weight", "slider", None),
            ]),
            # Frame enhancers upscale the WHOLE frame, so VRAM cost scales with
            # video resolution, not face size. Very easy to overcommit on VR.
            # hyperswap holds identity well but pulls expression towards the
            # source photo's, so a talking head can come out subtly deadpan.
            # LivePortrait puts the original expression back on top of the
            # swapped face. It is a second generative model per face, so it is
            # off unless asked for.
            ("Expression Restorer (optional, heavy)", [
                ("Enabled", "expression_restorer", "combobox", ["off", "on"]),
                ("Strength", "expression_restorer_factor", "slider", None),
                ("Areas", "expression_restorer_areas", "combobox",
                 ["upper-face lower-face", "lower-face", "upper-face"]),
            ]),
            ("Frame Enhancer (optional, heavy)", [
                ("Model", "frame_enhancer", "combobox",
                 ["none", "real_esrgan_x2_fp16", "real_esrgan_x2",
                  "real_esrgan_x4_fp16", "real_esrgan_x4", "clear_reality_x4",
                  "span_kendata_x4", "ultra_sharp_x4"]),
                ("Blend (%)", "frame_enhancer_blend", "slider", None),
            ]),
            ("Face Detector", [
                # 'mediapipe_face_detection', 'fast' and 'yoloface' were in this
                # list and none of them exist in 3.8.2 - picking one killed the
                # run at launch. The name is 'yolo_face', with an underscore.
                ("Model", "detector_model", "combobox",
                 ["retinaface", "yolo_face", "scrfd", "yunet", "many"]),
                # Only retinaface and scrfd accept anything below 640x640; the
                # others are 640x640 only, and nothing accepts 800x800.
                ("Size", "detector_size", "combobox",
                 ["640x640", "512x512", "480x480", "320x320", "160x160"]),
                ("Score Threshold", "detector_score", "slider", None),
                # Grows the detection box per side before landmarking; a box
                # that clips the jaw drags the landmarks in with it.
                ("Box Margin", "detector_margin", "combobox",
                 ["0 0 0 0", "5 5 5 5", "10 10 10 10", "0 0 10 0"]),
                # Landmarks decide the affine warp, so a bad set does not just
                # soften the swap, it makes it slide between frames. 'many'
                # runs both models and keeps whichever scores higher, but costs
                # ~85% more runtime and on ordinary front-facing footage 2dfan4
                # wins every frame anyway (measured: byte-identical output).
                # Worth it only for hard profiles and heavy head roll.
                ("Landmarker Model", "landmarker_model", "combobox",
                 ["2dfan4", "many", "peppa_wutz"]),
                ("Landmarker Score", "landmarker_score", "slider", None),
            ]),
            ("Encoding", [
                ("Encoder", "encoder", "combobox",
                 ["h264_nvenc", "hevc_nvenc", "h264_amf", "libx264", "libx265"]),
                ("Quality", "quality", "slider", None),
                # FaceFusion collapses these onto the encoder's own ladder: for
                # NVENC everything up to 'fast' becomes nvenc 'fast', 'medium'
                # becomes 'medium', and slow/slower/veryslow become nvenc
                # 'slow'. Its own default is 'veryfast' - the bottom rung.
                ("Encoder Preset", "output_video_preset", "combobox",
                 ["slow", "medium", "veryfast"]),
            ]),
            ("Hardware", [
                ("Provider", "provider", "combobox",
                 ["cuda", "rocm", "dml", "cpu"]),
                ("Threads", "threads", "slider", None),
                # 'tolerant' keeps every model resident in VRAM. On a card that
                # cannot hold them all, Windows pages GPU memory into system RAM
                # and the desktop stops responding.
                ("VRAM Strategy", "video_memory_strategy", "combobox",
                 ["moderate", "strict", "tolerant"]),
                ("Temp Frame Format", "temp_frame_format", "combobox",
                 ["jpeg", "png", "bmp", "tiff"]),
            ]),
            ("Face Masking", [
                ("Types", "mask_types", "combobox",
                  ["box occlusion", "box", "occlusion", "box occlusion region", "none"]),
                ("Blur", "mask_blur", "slider", None),
                ("Padding", "mask_padding", "combobox",
                  ["0 0 0 0", "0 5 0 5", "0 10 0 10", "5 5 5 5"]),
                # Decides what counts as "in front of the face" - a hand, hair,
                # a mic - and is therefore left alone. 'many' runs all three
                # xseg models and intersects them.
                ("Occluder Model", "occluder_model", "combobox",
                  ["xseg_1", "xseg_2", "xseg_3", "many"]),
                # Only consulted when mask Types includes 'region'.
                ("Parser Model", "parser_model", "combobox",
                  ["bisenet_resnet_34", "bisenet_resnet_18"]),
            ]),
            # Only consulted when the Source Face is a video rather than
            # photos. FaceFusion averages the ArcFace embeddings of every
            # source image it is handed, so these decide what goes into that
            # average - see engine.extract_source_faces.
            ("Source Face from Video", [
                ("Stills to Keep", "source_video_faces", "slider", None),
                # Two per second is already far above any rate a face actually
                # changes at, and keeps a 30 s clip to 60 detections.
                ("Sample Rate (fps)", "source_video_sample_fps", "entry", None),
                # The sample rate is lowered to fit this rather than the clip
                # being cut short, so a long clip is still covered end to end.
                ("Max Frames Sampled", "source_video_max_samples", "entry", None),
                # Below this the face has fewer pixels than the 112x112 the
                # recogniser wants and the embedding is mostly upscaling.
                ("Min Face Size (px)", "source_video_min_face_px", "entry", None),
                # Cosine similarity above which two frames are taken to be the
                # same person. Lower it if one person's own clip is being split
                # into two identities; raise it if a second face is getting in.
                ("Identity Threshold", "source_video_identity_threshold", "slider", None),
            ]),
            ("Output Upscaling (optional)", [
                ("Method", "upscale_method", "combobox",
                  ["none", "realesrgan", "ffmpeg"]),
                ("Scale Factor", "upscale_scale", "slider", None),
                ("Target Width (px)", "upscale_target_width", "entry", None),
            ]),
            ("Frame Interpolation - RIFE (optional, very slow)", [
                ("Frame Rate Multiplier", "interpolate_multi", "slider", None),
                ("Model Resolution", "interpolate_scale", "slider", None),
            ]),
            # The selected range is processed in pieces of this many seconds,
            # each one a finished, playable file in <output>_chunks/ before
            # the next starts, and joined at the end. A run that dies keeps
            # every finished piece and the next run continues from there.
            # 0 processes the whole range in one go, the old behaviour.
            ("Chunked Processing", [
                ("Chunk Length (seconds, 0 = off)", "chunk_seconds", "entry", None),
                # Both eyes swapped at once in two FaceFusion processes, when
                # the machine can hold them. "auto" measures the free VRAM,
                # commit and RAM before every run and says no when two lanes
                # would not fit - which on this card is any 4K-per-eye run with
                # a face enhancer, because one pass alone holds ~7 GB of 16.3.
                # "on" skips the commit and RAM margins but never the VRAM one:
                # overcommitting the card is not a slow run, it is Windows
                # paging GPU memory into system RAM and a desktop that stops
                # answering. The log says which way the verdict went and why,
                # before anything starts.
                ("Both Eyes At Once", "parallel_eyes", "combobox",
                 ["auto", "on", "off"]),
            ]),
        ]

        for section_name, fields in sections:
            row += 1
            ctk.CTkLabel(tab, text=section_name, font=ctk.CTkFont(size=13, weight="bold"),
                         anchor="w").grid(row=row, column=0, columnspan=3,
                                          sticky="w", padx=pad_x, pady=(10, 2))

            for label, key, widget_type, opts in fields:
                row += 1
                ctk.CTkLabel(tab, text=label, font=ctk.CTkFont(size=11)).grid(
                    row=row, column=0, sticky="w", padx=pad_x, pady=pad_y
                )

                val = (self.guard_settings.get(key) if key in GUARD_SETTINGS
                       else self.config.get(key))

                if widget_type == "combobox":
                    w = ctk.CTkComboBox(tab, values=opts, width=200, state="readonly")
                    if key in PRESET_KEYS:
                        w.configure(command=self.on_preset_key_edited)
                    if isinstance(val, bool):
                        val = "on" if val else "off"
                    w.set(val if val and val in opts else opts[0])
                    self.settings_widgets[key] = ("combobox", w)
                    w.grid(row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)

                elif widget_type == "entry":
                    w = ctk.CTkEntry(tab, width=200)
                    w.insert(0, str(val) if val is not None else "")
                    self.settings_widgets[key] = ("entry", w)
                    w.grid(row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)

                elif widget_type == "slider":
                    mn, mx, decimals = SLIDER_RANGES.get(key, (0, 100, 0))

                    frame = ctk.CTkFrame(tab, width=300, height=36)
                    frame.grid(row=row, column=1, sticky="w", padx=pad_x, pady=pad_y)
                    frame.grid_columnconfigure(0, weight=1)

                    # Without number_of_steps a CTkSlider is continuous, so a
                    # 0.0-0.5 tracker score lands on values like 0.1734.
                    steps = int(round((mx - mn) / (1 if decimals == 0 else 0.05)))
                    slider = ctk.CTkSlider(
                        frame, from_=mn, to=mx, number_of_steps=max(1, steps),
                        command=lambda v, k=key: self.update_slider_label(k, v)
                    )
                    initial = (mn + mx) / 2 if val is None else val
                    slider.set(initial)
                    lbl = ctk.CTkLabel(frame, text=f"{initial:.{decimals}f}", width=40)
                    slider.grid(row=0, column=0, padx=(5, 0), pady=3, sticky="ew")
                    lbl.grid(row=0, column=1, padx=(5, 5), pady=3, sticky="e")
                    self.settings_widgets[key] = ("slider", slider, lbl)
                    frame.update_idletasks()

            row += 1

        row += 2
        self.save_settings_btn = ctk.CTkButton(
            tab, text="Save Settings", command=self.save_settings,
            width=150, corner_radius=8
        )
        self.save_settings_btn.grid(row=row, column=1, padx=pad_x, pady=15)

    def on_vr_preset_change(self, preset_name):
        self._select_preset(preset_name)

    def on_nm_preset_change(self, preset_name):
        self._select_preset(preset_name)

    def _select_preset(self, preset_name):
        # Both tabs drive the same underlying settings, so a preset chosen on
        # one has to show on the other - otherwise the VR tab reads 'Good'
        # while the settings behind it are the Normal tab's 'Best'.
        self.vr_preset_var.set(preset_name)
        self.nm_preset_var.set(preset_name)
        self._sync_preset_tooltips(preset_name)
        self._apply_preset(preset_name, announce=True)

    def _sync_preset_tooltips(self, preset_name):
        if preset_name in QUALITY_PRESETS:
            tooltip = QUALITY_PRESETS[preset_name]['tooltip']
            self.vr_preset_tooltip.configure(text=tooltip)
            self.nm_preset_tooltip.configure(text=tooltip)

    def on_preset_key_edited(self, _value=None):
        """
        A setting a preset owns was changed by hand.

        Switch both dropdowns to Custom so run_vr/run_normal stop re-applying a
        preset over the edit. Without this the Settings tab was write-only for
        the four preset keys: the value went in, and the preset put its own
        value back a moment before the run started.
        """
        if self.vr_preset_var.get() == CUSTOM_PRESET:
            return
        self.vr_preset_var.set(CUSTOM_PRESET)
        self.nm_preset_var.set(CUSTOM_PRESET)
        self._sync_preset_tooltips(CUSTOM_PRESET)
        self.log("Quality preset: Custom (a preset setting was edited by hand)")

    def _apply_preset(self, preset_name, announce=False):
        if preset_name not in QUALITY_PRESETS:
            return
        preset = QUALITY_PRESETS[preset_name]
        if preset['swapper_model'] is None:
            # Custom. Selecting it deliberately means "use my saved settings",
            # so put the saved config back into the widgets - it used to do
            # nothing at all, which left the previous preset's models in place
            # and made Custom a copy of whatever was chosen before it.
            if announce:
                self._load_saved_settings_into_widgets()
                self.log("Quality preset: Custom - reloaded saved settings.")
            return
        mapping = {
            'swapper_model': preset['swapper_model'],
            'pixel_boost': preset['pixel_boost'],
            'enhancer_model': preset['enhancer_model'],
            'frame_enhancer': preset['frame_enhancer'],
        }
        for key, val in mapping.items():
            if key in self.settings_widgets:
                w = self.settings_widgets[key]
                if w[0] == "combobox":
                    w[1].set(val if val else "none")
        if announce:
            self.log(f"Quality preset: {preset_name}")

    def _load_saved_settings_into_widgets(self):
        """Put the settings a preset owns back to what pipeline_config.json holds."""
        if not os.path.exists(self.config_path):
            return
        saved = load_config(self.config_path)
        for key in PRESET_KEYS:
            w = self.settings_widgets.get(key)
            if not w or w[0] != "combobox":
                continue
            val = saved.get(key)
            values = w[1].cget("values")
            w[1].set(val if val in values else "none")

    def update_slider_label(self, key, value):
        w = self.settings_widgets[key]
        if w[0] == "slider":
            decimals = SLIDER_RANGES.get(key, (0, 100, 0))[2]
            w[2].configure(text=f"{value:.{decimals}f}")

    def save_settings(self):
        self.apply_settings()

        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_config.json")
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=2)

        self._save_guard_settings()

        self.log("Settings saved.")
        messagebox.showinfo("Settings", "Settings saved successfully.")

    def _save_guard_settings(self):
        """
        Write the guard-backed Settings widgets into scripts/guardrails.json.

        Re-read and merged rather than rewritten from anything held in memory:
        that file is meant to be edited by hand, load_guard_config() re-derives
        some of it from the machine on every load, and a save from this tab
        must not quietly stamp either of those out.
        """
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guardrails.json")
        try:
            with open(path) as handle:
                stored = json.load(handle)
        except (OSError, ValueError):
            stored = guardrails.load_guard_config()
        changed = {key: value for key, value in self.guard_settings.items()
                   if stored.get(key) != value}
        if not changed:
            return
        stored.update(self.guard_settings)
        try:
            with open(path, "w") as handle:
                json.dump(stored, handle, indent=2)
        except OSError as exc:
            self.log(f"[Settings] Could not write guardrails.json: {exc}")
            return
        for key, value in changed.items():
            self.log(f"[Settings] Guard rails: {key} is now {value} "
                     f"(scripts/guardrails.json).")

    def apply_settings(self):
        """
        Read every settings widget back into self.config, and the guard-backed
        ones into self.guard_settings.

        save_settings() is this plus a write to disk. The two used to carry
        identical copies of this loop, which is exactly the shape where a new
        setting gets added to one of them and silently never reaches a run.
        """
        for key, w in self.settings_widgets.items():
            if w[0] == "combobox":
                val = w[1].get()
                if key == "enhancer_model" and val == "none":
                    val = None
                if key == "frame_enhancer" and val == "none":
                    val = None
                if key == "upscale_method" and val == "none":
                    val = None
                if key == "expression_restorer":
                    val = (val == "on")
                if key in GUARD_SETTINGS:
                    self.guard_settings[key] = val
                else:
                    self.config[key] = val
            elif w[0] == "slider":
                # FaceFusion's CLI parses these as ints; a bare slider value
                # would arrive as "95.0" and fail argparse.
                value = w[1].get()
                if key in INT_SETTINGS:
                    value = int(round(value))
                else:
                    # A CTkSlider returns float noise (0.19999999999999996 for
                    # a 0.20 step), which then lands in the config file and the
                    # log verbatim.
                    value = round(value, SLIDER_RANGES.get(key, (0, 100, 2))[2])
                self.config[key] = value
            elif w[0] == "entry":
                raw = w[1].get().strip()
                if key == "upscale_target_width":
                    self.config[key] = int(raw) if raw else None
                elif key in INT_SETTINGS:
                    # An entry that backs an int setting has to reach the CLI as
                    # one; "0" and "" are not the same thing to argparse.
                    try:
                        self.config[key] = int(raw) if raw else 0
                    except ValueError:
                        self.config[key] = 0
                elif key in FLOAT_SETTINGS:
                    try:
                        self.config[key] = float(raw) if raw else None
                    except ValueError:
                        self.config[key] = None
                else:
                    self.config[key] = raw if raw else None
        # The clip labels quote the chunk count, and the chunk length lives in
        # the settings just read back, so a changed length shows up at once.
        for label, start_var, end_var in (
            (getattr(self, "vr_clip_label", None), getattr(self, "vr_trim_start_var", None),
             getattr(self, "vr_trim_end_var", None)),
            (getattr(self, "nm_clip_label", None), getattr(self, "nm_trim_start_var", None),
             getattr(self, "nm_trim_end_var", None)),
        ):
            if label is not None and start_var is not None:
                try:
                    self._show_clip_summary(label, self.parse_optional_float(start_var.get()),
                                            self.parse_optional_float(end_var.get()))
                except Exception:
                    pass

    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        self.log_messages.append(line)
        self.log_text.insert("end", line + "\n")

        if len(self.log_messages) > MAX_LOG_LINES:
            drop = len(self.log_messages) - LOG_TRIM_TO
            del self.log_messages[:drop]
            # Tk line numbers are 1-based, so dropping N lines means deleting
            # up to the start of line N+1.
            self.log_text.delete("1.0", f"{drop + 1}.0")

        self.log_text.see("end")

    def update_progress(self, percent, text=""):
        self.progress_bar.set(percent / 100)
        self.progress_label.configure(text=f"Progress: {percent}% - {text}")
        self.status_label.configure(text=text, text_color="#4ade80")

    def set_blocked(self, blocked):
        # A starting run is the moment worth remembering: a job the watchdog
        # kills, or that takes the machine down with it, never reaches
        # on_close, and its inputs are exactly the ones needed to resume.
        if blocked:
            self.save_session()
        self.is_running = blocked
        state = "disabled" if blocked else "normal"
        self.vr_run_btn.configure(state=state)
        self.nm_run_btn.configure(state=state)
        self.up_run_btn.configure(state=state)
        self.vr_run_btn.configure(text="Running..." if blocked else "Run VR Pipeline")
        self.nm_run_btn.configure(text="Running..." if blocked else "Run Normal Pipeline")
        self.up_run_btn.configure(text="Running..." if blocked else "Run Upscale")
        self.stop_btn.configure(state="normal" if blocked else "disabled")
        self.pause_btn.configure(state="normal" if blocked else "disabled",
                                 text="PAUSE")
        self._sync_queue_buttons()
        # The per-row Start buttons are disabled while a job runs and enabled
        # when the machine is free; rebuild the list so they track the change.
        self.refresh_queue_view()

    def log_machine_limits(self):
        """Show what the guard rails will allow, so limits are never a surprise."""
        guard_config = guardrails.load_guard_config()
        total_ram, avail_ram, _ = guardrails.memory_status()
        vram_total, _ = guardrails.gpu_memory_mb()
        temp_dir = os.path.join(PROJECT_DIR, "output", "temp")
        free_disk = guardrails.free_disk_bytes(temp_dir)

        self.log("Guard rails active. Limits for this machine:")
        self.log(f"  RAM {guardrails.fmt_gb(total_ram)} total, "
                 f"{guardrails.fmt_gb(total_ram - avail_ram)} in use by other programs, "
                 f"{guardrails.fmt_gb(avail_ram)} free for a run")
        self.log(f"  Commit limit {guardrails.fmt_gb(guardrails.commit_limit_bytes())} "
                 f"(RAM + pagefile) - pipeline capped at "
                 f"{guard_config['job_memory_cap_gb']} GB committed")
        # The commit ceiling is the limit that actually bites on this machine,
        # and unlike free RAM it is a setting rather than a fact. Say so before
        # a job that may run for days, not after it has been killed.
        info = guardrails.system_memory_breakdown()
        if info and info.get("commit_limit"):
            headroom = (info["commit_limit"] - info["commit_total"]
                        - float(guard_config["job_memory_cap_gb"]) * guardrails.GB)
            if info["commit_peak"] >= info["commit_limit"] * 0.98:
                self.log(f"  WARNING: commit charge has already hit that ceiling since "
                         f"boot. That is when Windows trims every working set and free "
                         f"RAM collapses - the usual cause of an abort here. Raising "
                         f"the pagefile raises the ceiling.")
            elif headroom < 4 * guardrails.GB:
                self.log(f"  WARNING: with the pipeline at its cap only "
                         f"{guardrails.fmt_gb(max(0, headroom))} of commit is left for "
                         f"the rest of the machine. Raising the pagefile is what buys "
                         f"more.")
        if guard_config.get("auto_pause_enabled", True):
            self.log(f"  Below {guard_config.get('auto_pause_free_ram_gb', 5.0)} GB free "
                     f"RAM a run FREEZES rather than dies - it hands its pages back and "
                     f"thaws itself once "
                     f"{guard_config.get('auto_resume_free_ram_gb', 7.5)} GB is free. "
                     f"It freezes even if RAM falls straight past "
                     f"{guard_config['min_free_ram_gb']} GB; it is only stopped if the "
                     f"memory never comes back, or if commit charge or temp disk runs "
                     f"out, which freezing cannot fix.")
        else:
            self.log(f"  A run stops automatically if free RAM drops below "
                     f"{guard_config['min_free_ram_gb']} GB - so what matters is the "
                     f"free figure above, not the total.")
        self.log(f"  CPU {os.cpu_count()} logical cores - pipeline capped at "
                 f"{guard_config['cpu_percent_cap']}%, below-normal priority "
                 f"({guard_config.get('cpu_percent_cap_unattended', 90)}% for a queued "
                 f"job, when nobody is at the keyboard)")
        self.log(f"  VRAM {vram_total} MB")
        mode = str(guard_config.get("parallel_eyes", "auto")).lower()
        if mode == "off":
            self.log("  Both eyes at once: off. Each eye is swapped in turn.")
        else:
            self.log(f"  Both eyes at once: {mode}. Before each VR run the free VRAM, "
                     f"commit and RAM are measured and the run is told whether two "
                     f"FaceFusion lanes fit; if not it swaps one eye at a time, as it "
                     f"always has. Either lane can be frozen on its own if the machine "
                     f"runs short - the other one keeps the GPU.")
        if guard_config.get("overlap_stages", True):
            self.log("  Chunked runs overlap their stages: the next chunk's split and "
                     "the previous chunk's merge run underneath the face swap, which "
                     "is otherwise using neither the encoder nor the CPU.")
        self.log(f"  Temp disk ({os.path.splitdrive(os.path.abspath(temp_dir))[0]}) "
                 f"{guardrails.fmt_gb(free_disk)} free - run stops automatically below "
                 f"{guard_config['min_free_disk_gb']} GB")
        if free_disk and free_disk < 100 * guardrails.GB:
            self.log("  NOTE: temp space is limited. Trim long clips or point "
                     "output/temp at a larger drive.")
        self.log("Tune any of this in scripts/guardrails.json.")
        self.refresh_resource_label()
        self.log_paging_setup()

    def _sync_vr_output_suffix(self, *_args):
        """Keep the generated output name in step with the layout radio."""
        if self.vr_output_var.get() != self._vr_output_auto:
            return
        suffix = vr_metadata.suffix_for(self.vr_layout_var.get())
        self._vr_output_auto = self._vr_output_stem + suffix + ".mp4"
        self.vr_output_var.set(self._vr_output_auto)

    def log_paging_setup(self):
        """
        Warn once, at startup, if Windows is paging to a mechanical disk.

        On a machine whose RAM is smaller than what a VR run commits, this is
        the difference between a run that finishes and one that locks the
        desktop and gets killed by the watchdog. It is worth saying before the
        user starts a two-hour job rather than after.

        Off the UI thread: the query shells out to PowerShell and takes about a
        second, and startup was deliberately made instant.
        """
        def probe():
            try:
                warning = guardrails.slow_pagefile_warning()
            except Exception:
                return
            if warning:
                self.after(0, self.log, f"[Guard] {warning}")

        threading.Thread(target=probe, name="vrdeep-paging", daemon=True).start()

    def start_vram_sampler(self):
        """Poll nvidia-smi on a background thread so the UI never waits on it."""
        def loop():
            while not self._vram_stop.wait(3.0):
                try:
                    _, used = guardrails.gpu_memory_mb()
                    self._vram_used_mb = used
                except Exception:
                    pass

        self._vram_thread = threading.Thread(target=loop, name="vrdeep-vram", daemon=True)
        self._vram_thread.start()

    def refresh_resource_label(self):
        """Live RAM / disk readout in the log header, refreshed every 3s."""
        try:
            # Both of these are cheap ctypes calls; the VRAM figure is whatever
            # the background sampler last saw, never a fresh nvidia-smi.
            total_ram, avail_ram, _ = guardrails.memory_status()
            free_disk = guardrails.free_disk_bytes(os.path.join(PROJECT_DIR, "output", "temp"))
            self.resource_label.configure(
                text=f"RAM {guardrails.fmt_gb(total_ram - avail_ram)} used / "
                     f"{guardrails.fmt_gb(total_ram)} - "
                     f"{guardrails.fmt_gb(avail_ram)} free for this run  |  "
                     f"Temp disk free {guardrails.fmt_gb(free_disk)}  |  "
                     f"VRAM used {self._vram_used_mb} MB"
            )
        except Exception:
            pass
        self._sync_pause_button()
        self.after(3000, self.refresh_resource_label)

    def stop_run(self):
        """
        Kill the whole pipeline process tree immediately.

        With a queue running this also pauses the queue: someone pressing STOP
        wants the machine back, and starting the next job a second later is
        the opposite of that. The waiting jobs are kept; Start Queue resumes.
        """
        if self.is_running:
            self._stop_requested = True
            if self.queue_active:
                self.queue_active = False
                self.log("[Queue] Paused - the job in flight is being stopped and the "
                         "next one will not start. Press Start Queue to carry on.")
            if self.guard:
                self.log("Stop requested - terminating all pipeline processes...")
                self.guard.cancel("Stopped by user")
                self.status_label.configure(text="Stopping...", text_color="#f59e0b")
            else:
                self.set_blocked(False)
            return
        if self.queue_active:
            self.pause_queue("Paused while waiting for a start time. The jobs are "
                             "still in the queue; Start Queue arms it again, Remove "
                             "on the Queue tab drops one.")

    # -- pause -------------------------------------------------------------

    def toggle_pause(self):
        """
        Freeze the run where it stands, or thaw it.

        Not the same offer as Stop. Stop ends the run and gives back the RAM
        and the VRAM, keeping every stage that had finished - but the stage in
        flight is lost, and at 8K that is up to an hour of swapping. Pause
        keeps the run alive and exactly where it was, at the cost of holding
        its memory. So: Pause to use the CPU for something else, Stop to get
        the graphics card back.
        """
        if not self.guard or not self.is_running:
            return
        if self.guard.paused:
            self.guard.resume()
            self.pause_btn.configure(text="PAUSE", fg_color="#a16207",
                                     hover_color="#ca8a04")
            self._set_row_pause_label(False)
            self.update_status("Running", "#4ade80")
            self.log("Resumed. The paused time is not counted against the estimate.")
        else:
            if not self.guard.pause():
                return
            self._pause_baseline = self.guard.paused_seconds()
            self.pause_btn.configure(text="RESUME", fg_color="#15803d",
                                     hover_color="#16a34a")
            self._set_row_pause_label(True)
            self.update_status("Paused", "#f59e0b")
            self.log("Paused. Nothing is lost - Resume picks the run up mid-frame. "
                     "The run keeps its RAM and VRAM while frozen; use STOP to get "
                     "those back (finished stages are kept either way).")
            self._pause_tick()

    def _pause_tick(self):
        """Keep the progress line honest about a run that is not moving."""
        if not (self.guard and self.guard.paused and self.is_running):
            return
        held = self.guard.paused_seconds() - self._pause_baseline
        if getattr(self.guard, "paused_by", None) == "guard":
            text = (f"FROZEN for {guardrails.fmt_duration(held)} - the machine ran "
                    f"low on memory; the run thaws itself when it comes back")
        else:
            text = (f"PAUSED for {guardrails.fmt_duration(held)} - "
                    f"press RESUME to carry on where it stopped")
        self.progress_label.configure(text=text)
        self.after(1000, self._pause_tick)

    def _sync_pause_button(self):
        """
        Follow the guard, which can freeze a run without being asked.

        The watchdog freezes rather than aborts when memory gets tight, so the
        button has to be able to say RESUME without anyone having pressed
        PAUSE - otherwise pressing it would resume a run the user believed
        they were pausing.
        """
        frozen = bool(self.guard and self.guard.paused)
        wanted = "RESUME" if frozen else "PAUSE"
        # The running job's queue row carries the same control; keep its label
        # following the guard too, since a guard-driven freeze moves both.
        row_btn = self._queue_row_pause_btn
        if row_btn is not None:
            try:
                row_btn.configure(
                    text="Resume" if frozen else "Pause",
                    fg_color="#15803d" if frozen else "#a16207",
                    hover_color="#16a34a" if frozen else "#ca8a04")
            except Exception:
                self._queue_row_pause_btn = None
        if self.pause_btn.cget("text") == wanted:
            return
        self.pause_btn.configure(
            text=wanted,
            fg_color="#15803d" if frozen else "#a16207",
            hover_color="#16a34a" if frozen else "#ca8a04")
        if frozen:
            by = getattr(self.guard, "paused_by", None)
            self.update_status("Frozen - low memory" if by == "guard" else "Paused",
                               "#f59e0b")
            self._pause_baseline = self.guard.paused_seconds()
            self._pause_tick()
        elif self.is_running:
            self.update_status("Running", "#4ade80")

    def _set_row_pause_label(self, frozen):
        """Point the running row's Pause/Resume button at the state just set."""
        row_btn = self._queue_row_pause_btn
        if row_btn is None:
            return
        try:
            row_btn.configure(
                text="Resume" if frozen else "Pause",
                fg_color="#15803d" if frozen else "#a16207",
                hover_color="#16a34a" if frozen else "#ca8a04")
        except Exception:
            self._queue_row_pause_btn = None

    # -- job queue -----------------------------------------------------------
    #
    # A job is a dict: kind, fields (a frozen copy of the tab), config (the
    # settings in force when it was added; None for an upscale), not_before,
    # status, note, and timings. Every run goes through the queue - an
    # immediate Run is a job that starts at once - so the Queue tab is also
    # the history of what this window has done.

    def build_queue_tab(self):
        tab = self.tabview.tab("Queue")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(3, weight=1)
        pad_x = 10

        ctk.CTkLabel(
            tab, text="Jobs run one after another, top to bottom, one at a time. A job "
                      "with a time against it waits for that time; the ones without "
                      "run as soon as the machine is free. Add jobs from the other "
                      "tabs with Add to Queue, or by pressing Run with a Start At time.",
            font=ctk.CTkFont(size=11), text_color="gray", justify="left", wraplength=1100
        ).grid(row=0, column=0, sticky="w", padx=pad_x, pady=(8, 6))

        bar = ctk.CTkFrame(tab, fg_color="transparent")
        bar.grid(row=1, column=0, sticky="ew", padx=pad_x, pady=(0, 4))
        self.queue_start_btn = ctk.CTkButton(
            bar, text="Start Queue", width=130, height=34,
            font=ctk.CTkFont(size=12, weight="bold"), command=self.start_queue)
        self.queue_start_btn.pack(side="left", padx=(0, 6))
        self.queue_pause_btn = ctk.CTkButton(
            bar, text="Pause Queue", width=130, height=34, fg_color="#a16207",
            hover_color="#ca8a04", font=ctk.CTkFont(size=12, weight="bold"),
            command=lambda: self.pause_queue(
                "Paused. The job in flight, if any, finishes; the next one waits "
                "for Start Queue."))
        self.queue_pause_btn.pack(side="left", padx=(0, 6))
        ctk.CTkButton(bar, text="Re-queue Failed", width=130, height=34, fg_color="#374151",
                      hover_color="#4b5563", command=self.requeue_failed).pack(
            side="left", padx=(0, 6))
        ctk.CTkButton(bar, text="Clear Finished", width=130, height=34, fg_color="#374151",
                      hover_color="#4b5563", command=self.clear_finished).pack(
            side="left", padx=(0, 6))
        self.queue_summary = ctk.CTkLabel(bar, text="", font=ctk.CTkFont(size=11),
                                          text_color="#888888")
        self.queue_summary.pack(side="left", padx=(12, 0))

        opts = ctk.CTkFrame(tab, fg_color="transparent")
        opts.grid(row=2, column=0, sticky="ew", padx=pad_x, pady=(0, 6))
        self.queue_stop_on_failure_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(opts, text="Stop the queue when a job fails",
                        variable=self.queue_stop_on_failure_var,
                        command=self.save_queue).pack(side="left", padx=(0, 20))
        self.queue_keep_order_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(opts, text="Keep strict order: wait for a timed job rather "
                                   "than running the ones after it first",
                        variable=self.queue_keep_order_var,
                        command=self.save_queue).pack(side="left")

        self.queue_rows = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        self.queue_rows.grid(row=3, column=0, sticky="nsew", padx=pad_x, pady=(0, 6))
        self.queue_rows.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            tab, text="Not before: a clock time (02:30), tomorrow 02:30, +90m, or a "
                      "date and time - type it and press Enter. Empty means as soon "
                      "as possible. Edit puts a job back on its tab to change it.",
            font=ctk.CTkFont(size=10), text_color="#888888", justify="left",
            wraplength=1100
        ).grid(row=4, column=0, sticky="w", padx=pad_x, pady=(0, 6))

        self.refresh_queue_view()

    def refresh_queue_view(self):
        """Rebuild the job list. The list is short; rebuilding beats diffing."""
        rows = getattr(self, "queue_rows", None)
        if rows is None:
            return
        # Rebuilt below; the old one is about to be destroyed with its row.
        self._queue_row_pause_btn = None
        for child in rows.winfo_children():
            child.destroy()

        if not self.queue:
            ctk.CTkLabel(rows, text="The queue is empty. Fill in a tab and press "
                                    "Add to Queue.",
                         text_color="#888888").grid(row=0, column=0, sticky="w", pady=10)
        for index, job in enumerate(self.queue):
            self._build_queue_row(rows, index, job)

        counts = {}
        for job in self.queue:
            counts[job["status"]] = counts.get(job["status"], 0) + 1
        parts = [f"{n} {status}" for status, n in counts.items()]
        state = "running" if self.queue_active else "paused"
        self.queue_summary.configure(
            text=(f"{len(self.queue)} job(s): " + ", ".join(parts) + f"  -  queue {state}")
            if self.queue else "")
        self._sync_queue_buttons()

    def _build_queue_row(self, parent, index, job):
        status = job["status"]
        frame = ctk.CTkFrame(parent)
        frame.grid(row=index, column=0, sticky="ew", pady=3)
        frame.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(frame, text=str(index + 1), width=28,
                     font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=0, column=0, rowspan=2, padx=(8, 4))

        when = job_not_before(job)
        if status == "waiting" and when is not None:
            status_text = "At " + when.strftime("%a %H:%M")
        else:
            status_text = status.capitalize()
        ctk.CTkLabel(frame, text=status_text, width=110, anchor="w",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=JOB_STATUS_COLORS.get(status, "#ffffff")).grid(
            row=0, column=1, rowspan=2, padx=(0, 6), sticky="w")

        ctk.CTkLabel(frame, text=job_title(job), anchor="w",
                     font=ctk.CTkFont(size=12, weight="bold")).grid(
            row=0, column=2, sticky="w", padx=(0, 6), pady=(4, 0))
        ctk.CTkLabel(frame, text=job_detail(job), anchor="w",
                     font=ctk.CTkFont(size=10), text_color="#aaaaaa").grid(
            row=1, column=2, sticky="w", padx=(0, 6), pady=(0, 4))

        if status == "waiting":
            ctk.CTkLabel(frame, text="Not before", font=ctk.CTkFont(size=10),
                         text_color="#888888").grid(row=0, column=3, rowspan=2, padx=(0, 4))
            entry = ctk.CTkEntry(frame, width=150, font=ctk.CTkFont(size=11))
            entry.insert(0, when.strftime("%Y-%m-%d %H:%M") if when else "")
            entry.grid(row=0, column=4, rowspan=2, padx=(0, 10))
            entry.bind("<Return>", lambda _e, j=job, w=entry: self._set_job_time(j, w))

        buttons = ctk.CTkFrame(frame, fg_color="transparent")
        buttons.grid(row=0, column=5, rowspan=2, padx=(0, 8))
        small = dict(width=34, height=26, font=ctk.CTkFont(size=11))
        if status == "running":
            # The run's controls, on the row that is actually running, so the
            # machine can be paused or stopped without leaving the Queue tab.
            # The bottom-bar PAUSE/STOP do the same thing; _sync_pause_button
            # keeps this button's label following the guard, which can freeze a
            # run on its own when memory gets tight.
            pause_text = "Resume" if (self.guard and self.guard.paused) else "Pause"
            self._queue_row_pause_btn = ctk.CTkButton(
                buttons, text=pause_text, width=64, height=26,
                font=ctk.CTkFont(size=11), fg_color="#a16207", hover_color="#ca8a04",
                command=self.toggle_pause)
            self._queue_row_pause_btn.pack(side="left", padx=2)
            ctk.CTkButton(buttons, text="Stop", width=56, height=26,
                          font=ctk.CTkFont(size=11), fg_color="#b91c1c",
                          hover_color="#dc2626", command=self.stop_run).pack(
                side="left", padx=2)
            return
        if status == "waiting":
            ctk.CTkButton(buttons, text="\u25b2", command=lambda j=job: self.move_job(j, -1),
                          fg_color="#374151", hover_color="#4b5563", **small).pack(
                side="left", padx=2)
            ctk.CTkButton(buttons, text="\u25bc", command=lambda j=job: self.move_job(j, 1),
                          fg_color="#374151", hover_color="#4b5563", **small).pack(
                side="left", padx=2)
            # Run this one now, ahead of its start time and of the jobs above
            # it. Disabled while anything is running - the machine does one job
            # at a time. The rest of the queue is left exactly as it was.
            ctk.CTkButton(buttons, text="Start", width=52, height=26,
                          font=ctk.CTkFont(size=11), fg_color="#166534",
                          hover_color="#15803d",
                          state="disabled" if self.is_running else "normal",
                          command=lambda j=job: self.start_job_now(j)).pack(
                side="left", padx=2)
            ctk.CTkButton(buttons, text="Edit", width=52, height=26,
                          font=ctk.CTkFont(size=11), fg_color="#374151",
                          hover_color="#4b5563",
                          command=lambda j=job: self.edit_job(j)).pack(side="left", padx=2)
        else:
            ctk.CTkButton(buttons, text="Re-queue", width=72, height=26,
                          font=ctk.CTkFont(size=11), fg_color="#374151",
                          hover_color="#4b5563",
                          command=lambda j=job: self.requeue_job(j)).pack(side="left", padx=2)
        ctk.CTkButton(buttons, text="Remove", width=64, height=26,
                      font=ctk.CTkFont(size=11), fg_color="#7f1d1d",
                      hover_color="#991b1b",
                      command=lambda j=job: self.remove_job(j)).pack(side="left", padx=2)

    def _sync_queue_buttons(self):
        """The queue's own buttons, and STOP while the queue waits with nothing running."""
        start_btn = getattr(self, "queue_start_btn", None)
        if start_btn is None:
            return
        waiting = any(j["status"] == "waiting" for j in self.queue)
        if self.queue_active:
            start_btn.configure(state="disabled", text="Queue Running")
            self.queue_pause_btn.configure(state="normal")
        else:
            start_btn.configure(state="normal" if waiting else "disabled",
                                text="Start Queue")
            self.queue_pause_btn.configure(state="disabled")
        if self.queue_active and not self.is_running:
            self.stop_btn.configure(state="normal")

    # -- building and running jobs ----------------------------------------------

    def _tab_fields(self, kind):
        fields = {}
        for name, attr in JOB_FIELD_VARS[kind].items():
            value = getattr(self, attr).get()
            fields[name] = value.strip() if isinstance(value, str) else value
        if kind == "upscale":
            fields["quality"] = int(round(self.up_quality_slider.get()))
        return fields

    def _build_job(self, kind):
        """
        Freeze the tab into a job, checking it the way Run always has.

        Returns None after showing the problem. The checks happen here, at the
        moment of adding, so a queue armed for the night is a queue of jobs
        that could actually start; the files are checked again at start time
        because a night is long enough for a drive to be unplugged.
        """
        fields = self._tab_fields(kind)
        config = None
        if kind == "upscale":
            if not fields["input"] or not fields["output"]:
                messagebox.showerror("Error", "Please choose an input video and an output path.")
                return None
            if not os.path.exists(fields["input"]):
                messagebox.showerror("Error", "Input not found: " + fields["input"])
                return None
        else:
            # The same two calls Run has always made before starting, so the
            # snapshot is exactly what a run pressed now would use. The copy is
            # what lets the Settings tab be changed while this job waits.
            self._apply_preset(fields["preset"])
            self.apply_settings()
            config = copy.deepcopy(self.config)
            if not fields["source"] or not fields["target"] or not fields["output"]:
                messagebox.showerror("Error", "Please fill in source face, target video, and output path.")
                return None
            if self.check_source(fields["source"]) is None:
                return None
            if not os.path.exists(fields["target"]):
                messagebox.showerror("Error", f"Target video not found: {fields['target']}")
                return None
        schedule_var = getattr(self, JOB_SCHEDULE_VARS[kind])
        try:
            when = parse_schedule_time(schedule_var.get())
        except ValueError as exc:
            messagebox.showerror("Start At", str(exc))
            return None
        return {
            "id": f"job_{int(time.time() * 1000)}",
            "kind": kind,
            "fields": fields,
            "config": config,
            "not_before": when.strftime(NOT_BEFORE_FMT) if when else None,
            "status": "waiting",
            "note": "",
            "added": datetime.now().strftime(NOT_BEFORE_FMT),
            "started": None,
            "finished": None,
            "elapsed": None,
        }

    def _output_taken(self, job):
        """Another unfinished job already writing to this job's output path."""
        output = os.path.normcase(os.path.abspath(job["fields"]["output"]))
        for other in self.queue:
            if other is job or other["status"] not in ("waiting", "running"):
                continue
            if os.path.normcase(os.path.abspath(other["fields"]["output"])) == output:
                return other
        return None

    def _job_problem(self, job):
        """A reason this job cannot start right now, or None."""
        f = job["fields"]
        if job["kind"] == "upscale":
            if not os.path.exists(f["input"]):
                return f"Input not found: {f['input']}"
            return None
        missing = [p for p in split_sources(f["source"]) if not os.path.exists(p)]
        if missing:
            return f"Source face not found: {missing[0]}"
        if not os.path.exists(f["target"]):
            return f"Target video not found: {f['target']}"
        return None

    def add_to_queue(self, kind):
        job = self._build_job(kind)
        if job is None:
            return
        if not self._enqueue(job):
            return
        when = job_not_before(job)
        self.log(f"[Queue] Added job {len(self.queue)}: {job_title(job)}"
                 + (f", not before {when.strftime('%a %d %b %H:%M')}" if when else "")
                 + ". Press Start Queue on the Queue tab when the list is ready"
                 + (" - or leave it, the queue is running." if self.queue_active else "."))

    def _enqueue(self, job):
        taken = self._output_taken(job)
        if taken is not None:
            messagebox.showerror(
                "Output already queued",
                f"Job {self.queue.index(taken) + 1} ({job_title(taken)}) already writes "
                f"to\n{job['fields']['output']}\n\nChoose a different output path.")
            return False
        self.queue.append(job)
        getattr(self, JOB_SCHEDULE_VARS[job["kind"]]).set("")
        # A queued render owns its name; the tab gets a fresh one so the next
        # Add to Queue cannot silently point at the same file.
        if job["fields"]["output"] in (self._vr_output_auto, self._nm_output_auto):
            self._retire_auto_output()
        self.save_queue()
        self.refresh_queue_view()
        return True

    def _run_now_or_queue(self, kind):
        """Run: at once, or - with a Start At time - as a queued job for that time."""
        job = self._build_job(kind)
        if job is None:
            return
        if job["not_before"]:
            if not self._enqueue(job):
                return
            when = job_not_before(job)
            self.log(f"[Schedule] {job_title(job)} armed for "
                     f"{when.strftime('%a %d %b %H:%M:%S')} as queue job "
                     f"{len(self.queue)}. This window has to stay open; STOP pauses "
                     f"the queue, Remove on the Queue tab cancels it.")
            if not self.queue_active:
                self.start_queue()
            return
        if self.is_running:
            return
        if self._output_taken(job) is not None:
            taken = self._output_taken(job)
            messagebox.showerror(
                "Output already queued",
                f"Queue job {self.queue.index(taken) + 1} already writes to\n"
                f"{job['fields']['output']}\n\nChoose a different output path.")
            return
        self.queue.append(job)
        getattr(self, JOB_SCHEDULE_VARS[kind]).set("")
        if not self._launch_job(job, interactive=True):
            self.queue.remove(job)
        self.save_queue()
        self.refresh_queue_view()

    def _launch_job(self, job, interactive):
        """
        Start one job on this machine now. True once it is running.

        `interactive` is whether someone is at the keyboard: an immediate Run
        may ask before forcing past the guard rails and gets a dialog at the
        end; a queued job never blocks on a dialog nobody will answer.
        """
        kind = job["kind"]
        f = job["fields"]
        problem = self._job_problem(job)
        if problem:
            self.log(f"[Job] Not started - {problem}")
            self._mark_job(job, "failed", problem)
            if interactive:
                messagebox.showerror("Error", problem)
            return False

        position = self.queue.index(job) + 1 if job in self.queue else 0
        self.log("=" * 50)
        self.log(f"{JOB_HEADINGS[kind]} STARTED"
                 + (f"  (queue job {position} of {len(self.queue)})"
                    if not interactive else ""))

        if kind == "upscale":
            self.log(f"Input: {f['input']}")
            self.log(f"Output: {f['output']}")
            guard_config = guardrails.load_guard_config()
            prepared = None
        else:
            is_vr = kind == "vr"
            self.log(f"Quality preset: {f['preset']}")
            self.log(self.check_source(f["source"]) or "")
            self.log(f"Source: {f['source']}")
            self.log(f"Target: {f['target']}")
            if is_vr:
                self.log(f"Layout: {f['layout'].upper()}")
            self.log(f"Output: {f['output']}")
            trim_start = self.parse_optional_float(f["trim_start"])
            trim_end = self.parse_optional_float(f["trim_end"])
            # The run log is the only record of what a two-hour job actually
            # processed, so it has to say which portion of the target that was.
            if trim_start is not None or trim_end is not None:
                self.log(f"Clip: {clip_selector.format_timecode(trim_start) or 'start'} to "
                         f"{clip_selector.format_timecode(trim_end) or 'end'}")
            self._log_chunk_plan(f["target"], trim_start, trim_end, f["output"],
                                 config=job["config"])
            prepared = self.preflight_or_confirm(
                f["target"], is_vr, trim_start, trim_end,
                unattended=not interactive, config=job["config"])
            if prepared is None:
                error = self._last_preflight_error
                # Unattended only. Someone sitting in front of the app has
                # already been offered the override dialog and said no, and
                # holding their job for five minutes after that is not what
                # they asked for.
                if not interactive and getattr(error, "transient", False):
                    self._defer_job(job, error)
                    return False
                self._mark_job(job, "failed", "Not started: the guard rails refused it "
                                              "(see the log)")
                return False
            guard_config = prepared[1]
            # It got past the preflight, so whatever it was waiting for is
            # back. Clear the deferral bookkeeping or the give-up clock would
            # keep counting from the first deferral of a job that has since run.
            job.pop("retry_after", None)
            job.pop("deferred_since", None)

        self._active_job = job
        self._active_interactive = interactive
        self._stop_requested = False
        job["started"] = datetime.now().strftime(NOT_BEFORE_FMT)
        self._mark_job(job, "running")
        self.set_blocked(True)
        self.progress_bar.set(0)
        if prepared is not None:
            self.show_estimate(prepared)
        self.guard = ResourceGuard(
            os.path.join(PROJECT_DIR, "output", "temp"), guard_config,
            log_cb=lambda m: self.after(0, self.log, m),
            # A queued job runs under the higher CPU cap: nobody is waiting on
            # the desktop at 03:00, and the log says which cap was applied.
        ).start(unattended=not interactive)
        self.current_thread = threading.Thread(
            target=self._job_thread, args=(job, prepared), daemon=True)
        self.current_thread.start()
        return True

    def _defer_job(self, job, error):
        """
        Hold a queued job that was refused for memory, instead of failing it.

        The job on Sep 10 01:16 was refused because another program still had
        8 GB of commit charged. It failed, the queue moved on, and the machine
        spent the night idle over a condition that clears by itself in minutes.
        Nothing is retried harder here - the retry is a fresh preflight, so the
        job starts the moment the memory is genuinely back and not a second
        before - it is only not thrown away in the meantime.
        """
        guard_config = guardrails.load_guard_config()
        minutes = float(guard_config.get("queue_retry_minutes", 5) or 5)
        hours = float(guard_config.get("queue_retry_hours", 6) or 6)
        now = datetime.now()
        reason = preflight_reason_line(error)
        first = not job.get("deferred_since")
        if first:
            job["deferred_since"] = now.strftime(NOT_BEFORE_FMT)
        retry_at = now + timedelta(minutes=minutes)
        job["retry_after"] = retry_at.strftime(NOT_BEFORE_FMT)
        index = self.queue.index(job) + 1 if job in self.queue else 0
        self.log(f"[Queue] Job {index} deferred - waiting for memory, retrying at "
                 f"{retry_at.strftime('%H:%M')} (reason: {reason})")
        if first:
            self.log(f"[Queue] The queue is NOT stopping: this job stays where it is and "
                     f"is tried again every {minutes:g} min for up to {hours:g} h. Only "
                     f"then does it count as failed. Closing whatever the log named "
                     f"above makes it start at the next attempt.")
        self._mark_job(job, "deferred", f"Waiting for memory: {reason}")

    def _deferral_expired(self, job, now):
        """
        True once a deferred job has waited longer than the queue will wait.

        The memory that was going to come back has come back by then; what is
        left is a machine that is simply too busy for this run, and a job that
        silently retried forever would look exactly like a queue that had
        stopped working. Marked failed with the reason from the last attempt.
        """
        since = _job_time(job, "deferred_since")
        if since is None:
            return False
        hours = float(guardrails.load_guard_config().get("queue_retry_hours", 6) or 6)
        if now - since < timedelta(hours=hours):
            return False
        index = self.queue.index(job) + 1 if job in self.queue else 0
        note = job.get("note") or "Waiting for memory"
        self.log(f"[Queue] Job {index} gave up after {hours:g} h of waiting for memory - "
                 f"marking it failed. Last reason: {note}")
        job.pop("retry_after", None)
        self._mark_job(job, "failed", note)
        return True

    def _job_thread(self, job, prepared):
        start_time = time.time()
        kind = job["kind"]
        f = job["fields"]
        status, note = "done", ""
        log_cb = lambda m: self.after(0, self.log, m)
        progress_cb = lambda p, t: self.after(0, self.update_progress, p, t)
        try:
            if kind == "upscale":
                code = upscale_vr.run_upscale(
                    self._upscale_args_from(f), log_fn=log_cb,
                    progress_cb=progress_cb, guard=self.guard)
                if code != 0:
                    status, note = "failed", "See the log for what stopped it."
            else:
                extra = {"layout": f["layout"]} if kind == "vr" else {}
                run_chunked_pipeline(
                    kind == "vr",
                    source_face=f["source"],
                    target_video=f["target"],
                    output_path=f["output"],
                    config=job["config"],
                    fps=self.parse_optional_float(f["fps"]),
                    trim_start=self.parse_optional_float(f["trim_start"]),
                    trim_end=self.parse_optional_float(f["trim_end"]),
                    keep_temp=bool(f.get("keep_temp")),
                    resume=bool(f.get("resume", True)),
                    progress_cb=progress_cb,
                    log_cb=log_cb,
                    guard=self.guard,
                    prepared=prepared,
                    refresh_source_faces=bool(f.get("refresh_faces")),
                    **extra,
                )
        except PipelineError as e:
            status, note = "failed", str(e)
        except Exception as e:
            status, note = "failed", f"Unexpected error: {e}"
        self.after(0, self._job_finished, job, status, note, time.time() - start_time)

    def _job_finished(self, job, status, note, elapsed):
        """Back on the UI thread: record the outcome, tidy up, move the queue on."""
        interactive = self._active_interactive
        kind = job["kind"]
        output = job["fields"]["output"]
        if status == "failed" and self._stop_requested:
            status, note = "stopped", "Stopped by user"
        self._stop_requested = False
        self._active_job = None

        if status == "done":
            self.log(f"{JOB_DONE_TEXT[kind]} ({elapsed:.0f}s)")
            self.log(f"Output: {output}")
            if kind == "upscale":
                self.update_progress(100, "Upscale complete")
            self.update_status("Done", "#4ade80")
            # A finished render owns its name; the next job gets a fresh one -
            # but only when the tab still points at this job's file.
            if output in (self._vr_output_auto, self._nm_output_auto):
                self._retire_auto_output()
        elif status == "stopped":
            self.log(f"{JOB_HEADINGS[kind].title()} stopped. ({elapsed:.0f}s)")
            self.update_status("Stopped", "#f59e0b")
        else:
            self.log(f"Pipeline failed: {note}")
            self.update_status("Failed", "#ef4444")

        job["finished"] = datetime.now().strftime(NOT_BEFORE_FMT)
        job["elapsed"] = round(elapsed)
        self._mark_job(job, status, note)

        # Tear the guard down off the UI thread. close() now waits for the
        # run's own processes to actually exit so their committed memory is
        # back before the next job is sized (see ResourceGuard.close) - without
        # that, a queue's next job saw the finished job's ~13 GB of CUDA commit
        # as still in use and the preflight refused it. That wait must not
        # freeze the window, so it runs on a worker thread, and the queue does
        # not move on - nor do the Run buttons re-enable - until it is done.
        guard, self.guard = self.guard, None
        self.vr_refresh_faces_var.set(False)
        self.nm_refresh_faces_var.set(False)
        self.progress_label.configure(text="Releasing memory...")
        threading.Thread(
            target=self._teardown_guard,
            args=(guard, job, status, note, elapsed, interactive),
            daemon=True).start()

    def _teardown_guard(self, guard, job, status, note, elapsed, interactive):
        """Worker thread: release the finished run's memory, then resume on the UI thread."""
        if guard:
            guard.close()
        # close() waited for the run's own processes to exit; this waits the
        # extra beat for the kernel to finish handing their committed memory
        # back, so the next queued job is never sized against memory that is
        # still mid-release. Bounded, and on this worker thread, so the window
        # stays live throughout.
        guardrails.wait_for_commit_release(log_cb=lambda m: self.after(0, self.log, m))
        self.after(0, self._after_teardown, job, status, note, elapsed, interactive)

    def _after_teardown(self, job, status, note, elapsed, interactive):
        """On the UI thread once the memory is freed: report, and move the queue on."""
        kind = job["kind"]
        output = job["fields"]["output"]
        # The pipeline builds a fair number of short-lived dicts and lists per
        # run; collecting here keeps the idle footprint flat between runs.
        gc.collect()
        self.set_blocked(False)

        if interactive:
            if status == "done":
                messagebox.showinfo(
                    "Success",
                    f"{JOB_DONE_TEXT[kind]}\n\nOutput: {output}\n"
                    f"Time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
            elif status == "failed":
                messagebox.showerror("Error", note)

        if self.queue_active:
            if status != "done" and self.queue_stop_on_failure_var.get():
                self.pause_queue(f"Stopped because job {self.queue.index(job) + 1} "
                                 f"{status}. Fix it, then press Start Queue.")
                return
            self._queue_after = self.after(500, self._queue_tick)

    def _mark_job(self, job, status, note=None):
        job["status"] = status
        if note is not None:
            job["note"] = note
        self.save_queue()
        self.refresh_queue_view()

    # -- the queue itself --------------------------------------------------------

    def start_queue(self):
        if self.queue_active:
            return
        waiting = [j for j in self.queue if j["status"] == "waiting"]
        if not waiting:
            messagebox.showinfo("Queue", "Nothing is waiting. Add a job from one of the "
                                         "other tabs first.")
            return
        self.queue_active = True
        timed = [j for j in waiting if job_not_before(j)]
        self.log(f"[Queue] Started: {len(waiting)} job(s) waiting"
                 + (f", {len(timed)} with a start time" if timed else "")
                 + ". This window has to stay open; STOP or Pause Queue holds it.")
        for job in self.queue:
            self._log_job_line(job)
        self.refresh_queue_view()
        self._queue_tick()

    def pause_queue(self, why):
        if self._queue_after is not None:
            try:
                self.after_cancel(self._queue_after)
            except Exception:
                pass
        self._queue_after = None
        was_active = self.queue_active
        self.queue_active = False
        if was_active:
            self.log(f"[Queue] {why}")
        if not self.is_running:
            self.update_status("Ready", "#4ade80")
            self.update_progress(0, "Progress: --")
            self.set_blocked(False)
        self.refresh_queue_view()

    def _queue_tick(self):
        """Once a second while the queue is armed and nothing is running."""
        self._queue_after = None
        if not self.queue_active or self.is_running:
            return
        now = datetime.now()
        due, earliest = None, None
        for job in self.queue:
            if job["status"] != "waiting":
                continue
            when = job_not_before(job)
            if when is None or when <= now:
                due = job
                break
            if earliest is None or when < earliest[0]:
                earliest = (when, job)
            if self.queue_keep_order_var.get():
                # Strict order: nothing behind a timed job may overtake it.
                break

        if due is None and earliest is None:
            self.queue_active = False
            counts = {}
            for job in self.queue:
                counts[job["status"]] = counts.get(job["status"], 0) + 1
            summary = ", ".join(f"{n} {s}" for s, n in counts.items())
            self.log(f"[Queue] Finished - nothing left waiting ({summary}).")
            self.update_status("Queue finished", "#4ade80")
            self.update_progress(0, f"Queue finished: {summary}")
            self.set_blocked(False)
            return

        if due is None:
            when, job = earliest
            left = (when - now).total_seconds()
            self.update_status("Scheduled", "#60a5fa")
            self.update_progress(
                0, f"Next: {job_title(job)} at {when.strftime('%H:%M:%S')} - "
                   f"{guardrails.fmt_duration(left)} to go")
            self.stop_btn.configure(state="normal")
            self._queue_after = self.after(1000, self._queue_tick)
            return

        self.log(f"[Queue] Starting job {self.queue.index(due) + 1} of {len(self.queue)}: "
                 f"{job_title(due)}")
        if not self._launch_job(due, interactive=False):
            if self.queue_stop_on_failure_var.get():
                self.pause_queue(f"Stopped because job {self.queue.index(due) + 1} could "
                                 f"not start. Fix it, then press Start Queue.")
                return
            self._queue_after = self.after(200, self._queue_tick)

    def _log_job_line(self, job):
        when = job_not_before(job)
        index = self.queue.index(job) + 1
        if job["status"] == "waiting":
            timing = (f"not before {when.strftime('%a %d %b %H:%M')}" if when
                      else "as soon as the machine is free")
        else:
            timing = job["status"]
        self.log(f"  {index:>2}. {job_title(job)}  [{timing}]")

    # -- editing the list --------------------------------------------------------

    def _set_job_time(self, job, entry):
        try:
            when = parse_schedule_time(entry.get())
        except ValueError as exc:
            messagebox.showerror("Not before", str(exc))
            return
        job["not_before"] = when.strftime(NOT_BEFORE_FMT) if when else None
        self.log(f"[Queue] Job {self.queue.index(job) + 1} ({job_title(job)}) now starts "
                 + (f"no earlier than {when.strftime('%a %d %b %H:%M')}." if when
                    else "as soon as the machine is free."))
        self.save_queue()
        # The entry that fired this is destroyed by the rebuild; let the event
        # finish first.
        self.after(0, self.refresh_queue_view)

    def move_job(self, job, delta):
        index = self.queue.index(job)
        target = index + delta
        if target < 0 or target >= len(self.queue):
            return
        self.queue[index], self.queue[target] = self.queue[target], self.queue[index]
        self.save_queue()
        self.refresh_queue_view()

    def start_job_now(self, job):
        """
        Run one waiting job immediately, ahead of its start time and the order.

        The machine does one job at a time, so this is refused while anything
        is running. The rest of the queue is untouched: if it was armed it
        carries on from wherever it was once this job finishes; if it was not,
        only this job runs. A start time on the job is honoured no longer -
        pressing Start is the explicit "now" the time was standing in for.
        """
        if self.is_running:
            messagebox.showinfo("Queue", "A job is already running - the machine does "
                                         "one at a time. Wait for it, or press Stop first.")
            return
        if job not in self.queue or job["status"] != "waiting":
            return
        # A scheduled tick could otherwise fire between here and the launch and
        # start a different job into the same one-at-a-time slot.
        if self._queue_after is not None:
            try:
                self.after_cancel(self._queue_after)
            except Exception:
                pass
            self._queue_after = None
        self.log(f"[Queue] Starting job {self.queue.index(job) + 1} of {len(self.queue)} "
                 f"now by request: {job_title(job)}")
        self._launch_job(job, interactive=True)

    def remove_job(self, job):
        if job["status"] == "running":
            return
        self.queue.remove(job)
        self.log(f"[Queue] Removed: {job_title(job)}")
        self.save_queue()
        self.refresh_queue_view()

    def requeue_job(self, job):
        """Put a finished, failed or stopped job back in line, at the end."""
        if job["status"] == "running":
            return
        self.queue.remove(job)
        job.update(status="waiting", note="", started=None, finished=None, elapsed=None)
        self.queue.append(job)
        self.log(f"[Queue] Re-queued as job {len(self.queue)}: {job_title(job)}")
        self.save_queue()
        self.refresh_queue_view()

    def requeue_failed(self):
        failed = [j for j in self.queue if j["status"] in ("failed", "stopped")]
        if not failed:
            messagebox.showinfo("Queue", "No failed or stopped jobs to re-queue.")
            return
        for job in failed:
            self.requeue_job(job)

    def clear_finished(self):
        keep = [j for j in self.queue if j["status"] in ("waiting", "running")]
        dropped = len(self.queue) - len(keep)
        self.queue = keep
        if dropped:
            self.log(f"[Queue] Cleared {dropped} finished job(s) from the list.")
        self.save_queue()
        self.refresh_queue_view()

    def edit_job(self, job):
        """Send a waiting job back to its tab, so it can be changed and re-added."""
        if job["status"] != "waiting":
            return
        kind = job["kind"]
        self.queue.remove(job)
        for name, attr in JOB_FIELD_VARS[kind].items():
            if name in job["fields"]:
                getattr(self, attr).set(job["fields"][name])
        if kind == "upscale":
            self.up_quality_slider.set(job["fields"].get("quality", 95))
            self.on_up_quality_change(job["fields"].get("quality", 95))
        elif job.get("config"):
            # The job froze its settings; the tab has to show those, or the
            # re-added job would quietly pick up whatever is there now.
            self.config = copy.deepcopy(job["config"])
            for key, w in self.settings_widgets.items():
                val = self.config.get(key)
                try:
                    if w[0] == "combobox":
                        if key == "expression_restorer":
                            val = "on" if val else "off"
                        w[1].set(val if val is not None else "none")
                    elif w[0] == "slider" and val is not None:
                        w[1].set(val)
                        self.update_slider_label(key, float(val))
                    elif w[0] == "entry":
                        w[1].delete(0, "end")
                        if val is not None:
                            w[1].insert(0, str(val))
                except Exception:
                    pass
            self._sync_preset_tooltips(job["fields"].get("preset", ""))
        when = job_not_before(job)
        getattr(self, JOB_SCHEDULE_VARS[kind]).set(
            when.strftime("%Y-%m-%d %H:%M") if when else "")
        for label, start_var, end_var in (
            (getattr(self, "vr_clip_label", None), self.vr_trim_start_var, self.vr_trim_end_var),
            (getattr(self, "nm_clip_label", None), self.nm_trim_start_var, self.nm_trim_end_var),
        ):
            self._show_clip_summary(label, self.parse_optional_float(start_var.get()),
                                    self.parse_optional_float(end_var.get()))
        if kind == "upscale":
            self.refresh_upscale_preview()
        self.log(f"[Queue] {job_title(job)} taken off the queue and put back on the "
                 f"{JOB_TABS[kind]} tab. Press Add to Queue when it is right.")
        self.save_queue()
        self.refresh_queue_view()
        self.tabview.set(JOB_TABS[kind])

    # -- persistence -----------------------------------------------------------

    def save_queue(self):
        data = {
            "saved": datetime.now().strftime(NOT_BEFORE_FMT),
            "stop_on_failure": bool(self.queue_stop_on_failure_var.get()),
            "keep_order": bool(self.queue_keep_order_var.get()),
            "jobs": self.queue,
        }
        try:
            os.makedirs(os.path.dirname(QUEUE_PATH), exist_ok=True)
            with open(QUEUE_PATH, "w") as handle:
                json.dump(data, handle, indent=2)
        except OSError as exc:
            self.log(f"[Queue] Could not save the queue: {exc}")

    def load_queue(self):
        """
        Bring back last session's list. Never auto-starts: a machine that
        rebooted overnight should not begin rendering the moment the app opens.
        """
        try:
            with open(QUEUE_PATH) as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return
        jobs = []
        for job in data.get("jobs") or []:
            if not isinstance(job, dict) or job.get("kind") not in JOB_FIELD_VARS:
                continue
            if job.get("status") == "running":
                job["status"] = "waiting"
                job["note"] = ("Interrupted - the app closed while it ran; it restarts "
                               "from its checkpoints")
            jobs.append(job)
        self.queue = jobs
        self.queue_stop_on_failure_var.set(bool(data.get("stop_on_failure")))
        self.queue_keep_order_var.set(bool(data.get("keep_order")))
        self.refresh_queue_view()
        waiting = [j for j in jobs if j["status"] == "waiting"]
        if jobs:
            self.log(f"[Queue] Restored {len(jobs)} job(s) from "
                     f"{data.get('saved', 'an earlier session')}"
                     + (f"; {len(waiting)} still waiting. Open the Queue tab and press "
                        f"Start Queue to run them." if waiting else "."))
            now = datetime.now()
            overdue = [j for j in waiting if job_not_before(j) and job_not_before(j) < now]
            if overdue:
                self.log(f"[Queue] {len(overdue)} of them had a start time that has "
                         f"already passed - they run as soon as the queue is started.")

    def log_resume_state(self):
        """
        Say at launch what output/temp holds, and whether it is still useful.

        Checked against the fields that were just restored, because half-built
        eyes from a different target are not resumable work - they are 7 GB of
        drive doing nothing, and nothing on the window said so.
        """
        keys = parts = None
        target = self.vr_target_var.get().strip()
        if target and os.path.exists(target):
            try:
                keys, parts = vr_stage_keys(
                    split_sources(self.vr_source_var.get()), target, self.config,
                    self.vr_layout_var.get(),
                    self.parse_optional_float(self.vr_fps_var.get()),
                    self.parse_optional_float(self.vr_trim_start_var.get()),
                    self.parse_optional_float(self.vr_trim_end_var.get()),
                    self.config.get("encoder", "hevc_nvenc"),
                    quality_to_cq(self.config.get("quality", 80)), None,
                )
            except Exception:
                keys = parts = None
        for line in describe_run_state(keys=keys, parts=parts):
            self.log(line)

    def start_session_memory(self):
        """Restore the last launch's fields, then keep them saved as they change."""
        self._session_save_job = None
        self._session_loading = True
        try:
            self.restore_session()
        finally:
            self._session_loading = False
        for name in SESSION_VARS:
            var = getattr(self, name, None)
            if var is not None:
                var.trace_add("write", self._session_changed)

    def _session_changed(self, *_args):
        """Debounced save - an entry box fires this once per keystroke."""
        if self._session_loading:
            return
        if self._session_save_job is not None:
            try:
                self.after_cancel(self._session_save_job)
            except Exception:
                pass
        self._session_save_job = self.after(1000, self.save_session)

    def save_session(self):
        self._session_save_job = None
        data = {
            "saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Whether each output path is still the one this app generated. A
            # path the user typed or browsed to is never moved under them.
            "vr_output_is_auto": self.vr_output_var.get() == self._vr_output_auto,
            "nm_output_is_auto": self.nm_output_var.get() == self._nm_output_auto,
            "vars": {},
        }
        for name in SESSION_VARS:
            var = getattr(self, name, None)
            if var is not None:
                data["vars"][name] = var.get()
        try:
            os.makedirs(os.path.dirname(SESSION_PATH), exist_ok=True)
            with open(SESSION_PATH, "w") as handle:
                json.dump(data, handle, indent=2)
        except OSError as exc:
            self.log(f"[Session] Could not save the current fields: {exc}")

    def restore_session(self):
        """Put back what the window held when it was last used."""
        try:
            with open(SESSION_PATH) as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return
        saved = data.get("vars") or {}
        for name in SESSION_VARS:
            var = getattr(self, name, None)
            if var is None or name not in saved:
                continue
            try:
                var.set(saved[name])
            except Exception:
                # One unreadable field must not cost the user the other
                # twenty-eight; it simply keeps its default.
                pass

        # The generated names carry a launch timestamp, so a restored one has
        # to become the new "generated" name or the layout radio stops moving
        # it and a finished run stops replacing it.
        if data.get("vr_output_is_auto"):
            self._adopt_vr_auto_output(self.vr_output_var.get())
        if data.get("nm_output_is_auto"):
            self._nm_output_auto = self.nm_output_var.get()

        self._sync_preset_tooltips(self.vr_preset_var.get())
        for label, start_var, end_var in (
            (self.vr_clip_label, self.vr_trim_start_var, self.vr_trim_end_var),
            (self.nm_clip_label, self.nm_trim_start_var, self.nm_trim_end_var),
        ):
            self._show_clip_summary(label,
                                    self.parse_optional_float(start_var.get()),
                                    self.parse_optional_float(end_var.get()))
        try:
            self.refresh_upscale_preview()
        except Exception:
            pass
        self.report_session(data.get("saved", "an earlier session"))

    def report_session(self, saved_at):
        """
        Say what came back, and say it loudly when an input has since moved.

        A source or target that is gone is not a cosmetic problem: those paths
        are part of the resume key, so a re-browse to the same file in a new
        location restarts the whole job from the split.
        """
        self.log(f"[Session] Restored the clip, source and timings from {saved_at}.")
        if self.vr_target_var.get():
            self.log(f"  Target: {self.vr_target_var.get()}")
        if self.vr_source_var.get():
            self.log(f"  Source: {self.vr_source_var.get()}")
        start = self.vr_trim_start_var.get().strip()
        end = self.vr_trim_end_var.get().strip()
        if start or end:
            self.log(f"  Clip: {start or 'start'} to {end or 'end'}")
        missing = [path for path in
                   ([self.vr_target_var.get().strip()]
                    + split_sources(self.vr_source_var.get()))
                   if path and not os.path.exists(path)]
        for path in missing:
            self.log(f"[Session] WARNING - this file is no longer where it was: {path}")

    def _adopt_vr_auto_output(self, path):
        """Treat `path` as the name this app generated, layout suffix and all."""
        self._vr_output_auto = path
        stem = os.path.splitext(path)[0]
        suffix = vr_metadata.suffix_for(self.vr_layout_var.get())
        if stem.upper().endswith(suffix.upper()):
            stem = stem[:-len(suffix)]
        self._vr_output_stem = stem

    def _retire_auto_output(self):
        """
        A finished render owns its name; the next job gets a fresh one.

        Only for names this app generated - a path the user chose stays put,
        and a run that failed keeps its name so the retry writes to the file
        the resume is rebuilding.
        """
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.vr_output_var.get() == self._vr_output_auto:
            suffix = vr_metadata.suffix_for(self.vr_layout_var.get())
            self._vr_output_stem = self._fresh_auto_stem("vr_deepfake_", stamp, suffix)
            self._vr_output_auto = self._vr_output_stem + suffix + ".mp4"
            self.vr_output_var.set(self._vr_output_auto)
        if self.nm_output_var.get() == self._nm_output_auto:
            self._nm_output_auto = self._fresh_auto_stem("deepfake_", stamp, "") + ".mp4"
            self.nm_output_var.set(self._nm_output_auto)
        self.save_session()

    def _fresh_auto_stem(self, prefix, stamp, suffix):
        """
        A generated name nothing else is using.

        The stamp only has seconds in it, and two presses of Add to Queue can
        land in the same second - which used to hand the second job the first
        job's output path and get it refused as a duplicate.
        """
        taken = {os.path.normcase(os.path.abspath(j["fields"]["output"]))
                 for j in self.queue if j.get("fields", {}).get("output")}
        taken.add(os.path.normcase(os.path.abspath(self._vr_output_auto)))
        taken.add(os.path.normcase(os.path.abspath(self._nm_output_auto)))
        base = os.path.join(PROJECT_DIR, "output", "final", prefix + stamp)
        stem, n = base, 1
        while True:
            path = stem + suffix + ".mp4"
            if os.path.normcase(os.path.abspath(path)) not in taken and not os.path.exists(path):
                return stem
            n += 1
            stem = f"{base}_{n}"

    def on_close(self):
        if self.queue_active and not self.is_running:
            waiting = [j for j in self.queue if j["status"] == "waiting"]
            if not messagebox.askyesno(
                "Quit",
                f"The queue is waiting to start {len(waiting)} job(s).\n\n"
                "Closing this window stops the wait - a queue needs the app open. "
                "The jobs stay in the list for next time.\n\nQuit anyway?"
            ):
                return
            self.pause_queue("Paused - the app was closed.")
        if self.is_running:
            paused = " It is paused; quitting still kills it." if (
                self.guard and self.guard.paused) else ""
            queued = ("\nThe queue stops too; its waiting jobs are kept for next time."
                      if self.queue_active else "")
            if not messagebox.askyesno(
                "Quit",
                "A conversion is still running.\n\n"
                "Quitting will stop it and kill every FaceFusion/ffmpeg process."
                + paused + queued + "\n\nQuit anyway?"
            ):
                return
            self.queue_active = False
            if self._active_job is not None:
                self._active_job["status"] = "stopped"
                self._active_job["note"] = "The app was closed while it ran"
        self.save_queue()
        self.save_session()
        self._vram_stop.set()
        if self.guard:
            self.guard.cancel("Application closed")
            # No wait: the window is closing and the process is about to exit,
            # so the OS reclaims everything wholesale - blocking the quit on a
            # clean teardown would only make closing feel broken.
            self.guard.close(wait_timeout=0)
        self.destroy()

    # A clip is offered alongside photos because FaceFusion averages the
    # ArcFace embeddings of every source image it is given, and a few seconds
    # of video is simply a better way to fill that list than hunting for
    # separate photos: the pipeline picks the stills, and picks them for spread
    # rather than for whichever ones happened to be to hand. Mixing the two is
    # allowed, and is often the best input - a sharp portrait for detail plus a
    # clip for the range of poses.
    SOURCE_FILETYPES = [
        ("Photos or video", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff "
                            "*.mp4 *.mkv *.mov *.avi *.webm *.m4v"),
        ("Photos", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
        ("Video", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v"),
    ]
    SOURCE_DIALOG_TITLE = ("Select Source Face - photos (1 to 5), or one short "
                           "video clip of the face")

    def browse_vr_source(self):
        paths = filedialog.askopenfilenames(
            title=self.SOURCE_DIALOG_TITLE, filetypes=self.SOURCE_FILETYPES
        )
        if paths:
            self.vr_source_var.set("; ".join(paths))

    def browse_vr_target(self):
        path = filedialog.askopenfilename(
            title="Select Target VR Video",
            filetypes=[("Videos", "*.mp4 *.mkv *.avi *.mov *.webm")]
        )
        if path:
            self.vr_target_var.set(path)

    def browse_vr_output(self):
        path = filedialog.asksaveasfilename(
            title="Save VR Output As",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("MKV Video", "*.mkv")]
        )
        if path:
            self.vr_output_var.set(path)

    def browse_nm_source(self):
        paths = filedialog.askopenfilenames(
            title=self.SOURCE_DIALOG_TITLE, filetypes=self.SOURCE_FILETYPES
        )
        if paths:
            self.nm_source_var.set("; ".join(paths))

    def browse_nm_target(self):
        path = filedialog.askopenfilename(
            title="Select Target Video",
            filetypes=[("Videos", "*.mp4 *.mkv *.avi *.mov *.webm")]
        )
        if path:
            self.nm_target_var.set(path)

    def browse_nm_output(self):
        path = filedialog.asksaveasfilename(
            title="Save Output As",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("MKV Video", "*.mkv")]
        )
        if path:
            self.nm_output_var.set(path)

    def check_source(self, source):
        """
        Validate the Source Face box, and describe what was given.

        Returns the line to log, or None after showing the error - so a caller
        can just check for None and return.
        """
        entries = split_sources(source)
        missing = [p for p in entries if not os.path.exists(p)]
        if missing:
            messagebox.showerror("Error", f"Source face not found: {missing[0]}")
            return None

        unknown = [p for p in entries if not (is_image_path(p) or is_video_path(p))]
        if unknown:
            messagebox.showerror(
                "Error",
                f"{os.path.basename(unknown[0])} is neither an image nor a video.\n\n"
                f"The Source Face can be one or more photos, or a short video clip "
                f"of the face, or a mixture of the two."
            )
            return None

        photos = [p for p in entries if is_image_path(p)]
        clips = [p for p in entries if is_video_path(p)]
        if not clips:
            return f"Using {len(photos)} reference photo(s)"

        count = int(self.config.get("source_video_faces") or 6)
        parts = [f"{len(clips)} source clip(s), up to {count} still(s) extracted from each"]
        if photos:
            parts.append(f"{len(photos)} reference photo(s)")
        return "Using " + " + ".join(parts)

    def parse_optional_float(self, val):
        """
        A number from a box that is allowed to be empty.

        Timecodes go through the same parser as plain seconds, because the clip
        picker writes its choices back as m:ss.s - if this only understood
        floats, the value it wrote would read back as None and the clip would
        silently become the whole video.
        """
        return clip_selector.parse_timecode(val)

    def select_vr_clip(self):
        self._select_clip(self.vr_target_var, self.vr_trim_start_var,
                          self.vr_trim_end_var, self.vr_clip_label, "VR")

    def select_nm_clip(self):
        self._select_clip(self.nm_target_var, self.nm_trim_start_var,
                          self.nm_trim_end_var, self.nm_clip_label, "Normal")

    def select_up_clip(self):
        self._select_clip(self.up_input_var, self.up_trim_start_var,
                          self.up_trim_end_var, None, "Upscale")
        self.refresh_upscale_preview()

    def clear_vr_clip(self):
        self._clear_clip(self.vr_trim_start_var, self.vr_trim_end_var, self.vr_clip_label)

    def clear_nm_clip(self):
        self._clear_clip(self.nm_trim_start_var, self.nm_trim_end_var, self.nm_clip_label)

    def _clear_clip(self, start_var, end_var, clip_label):
        start_var.set("")
        end_var.set("")
        self._show_clip_summary(clip_label, None, None)

    def _show_clip_summary(self, clip_label, start, end):
        if clip_label is None:
            return
        if start is None and end is None:
            text = ""
        elif start is None:
            text = f"up to {end:.1f}s"
        elif end is None:
            text = f"from {start:.1f}s"
        else:
            text = f"{end - start:.1f}s selected"
        # How the range will be cut up, so the answer to "is this going to be
        # chunked?" is on screen before Run is pressed rather than in the log
        # after it.
        plan = self._chunk_plan_for(self._clip_target_for(clip_label), start, end)
        if plan is not None:
            if len(plan) > 1:
                text = (f"{text} - {len(plan)} chunks of "
                        f"~{int(round(self.config.get('chunk_seconds') or 0))}s").lstrip(" -")
            elif self.config.get("chunk_seconds"):
                text = f"{text} - 1 chunk".lstrip(" -")
        clip_label.configure(text=text)

    def _clip_target_for(self, clip_label):
        if clip_label is getattr(self, "vr_clip_label", None):
            return self.vr_target_var.get().strip()
        if clip_label is getattr(self, "nm_clip_label", None):
            return self.nm_target_var.get().strip()
        return ""

    def _chunk_plan_for(self, target, start, end, config=None):
        """The (start, end) chunks a run would use now, or None if unknowable."""
        chunk_seconds = (config or self.config).get("chunk_seconds") or 0
        if not target or not os.path.exists(target) or is_image_path(target):
            return None
        cache = getattr(self, "_duration_cache", None)
        if cache is None:
            cache = self._duration_cache = {}
        try:
            stamp = (target, os.path.getsize(target), int(os.path.getmtime(target)))
            if stamp not in cache:
                cache.clear()
                cache[stamp] = probe_video(target).get("duration") or 0.0
            duration = cache[stamp]
        except Exception:
            return None
        if not duration:
            return None
        return chunk_plan(duration, start, end, chunk_seconds)

    def _log_chunk_plan(self, target, start, end, output, config=None):
        """Say, before anything starts, exactly how the range will be cut up."""
        plan = self._chunk_plan_for(target, start, end, config=config)
        if plan is None:
            return
        chunk_seconds = (config or self.config).get("chunk_seconds") or 0
        if len(plan) < 2:
            if chunk_seconds:
                self.log(f"[Chunks] The selected {plan[0][1] - plan[0][0]:.0f}s fits in one "
                         f"chunk (chunk length is {chunk_seconds:.0f}s), so it runs as a "
                         f"single job.")
            else:
                self.log("[Chunks] Chunking is off (chunk length 0); the whole range runs "
                         "as a single job.")
            return
        self.log(f"[Chunks] The range will be processed in {len(plan)} chunks of "
                 f"~{chunk_seconds:.0f}s, one after another:")
        for i, (a, b) in enumerate(plan):
            self.log(f"  {i + 1:>2}. {clip_selector.format_timecode(a)} to "
                     f"{clip_selector.format_timecode(b)}  ({b - a:.0f}s)")
        self.log(f"[Chunks] Each chunk is saved as a playable file in "
                 f"{chunk_dir_for(output)} the moment it finishes. When all {len(plan)} "
                 f"are done they are joined into {os.path.basename(output)} and the "
                 f"chunk files are deleted. If the run stops early, the finished chunks "
                 f"stay there and the next run continues from the first unfinished one.")

    def _select_clip(self, target_var, start_var, end_var, clip_label, label):
        """
        Open the in/out point picker for whichever video box this tab uses.

        The picker owns the whole choice, including "actually, all of it" - so
        it can also clear a range that is already set, and the two entry boxes
        stay authoritative for what the run will use.
        """
        target = target_var.get().strip()
        if not target:
            messagebox.showinfo(
                "Pick a video first",
                "Choose the video first, then select the portion of it to use.")
            return
        if not os.path.exists(target):
            messagebox.showerror("Error", f"Video not found: {target}")
            return
        if is_image_path(target):
            messagebox.showinfo(
                "Still image",
                "That target is a still image, so there is no portion to pick.")
            return

        chosen = clip_selector.select_clip(
            self, target,
            self.parse_optional_float(start_var.get()),
            self.parse_optional_float(end_var.get()),
        )
        if chosen is None:
            return

        start, end = chosen
        start_var.set("" if start is None else clip_selector.format_timecode(start, 2))
        end_var.set("" if end is None else clip_selector.format_timecode(end, 2))
        self._show_clip_summary(clip_label, start, end)
        if start is None:
            self.log(f"[{label}] Target portion: the whole video.")
        else:
            self.log(f"[{label}] Target portion: {clip_selector.format_timecode(start)} "
                     f"to {clip_selector.format_timecode(end)} ({end - start:.1f}s).")

    def preflight_or_confirm(self, target, is_vr, trim_start, trim_end,
                             unattended=False, config=None):
        """
        Run the guard-rail preflight on the UI thread so problems can be shown
        in a dialog before anything heavy is spawned.

        Returns the `prepared` tuple to hand to the pipeline, or None to abort.
        None on its own does not say why, and _launch_job has to know whether
        the refusal was transient before it decides between failing a queued
        job and holding it, so the exception is left in
        self._last_preflight_error for the caller to read. Cleared on entry, so
        a stale one can never be mistaken for this call's.
        """
        config = config or self.config
        self._last_preflight_error = None
        try:
            return prepare_run(
                target, config, is_vr=is_vr,
                trim_start=trim_start, trim_end=trim_end,
                log_cb=self.log, force=False,
            )
        except PipelineError as e:
            self._last_preflight_error = e
            self.log(str(e))
            if unattended:
                # Nobody is at the keyboard - that is the whole point of a
                # scheduled start. A dialog here would simply wait until
                # morning with the run never begun and no reason on screen.
                #
                # A transient refusal is not reported as a dead end here: the
                # caller is about to hold the job and try again, and saying
                # "NOT started" of a job that starts twenty minutes later reads
                # as a failure in the log for the rest of the night.
                if not getattr(e, "transient", False):
                    self.log("[Queue] The job was NOT started: the guard rails refused "
                             "it, and an unattended run is never forced past them. The "
                             "reason is directly above.")
                    self.update_status("Not started", "#ef4444")
                return None
            proceed = messagebox.askyesno(
                "Guard rails: unsafe run",
                f"{e}\n\n"
                "Starting anyway risks locking up Windows.\n"
                "The watchdog will still stop the run if things get critical.\n\n"
                "Start anyway?",
                icon="warning",
            )
            if not proceed:
                self.log("Run cancelled by guard rails.")
                return None
            return prepare_run(
                target, config, is_vr=is_vr,
                trim_start=trim_start, trim_end=trim_end,
                log_cb=self.log, force=True,
            )

    def show_estimate(self, prepared):
        """
        Put the up-front time estimate where it cannot be missed.

        prepare_run has already written the stage-by-stage breakdown into the
        log, but the log is a scrolling wall during a run and this is the one
        number someone wants before deciding to start a job that may still be
        going tomorrow. The progress label carries it until the first stage
        replaces it with a live figure.
        """
        try:
            plan = (prepared[2] or {}).get("runtime") or {}
            total = plan.get("total_seconds") or 0
        except Exception:
            return
        if total:
            chunks = (prepared[2] or {}).get("chunks") or []
            self.update_progress(
                0, f"Estimated {guardrails.fmt_duration(total)}"
                   + (f" in {len(chunks)} chunks" if len(chunks) > 1 else "")
                   + f" - finishing around {guardrails.fmt_clock(total)}")

    def run_vr(self):
        self._run_now_or_queue("vr")

    def run_normal(self):
        self._run_now_or_queue("normal")

    def update_status(self, text, color):
        self.status_label.configure(text=text, text_color=color)


def main():
    app = App()
    try:
        app.mainloop()
    finally:
        # Last line of defence: never leave children running behind the UI.
        if getattr(app, "guard", None):
            app.guard.close(wait_timeout=0)


if __name__ == "__main__":
    main()
