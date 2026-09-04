import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
import os
import sys
import gc
import json
import time
import threading
from datetime import datetime
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
import upscale_vr
import clip_selector
from guardrails import ResourceGuard
from engine import (
    run_vr_pipeline, run_normal_pipeline, load_config, prepare_run,
    split_sources, is_video_path, is_image_path,
    PROJECT_DIR, DEFAULT_CONFIG, PipelineError
)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

APP_WIDTH = 1200
APP_HEIGHT = 850

# The log pane and its backing list both grew for the lifetime of the session,
# and a Tk text widget holds every line it has ever been given. Nothing here is
# large per line, but the app is meant to sit open across many runs.
MAX_LOG_LINES = 2000
LOG_TRIM_TO = 1500

# Slider-backed settings that must reach the FaceFusion CLI as integers.
INT_SETTINGS = {"enhancer_blend", "frame_enhancer_blend", "quality", "threads", "upscale_scale",
                "target_frame_amount", "reference_frame_number", "expression_restorer_factor",
                "source_video_faces", "source_video_max_samples", "source_video_min_face_px"}

# Encoders offered on the Upscale tab. NVENC first because it is 10x the speed
# of software at these sizes; libx265/libx264 are there because NVENC refuses
# anything over 8192px, and a software encoder is the only way past that.
UPSCALE_ENCODERS = ["hevc_nvenc", "av1_nvenc", "h264_nvenc", "libx265", "libx264"]

# Entry-backed settings that must reach the worker as floats. Without this an
# entry falls through to the "store the string" branch, and "2.0" in the config
# file is not the same thing as 2.0 to the code that reads it back.
FLOAT_SETTINGS = {"source_video_sample_fps"}

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
QUALITY_PRESETS = {
    'Fast (~17 min/1min video)': {
        'tooltip': 'Face swap only, no enhancement. Measured 3.5 frame/s.',
        'swapper_model': 'hyperswap_1a_256',
        'pixel_boost': '256x256',
        'enhancer_model': 'none',
        'frame_enhancer': 'none',
    },
    'Good (~31 min/1min video)': {
        'tooltip': 'Face swap + 512px enhancement. Best speed/quality balance; '
                   'measured 1.95 frame/s.',
        'swapper_model': 'hyperswap_1a_256',
        'pixel_boost': '256x256',
        'enhancer_model': 'gpen_bfr_512',
        'frame_enhancer': 'none',
    },
    'Better (~38 min/1min video)': {
        'tooltip': 'Adds 512px pixel boost for more detail per face. '
                   'Measured 1.6 frame/s.',
        'swapper_model': 'hyperswap_1a_256',
        'pixel_boost': '512x512',
        'enhancer_model': 'gpen_bfr_512',
        'frame_enhancer': 'none',
    },
    'Best (~46 min/1min video)': {
        'tooltip': '512px pixel boost + 1024px enhancement. gpen_bfr_2048 is '
                   'deliberately not used here: it costs ~2x again for detail a '
                   'face a few hundred pixels wide cannot carry.',
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
DEFAULT_PRESET = 'Good (~31 min/1min video)'
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

        self.is_running = False
        self.current_thread = None
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

        # Nothing must outlive the window. Closing used to leave FaceFusion and
        # ffmpeg running, still eating the machine with no UI left to stop them.
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.start_vram_sampler()
        self.log_machine_limits()

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
        self.tabview.add("Settings")
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.build_vr_tab()
        self.build_normal_tab()
        self.build_upscale_tab()
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

        # Always available, never disabled: if a run is misbehaving this is the
        # way out, and it kills the whole process tree rather than just asking.
        self.stop_btn = ctk.CTkButton(
            header_row, text="STOP", width=90, height=26,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#b91c1c", hover_color="#dc2626",
            command=self.stop_run, state="disabled"
        )
        self.stop_btn.grid(row=0, column=2, sticky="e")

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
        self.vr_output_var = ctk.StringVar(
            value=os.path.join(PROJECT_DIR, "output", "final",
                              f"vr_deepfake_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        )
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

        row += 2
        self.vr_run_btn = ctk.CTkButton(
            tab, text="Run VR Pipeline", height=45, font=ctk.CTkFont(size=14, weight="bold"),
            command=self.run_vr, corner_radius=10
        )
        self.vr_run_btn.grid(row=row, column=1, padx=pad_x, pady=(10, 5))

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
        self.nm_output_var = ctk.StringVar(
            value=os.path.join(PROJECT_DIR, "output", "final",
                              f"deepfake_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        )
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
        self.up_run_btn = ctk.CTkButton(
            tab, text="Run Upscale", height=45, font=ctk.CTkFont(size=14, weight="bold"),
            command=self.run_upscale_job, corner_radius=10)
        self.up_run_btn.grid(row=row, column=1, padx=pad_x, pady=(6, 5))

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
        headset = self.up_headset_var.get()
        return SimpleNamespace(
            input=self.up_input_var.get().strip(),
            output=self.up_output_var.get().strip(),
            target=self.up_target_var.get(),
            scale=None,
            layout=self.up_layout_var.get(),
            method=self.up_method_var.get(),
            content=None,
            headset=None if headset == "none" else headset,
            # The tab owns these rather than reading the Settings tab: that one
            # configures the deepfake pipeline, and an upscale is a different
            # job with different trade-offs - a software encoder here is a
            # deliberate choice to exceed 8192px, not a fallback.
            encoder=self.up_encoder_var.get(),
            quality=int(round(self.up_quality_slider.get())),
            allow_software=self.up_allow_sw_var.get(),
            split_eyes=self.up_split_eyes_var.get(),
            trim_start=self.parse_optional_float(self.up_trim_start_var.get()),
            trim_end=self.parse_optional_float(self.up_trim_end_var.get()),
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
        args = self.upscale_args()
        if not args.input or not args.output:
            messagebox.showerror("Error", "Please choose an input video and an output path.")
            return
        if not os.path.exists(args.input):
            messagebox.showerror("Error", "Input not found: " + args.input)
            return

        self.log("=" * 50)
        self.log("UPSCALE STARTED")
        self.set_blocked(True)
        self.progress_bar.set(0)
        self.guard = ResourceGuard(
            os.path.join(PROJECT_DIR, "output", "temp"), guardrails.load_guard_config(),
            log_cb=lambda m: self.after(0, self.log, m),
        ).start()

        self.current_thread = threading.Thread(
            target=self._run_upscale_thread, args=(args,), daemon=True)
        self.current_thread.start()

    def _run_upscale_thread(self, args):
        start_time = time.time()
        try:
            code = upscale_vr.run_upscale(
                args,
                log_fn=lambda m: self.after(0, self.log, m),
                progress_cb=lambda p, t: self.after(0, self.update_progress, p, t),
                guard=self.guard,
            )
            elapsed = time.time() - start_time
            if code == 0:
                self.after(0, self.update_progress, 100, "Upscale complete")
                self.after(0, self.log, "Upscale complete! (%.0fs)" % elapsed)
                self.after(0, self.update_status, "Done", "#4ade80")
                self.after(0, lambda: messagebox.showinfo(
                    "Success", "Upscale complete!\n\nOutput: %s\nTime: %.0fs"
                               % (args.output, elapsed)))
            else:
                self.after(0, self.update_status, "Failed", "#f87171")
                self.after(0, lambda: messagebox.showerror(
                    "Upscale failed", "See the log for what stopped it."))
        except Exception as e:
            self.after(0, self.log, "Upscale failed: %s" % e)
            self.after(0, self.update_status, "Failed", "#f87171")
            self.after(0, lambda err=e: messagebox.showerror("Upscale failed", str(err)))
        finally:
            self.after(0, self.finish_run)

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

                val = self.config.get(key)

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

        self.log("Settings saved.")
        messagebox.showinfo("Settings", "Settings saved successfully.")

    def apply_settings(self):
        """
        Read every settings widget back into self.config.

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
        self.is_running = blocked
        state = "disabled" if blocked else "normal"
        self.vr_run_btn.configure(state=state)
        self.nm_run_btn.configure(state=state)
        self.up_run_btn.configure(state=state)
        self.vr_run_btn.configure(text="Running..." if blocked else "Run VR Pipeline")
        self.nm_run_btn.configure(text="Running..." if blocked else "Run Normal Pipeline")
        self.up_run_btn.configure(text="Running..." if blocked else "Run Upscale")
        self.stop_btn.configure(state="normal" if blocked else "disabled")

    def log_machine_limits(self):
        """Show what the guard rails will allow, so limits are never a surprise."""
        guard_config = guardrails.load_guard_config()
        total_ram, avail_ram, _ = guardrails.memory_status()
        vram_total, _ = guardrails.gpu_memory_mb()
        temp_dir = os.path.join(PROJECT_DIR, "output", "temp")
        free_disk = guardrails.free_disk_bytes(temp_dir)

        self.log("Guard rails active. Limits for this machine:")
        self.log(f"  RAM {guardrails.fmt_gb(total_ram)} total / {guardrails.fmt_gb(avail_ram)} free "
                 f"- pipeline capped at {guard_config['job_memory_cap_gb']} GB")
        self.log(f"  CPU {os.cpu_count()} logical cores - pipeline capped at "
                 f"{guard_config['cpu_percent_cap']}%, below-normal priority")
        self.log(f"  VRAM {vram_total} MB")
        self.log(f"  Temp disk ({os.path.splitdrive(os.path.abspath(temp_dir))[0]}) "
                 f"{guardrails.fmt_gb(free_disk)} free - run stops automatically below "
                 f"{guard_config['min_free_disk_gb']} GB")
        if free_disk and free_disk < 100 * guardrails.GB:
            self.log("  NOTE: temp space is limited. Trim long clips or point "
                     "output/temp at a larger drive.")
        self.log("Tune any of this in scripts/guardrails.json.")
        self.refresh_resource_label()

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
            _, avail_ram, _ = guardrails.memory_status()
            free_disk = guardrails.free_disk_bytes(os.path.join(PROJECT_DIR, "output", "temp"))
            self.resource_label.configure(
                text=f"RAM free {guardrails.fmt_gb(avail_ram)}  |  "
                     f"Temp disk free {guardrails.fmt_gb(free_disk)}  |  "
                     f"VRAM used {self._vram_used_mb} MB"
            )
        except Exception:
            pass
        self.after(3000, self.refresh_resource_label)

    def stop_run(self):
        """Kill the whole pipeline process tree immediately."""
        if self.guard:
            self.log("Stop requested - terminating all pipeline processes...")
            self.guard.cancel("Stopped by user")
            self.status_label.configure(text="Stopping...", text_color="#f59e0b")
        else:
            self.set_blocked(False)

    def on_close(self):
        if self.is_running:
            if not messagebox.askyesno(
                "Quit",
                "A conversion is still running.\n\n"
                "Quitting will stop it and kill every FaceFusion/ffmpeg process.\n\nQuit anyway?"
            ):
                return
        self._vram_stop.set()
        if self.guard:
            self.guard.cancel("Application closed")
            self.guard.close()
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
            clip_label.configure(text="")
        else:
            clip_label.configure(text=f"{end - start:.1f}s selected")

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

    def preflight_or_confirm(self, target, is_vr, trim_start, trim_end):
        """
        Run the guard-rail preflight on the UI thread so problems can be shown
        in a dialog before anything heavy is spawned.

        Returns the `prepared` tuple to hand to the pipeline, or None to abort.
        """
        try:
            return prepare_run(
                target, self.config, is_vr=is_vr,
                trim_start=trim_start, trim_end=trim_end,
                log_cb=self.log, force=False,
            )
        except PipelineError as e:
            self.log(str(e))
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
                target, self.config, is_vr=is_vr,
                trim_start=trim_start, trim_end=trim_end,
                log_cb=self.log, force=True,
            )

    def run_vr(self):
        self._apply_preset(self.vr_preset_var.get())
        self.apply_settings()
        self.log(f"Quality preset: {self.vr_preset_var.get()}")

        source = self.vr_source_var.get().strip()
        target = self.vr_target_var.get().strip()
        output = self.vr_output_var.get().strip()

        if not source or not target or not output:
            messagebox.showerror("Error", "Please fill in source face, target video, and output path.")
            return

        source_summary = self.check_source(source)
        if source_summary is None:
            return
        if not os.path.exists(target):
            messagebox.showerror("Error", f"Target video not found: {target}")
            return

        self.log("=" * 50)
        self.log("VR PIPELINE STARTED")
        self.log(source_summary)
        self.log(f"Source: {source}")
        self.log(f"Target: {target}")
        self.log(f"Layout: {self.vr_layout_var.get().upper()}")
        self.log(f"Output: {output}")

        trim_start = self.parse_optional_float(self.vr_trim_start_var.get())
        trim_end = self.parse_optional_float(self.vr_trim_end_var.get())

        # The run log is the only record of what a two-hour job actually
        # processed, so it has to say which portion of the target that was.
        if trim_start is not None or trim_end is not None:
            self.log(f"Clip: {clip_selector.format_timecode(trim_start) or 'start'} to "
                     f"{clip_selector.format_timecode(trim_end) or 'end'}")

        prepared = self.preflight_or_confirm(target, True, trim_start, trim_end)
        if prepared is None:
            return

        self.set_blocked(True)
        self.progress_bar.set(0)
        self.guard = ResourceGuard(
            os.path.join(PROJECT_DIR, "output", "temp"), prepared[1],
            log_cb=lambda m: self.after(0, self.log, m),
        ).start()

        self.current_thread = threading.Thread(
            target=self._run_vr_thread,
            args=(source, target, output, prepared, trim_start, trim_end),
            daemon=True
        )
        self.current_thread.start()

    def _run_vr_thread(self, source, target, output, prepared, trim_start, trim_end):
        start_time = time.time()
        try:
            run_vr_pipeline(
                source_face=source,
                target_video=target,
                output_path=output,
                config=self.config,
                layout=self.vr_layout_var.get(),
                fps=self.parse_optional_float(self.vr_fps_var.get()),
                trim_start=trim_start,
                trim_end=trim_end,
                keep_temp=self.vr_keep_temp_var.get(),
                progress_cb=lambda p, t: self.after(0, self.update_progress, p, t),
                log_cb=lambda m: self.after(0, self.log, m),
                guard=self.guard,
                prepared=prepared,
                refresh_source_faces=self.vr_refresh_faces_var.get(),
            )

            elapsed = time.time() - start_time
            self.after(0, self.log, f"VR Pipeline complete! ({elapsed:.0f}s)")
            self.after(0, self.log, f"Output: {output}")
            self.after(0, self.update_status, "Done", "#4ade80")

            self.after(0, lambda: messagebox.showinfo(
                "Success",
                f"VR deepfake complete!\n\nOutput: {output}\nTime: {elapsed:.0f}s ({elapsed/60:.1f} min)"
            ))

        except PipelineError as e:
            self.after(0, self.log, f"Pipeline failed: {e}")
            self.after(0, self.update_status, "Failed", "#ef4444")
            self.after(0, lambda: messagebox.showerror("Error", str(e)))

        except Exception as e:
            self.after(0, self.log, f"Unexpected error: {e}")
            self.after(0, self.update_status, "Error", "#ef4444")
            self.after(0, lambda: messagebox.showerror("Error", f"Unexpected error: {e}"))

        finally:
            self.after(0, self.finish_run)

    def finish_run(self):
        """Tear the guard down on the UI thread once a run ends, however it ended."""
        if self.guard:
            # close() closes the Job Object, and KILL_ON_JOB_CLOSE reaps every
            # child with it - which is what actually returns the run's memory,
            # host and device alike, since every model lives in a child process
            # and never in this one.
            self.guard.close()
            self.guard = None
        # Re-extract is a one-shot: leaving it ticked would silently re-run the
        # source analysis on every subsequent run against the same clip, which
        # is exactly the cost the cache exists to avoid.
        self.vr_refresh_faces_var.set(False)
        self.nm_refresh_faces_var.set(False)
        # The pipeline builds a fair number of short-lived dicts and lists per
        # run; collecting here keeps the idle footprint flat between runs
        # instead of drifting up until a generation happens to trigger.
        gc.collect()
        self.set_blocked(False)

    def run_normal(self):
        self._apply_preset(self.nm_preset_var.get())
        self.apply_settings()
        self.log(f"Quality preset: {self.nm_preset_var.get()}")

        source = self.nm_source_var.get().strip()
        target = self.nm_target_var.get().strip()
        output = self.nm_output_var.get().strip()

        if not source or not target or not output:
            messagebox.showerror("Error", "Please fill in source face, target video, and output path.")
            return

        source_summary = self.check_source(source)
        if source_summary is None:
            return
        if not os.path.exists(target):
            messagebox.showerror("Error", f"Target video not found: {target}")
            return

        self.log("=" * 50)
        self.log("NORMAL PIPELINE STARTED")
        self.log(source_summary)
        self.log(f"Source: {source}")
        self.log(f"Target: {target}")
        self.log(f"Output: {output}")

        trim_start = self.parse_optional_float(self.nm_trim_start_var.get())
        trim_end = self.parse_optional_float(self.nm_trim_end_var.get())

        # The run log is the only record of what a two-hour job actually
        # processed, so it has to say which portion of the target that was.
        if trim_start is not None or trim_end is not None:
            self.log(f"Clip: {clip_selector.format_timecode(trim_start) or 'start'} to "
                     f"{clip_selector.format_timecode(trim_end) or 'end'}")

        prepared = self.preflight_or_confirm(target, False, trim_start, trim_end)
        if prepared is None:
            return

        self.set_blocked(True)
        self.progress_bar.set(0)
        self.guard = ResourceGuard(
            os.path.join(PROJECT_DIR, "output", "temp"), prepared[1],
            log_cb=lambda m: self.after(0, self.log, m),
        ).start()

        self.current_thread = threading.Thread(
            target=self._run_normal_thread,
            args=(source, target, output, prepared, trim_start, trim_end),
            daemon=True
        )
        self.current_thread.start()

    def _run_normal_thread(self, source, target, output, prepared, trim_start, trim_end):
        start_time = time.time()
        try:
            run_normal_pipeline(
                source_face=source,
                target_video=target,
                output_path=output,
                config=self.config,
                fps=self.parse_optional_float(self.nm_fps_var.get()),
                trim_start=trim_start,
                trim_end=trim_end,
                keep_temp=self.nm_keep_temp_var.get(),
                progress_cb=lambda p, t: self.after(0, self.update_progress, p, t),
                log_cb=lambda m: self.after(0, self.log, m),
                guard=self.guard,
                prepared=prepared,
                refresh_source_faces=self.nm_refresh_faces_var.get(),
            )

            elapsed = time.time() - start_time
            self.after(0, self.log, f"Normal Pipeline complete! ({elapsed:.0f}s)")
            self.after(0, self.log, f"Output: {output}")
            self.after(0, self.update_status, "Done", "#4ade80")

            self.after(0, lambda: messagebox.showinfo(
                "Success",
                f"Deepfake complete!\n\nOutput: {output}\nTime: {elapsed:.0f}s ({elapsed/60:.1f} min)"
            ))

        except PipelineError as e:
            self.after(0, self.log, f"Pipeline failed: {e}")
            self.after(0, self.update_status, "Failed", "#ef4444")
            self.after(0, lambda: messagebox.showerror("Error", str(e)))

        except Exception as e:
            self.after(0, self.log, f"Unexpected error: {e}")
            self.after(0, self.update_status, "Error", "#ef4444")
            self.after(0, lambda: messagebox.showerror("Error", f"Unexpected error: {e}"))

        finally:
            self.after(0, self.finish_run)

    def update_status(self, text, color):
        self.status_label.configure(text=text, text_color=color)


def main():
    app = App()
    try:
        app.mainloop()
    finally:
        # Last line of defence: never leave children running behind the UI.
        if getattr(app, "guard", None):
            app.guard.close()


if __name__ == "__main__":
    main()
