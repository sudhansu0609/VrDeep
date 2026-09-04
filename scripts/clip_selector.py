"""
Pick a portion of a video to act as the target, instead of the whole file.

Trim Start / Trim End have always been on the VR and Normal tabs, but typing
seconds into them is blind: nothing on screen says what sits at 412.5s, so
choosing the stretch of a long clip that actually holds a usable face means
guessing, running, and looking at the result an hour later. This dialog puts
the frame under the playhead on screen and writes the two numbers back.

Frames come from one open cv2 capture rather than an ffmpeg subprocess per
scrub - seeking a capture that is already open is fast enough to feel live,
where spawning a process for every pixel the slider moves is not. All seeking
happens on a single worker thread, because a VideoCapture is not safe to drive
from two, and only the most recent request survives: dragging across a
10-minute file should decode the frame you stopped on, not the 300 you passed.
"""

import os
import threading
import tkinter as tk

import customtkinter as ctk
from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover - dialog degrades to a numbers-only picker
    cv2 = None

import guardrails


# The preview box. A VR SBS frame is 2:1 or wider and a flat clip is 16:9, so
# frames are fitted inside this rather than stretched to it, and the box keeps
# its size so the dialog does not resize itself every time you scrub.
PREVIEW_W = 720
PREVIEW_H = 400

# The app forces dark mode at import, so the scrub bar can use flat colours
# instead of resolving a theme pair for a plain tk.Canvas.
BAR_TRACK = "#2b2b2b"
BAR_SELECTED = "#1f6aa5"
BAR_PLAYHEAD = "#e6e6e6"
BAR_BORDER = "#404040"

# Below this a "clip" is not a clip, and ffmpeg's -ss/-to would produce an
# empty or single-frame file that fails much later in the pipeline.
MIN_CLIP_SECONDS = 0.1


def format_timecode(seconds, decimals=1):
    """Seconds as mm:ss.s, or h:mm:ss.s once there is an hour to show."""
    if seconds is None:
        return ""
    seconds = max(0.0, float(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    width = 2 if decimals <= 0 else 3 + decimals
    if hours:
        return f"{int(hours)}:{int(minutes):02d}:{secs:0{width}.{decimals}f}"
    return f"{int(minutes)}:{secs:0{width}.{decimals}f}"


def parse_timecode(text):
    """
    Seconds from '12.5', '1:23' or '1:02:03.5'; None for empty or unreadable.

    Colon-separated fields read as h:m:s from the right, which is how every
    player writes a position down - so a position copied out of one pastes in.
    """
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    try:
        if ":" in text:
            parts = [part.strip() for part in text.split(":")]
            if len(parts) > 3 or any(part == "" for part in parts):
                return None
            total = 0.0
            for part in parts:
                total = total * 60.0 + float(part)
            return total
        return float(text)
    except ValueError:
        return None


def _clamp(value, low, high):
    return max(low, min(high, value))


class ClipRangeDialog(ctk.CTkToplevel):
    """
    Modal in/out point picker. Read `.result` after the window closes:

      (start, end)   seconds, a real sub-range of the video
      (None, None)   the whole video - the caller should clear its trim boxes
      None           cancelled, leave whatever was there alone
    """

    def __init__(self, parent, video_path, start=None, end=None):
        super().__init__(parent)

        self.video_path = video_path
        self.result = None

        probe = guardrails.probe_media(video_path)
        self.duration = max(MIN_CLIP_SECONDS, float(probe.get("duration") or 0.0))
        self.fps = float(probe.get("fps") or 30.0) or 30.0

        self.start = _clamp(start if start is not None else 0.0, 0.0, self.duration)
        self.end = _clamp(end if end is not None else self.duration, 0.0, self.duration)
        if self.end - self.start < MIN_CLIP_SECONDS:
            self.start, self.end = 0.0, self.duration
        self.pos = self.start

        self._closing = False
        self._lock = threading.Lock()
        self._request = None      # seconds the decode thread should seek to next
        self._ready = None        # the newest decoded frame, waiting to be shown
        self._preview_error = False
        self._wake = threading.Event()
        self._worker = None
        self._pump_id = None
        self._photo = None

        self.title("Select clip - " + os.path.basename(video_path))
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind("<Escape>", lambda _e: self._cancel())

        self._build_ui()
        self._center_on(parent)
        self._start_worker()
        self._sync(request_frame=True)

        # CTkToplevel is not viewable the instant it is created, and grab_set on
        # a window that is not viewable raises. Claiming the grab a beat later
        # is the standard way round it.
        self.after(150, self._make_modal)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        pad = 14
        row = 0

        preview_frame = ctk.CTkFrame(self, width=PREVIEW_W, height=PREVIEW_H,
                                     fg_color="#111111", corner_radius=8)
        preview_frame.grid(row=row, column=0, padx=pad, pady=(pad, 6))
        preview_frame.grid_propagate(False)
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(0, weight=1)

        self.preview_label = ctk.CTkLabel(
            preview_frame,
            text=("Preview needs OpenCV, which is not installed.\n"
                  "The in and out points below still work."
                  if cv2 is None else "Loading frame..."),
            text_color="#888888",
        )
        self.preview_label.grid(row=0, column=0)

        row += 1
        self.pos_var = tk.DoubleVar(value=self.pos)
        self.scrub = ctk.CTkSlider(self, from_=0.0, to=self.duration,
                                   variable=self.pos_var, width=PREVIEW_W,
                                   command=self._on_scrub)
        self.scrub.grid(row=row, column=0, padx=pad, pady=(2, 2))

        row += 1
        # A plain canvas rather than a second slider: this shows the selected
        # span *and* the playhead at once, which two sliders cannot.
        self.bar = tk.Canvas(self, height=20, highlightthickness=1,
                             highlightbackground=BAR_BORDER, bd=0, bg=BAR_TRACK)
        self.bar.grid(row=row, column=0, sticky="ew", padx=pad, pady=(0, 8))
        self.bar.bind("<Configure>", lambda _e: self._draw_bar())
        self.bar.bind("<Button-1>", self._on_bar_click)
        self.bar.bind("<B1-Motion>", self._on_bar_click)

        row += 1
        nudge = ctk.CTkFrame(self, fg_color="transparent")
        nudge.grid(row=row, column=0, padx=pad, pady=(0, 8))
        self.pos_label = ctk.CTkLabel(nudge, text="", width=170,
                                      font=ctk.CTkFont(size=13, weight="bold"))
        self.pos_label.pack(side="left", padx=(0, 10))
        frame_step = 1.0 / self.fps
        for text, delta in (("-10s", -10.0), ("-1s", -1.0), ("-1f", -frame_step),
                            ("+1f", frame_step), ("+1s", 1.0), ("+10s", 10.0)):
            ctk.CTkButton(nudge, text=text, width=46,
                          command=lambda d=delta: self._nudge(d)).pack(side="left", padx=2)

        row += 1
        marks = ctk.CTkFrame(self, fg_color="transparent")
        marks.grid(row=row, column=0, padx=pad, pady=(0, 8))
        ctk.CTkButton(marks, text="[  Set Start", width=110,
                      command=self._set_start_here).pack(side="left", padx=4)
        ctk.CTkButton(marks, text="Set End  ]", width=110,
                      command=self._set_end_here).pack(side="left", padx=4)
        ctk.CTkButton(marks, text="Whole Video", width=110, fg_color="#444444",
                      hover_color="#555555",
                      command=self._select_all).pack(side="left", padx=(20, 4))

        row += 1
        fields = ctk.CTkFrame(self, fg_color="transparent")
        fields.grid(row=row, column=0, padx=pad, pady=(0, 6))

        self.start_var = tk.StringVar()
        self.end_var = tk.StringVar()
        for column, (label, var, jump) in enumerate((
            ("Start", self.start_var, "start"),
            ("End", self.end_var, "end"),
        )):
            ctk.CTkLabel(fields, text=label, font=ctk.CTkFont(weight="bold")).grid(
                row=0, column=column * 3, padx=(20 if column else 0, 6))
            entry = ctk.CTkEntry(fields, textvariable=var, width=110)
            entry.grid(row=0, column=column * 3 + 1)
            entry.bind("<Return>", lambda _e: self._read_entries())
            entry.bind("<FocusOut>", lambda _e: self._read_entries())
            ctk.CTkButton(fields, text="Go", width=40,
                          command=lambda which=jump: self._go_to(which)).grid(
                row=0, column=column * 3 + 2, padx=6)

        row += 1
        self.summary_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=11),
                                          text_color="#9a9a9a")
        self.summary_label.grid(row=row, column=0, padx=pad, pady=(0, 10))

        row += 1
        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.grid(row=row, column=0, padx=pad, pady=(0, pad))
        ctk.CTkButton(buttons, text="Cancel", width=110, fg_color="#444444",
                      hover_color="#555555", command=self._cancel).pack(side="left", padx=6)
        ctk.CTkButton(buttons, text="Use This Clip", width=150, height=36,
                      font=ctk.CTkFont(size=13, weight="bold"),
                      command=self._accept).pack(side="left", padx=6)

    def _center_on(self, parent):
        self.update_idletasks()
        try:
            x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
            y = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
            self.geometry(f"+{max(0, x)}+{max(0, y)}")
        except Exception:
            pass

    def _make_modal(self):
        if self._closing:
            return
        try:
            self.grab_set()
            self.lift()
            self.focus_force()
        except Exception:
            pass

    # --------------------------------------------------------------- state

    def _sync(self, request_frame=False):
        """Push the three numbers out to every widget that shows them."""
        self.pos_var.set(self.pos)
        # Two decimals in the boxes, one in the read-outs: rounding the marks
        # to a tenth would undo the +1f / -1f buttons on anything over 20 fps.
        self.start_var.set(format_timecode(self.start, 2))
        self.end_var.set(format_timecode(self.end, 2))
        self.pos_label.configure(
            text=f"{format_timecode(self.pos)}  /  {format_timecode(self.duration)}")

        span = max(0.0, self.end - self.start)
        frames = int(round(span * self.fps))
        if self.start <= 1e-3 and self.end >= self.duration - 1e-3:
            headline = f"Whole video - {format_timecode(self.duration)}"
        else:
            headline = (f"Clip {format_timecode(self.start)} -> "
                        f"{format_timecode(self.end)}   ({span:.1f}s of "
                        f"{self.duration:.1f}s)")
        self.summary_label.configure(
            text=f"{headline}   -   ~{frames:,} frames @ {self.fps:.4g} fps")

        self._draw_bar()
        if request_frame:
            self._request_frame(self.pos)

    def _set_pos(self, seconds, from_slider=False, request_frame=True):
        self.pos = _clamp(float(seconds), 0.0, self.duration)
        if not from_slider:
            self.pos_var.set(self.pos)
        self.pos_label.configure(
            text=f"{format_timecode(self.pos)}  /  {format_timecode(self.duration)}")
        self._draw_bar()
        if request_frame:
            self._request_frame(self.pos)

    def _on_scrub(self, value):
        self._set_pos(value, from_slider=True)

    def _nudge(self, delta):
        self._set_pos(self.pos + delta)

    def _on_bar_click(self, event):
        width = max(1, self.bar.winfo_width())
        self._set_pos((event.x / width) * self.duration)

    def _go_to(self, which):
        self._read_entries()
        self._set_pos(self.start if which == "start" else self.end)

    def _set_start_here(self):
        self.start = min(self.pos, max(0.0, self.end - MIN_CLIP_SECONDS))
        self._sync()

    def _set_end_here(self):
        self.end = max(self.pos, min(self.duration, self.start + MIN_CLIP_SECONDS))
        self._sync()

    def _select_all(self):
        self.start, self.end = 0.0, self.duration
        self._sync()

    def _read_entries(self):
        """
        Adopt whatever is typed in the two boxes, ignoring anything unreadable.

        Typed values are clamped rather than rejected: someone pasting a
        position from a player into a shorter re-encode should land on the end
        of the file, not get an error dialog mid-edit.
        """
        start = parse_timecode(self.start_var.get())
        end = parse_timecode(self.end_var.get())
        if start is not None:
            self.start = _clamp(start, 0.0, self.duration)
        if end is not None:
            self.end = _clamp(end, 0.0, self.duration)
        if self.end - self.start < MIN_CLIP_SECONDS:
            self.end = min(self.duration, self.start + MIN_CLIP_SECONDS)
            if self.end - self.start < MIN_CLIP_SECONDS:
                self.start = max(0.0, self.end - MIN_CLIP_SECONDS)
        self._sync()

    def _draw_bar(self):
        canvas = self.bar
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 1 or self.duration <= 0:
            return
        canvas.delete("all")

        def x_of(seconds):
            return (seconds / self.duration) * (width - 1)

        canvas.create_rectangle(x_of(self.start), 0, x_of(self.end), height,
                                fill=BAR_SELECTED, outline="")
        playhead = x_of(self.pos)
        canvas.create_line(playhead, 0, playhead, height, fill=BAR_PLAYHEAD, width=2)

    # -------------------------------------------------------------- frames

    def _start_worker(self):
        if cv2 is None:
            return
        self._worker = threading.Thread(target=self._frame_worker, daemon=True)
        self._worker.start()
        self._pump()

    def _pump(self):
        """
        Collect whatever the decode thread has left in the hand-off slot.

        Polling a slot, rather than letting the worker call self.after(), keeps
        every Tk call on this thread. A cross-thread after() blocks until the
        interpreter goes idle, so a worker posting a frame at the moment the UI
        thread is waiting on that worker deadlocks both - which is precisely
        what closing the dialog mid-decode used to do.
        """
        self._pump_id = None
        if self._closing:
            return
        with self._lock:
            image, self._ready = self._ready, None
            failed = self._preview_error
        if failed:
            self.preview_label.configure(
                text="Could not open this file for preview.\n"
                     "The in and out points below still work.")
            return  # the worker has already given up; nothing more will arrive
        if image is not None:
            self._show_frame(image)
        self._pump_id = self.after(60, self._pump)

    def _request_frame(self, seconds):
        with self._lock:
            self._request = seconds
        self._wake.set()

    def _frame_worker(self):
        """
        Own the capture from open to release, on this thread and no other.

        Releasing a VideoCapture from the UI thread while this one sits inside
        read() aborts the whole process from libavcodec - an async_lock
        assertion in pthread_frame.c, not a Python exception anything can
        catch. So the close path only sets a flag, and the release below is the
        one that ever runs.
        """
        cap = None
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                cap.release()
                cap = None
        except Exception:
            cap = None

        if cap is None:
            with self._lock:
                self._preview_error = True
            return

        try:
            while True:
                self._wake.wait()
                if self._closing:
                    return
                with self._lock:
                    seconds = self._request
                    self._request = None
                    self._wake.clear()
                if seconds is None:
                    continue
                image = self._grab(cap, seconds)
                if image is not None:
                    # Only the newest frame is worth keeping: anything still
                    # sitting here was overtaken while it was being decoded.
                    with self._lock:
                        self._ready = image
        finally:
            cap.release()

    def _grab(self, cap, seconds):
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000.0)
            ok, frame = cap.read()
            if not ok:
                # Seeking to the very end lands past the last decodable frame on
                # plenty of containers. Backing off half a second shows the last
                # real frame instead of a blank box.
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds - 0.5) * 1000.0)
                ok, frame = cap.read()
            if not ok:
                return None

            # Downscale before the colour convert and the PIL round-trip: an 8K
            # VR frame is 45 megapixels, and every operation on it at full size
            # costs more than the seek that produced it.
            height, width = frame.shape[:2]
            scale = min(PREVIEW_W / width, PREVIEW_H / height, 1.0)
            if scale < 1.0:
                frame = cv2.resize(frame, (max(1, int(width * scale)),
                                           max(1, int(height * scale))),
                                   interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame)
        except Exception:
            return None

    def _show_frame(self, image):
        if self._closing:
            return
        self._photo = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
        self.preview_label.configure(image=self._photo, text="")

    # ---------------------------------------------------------------- exit

    def _shutdown(self):
        # The worker owns the capture and closes it itself, so this is the whole
        # handshake: raise the flag, wake it, stop polling. Joining it here
        # would be worse than useless - the join holds the Tcl interpreter that
        # the worker would need to hand its last frame back through.
        self._closing = True
        self._wake.set()
        if self._pump_id is not None:
            try:
                self.after_cancel(self._pump_id)
            except Exception:
                pass
            self._pump_id = None
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

    def _cancel(self):
        self.result = None
        self._shutdown()

    def _accept(self):
        self._read_entries()
        # A selection that covers the file is reported as "no trim" rather than
        # as 0 -> duration, so the pipeline skips the trim re-encode entirely
        # instead of spending a pass to copy the video back to itself.
        if self.start <= 1e-3 and self.end >= self.duration - 1e-3:
            self.result = (None, None)
        else:
            self.result = (round(self.start, 3), round(self.end, 3))
        self._shutdown()


def select_clip(parent, video_path, start=None, end=None):
    """
    Show the picker modally and return what was chosen.

    (start, end) for a sub-range, (None, None) for the whole video, or None if
    the dialog was cancelled.
    """
    dialog = ClipRangeDialog(parent, video_path, start, end)
    parent.wait_window(dialog)
    return dialog.result
