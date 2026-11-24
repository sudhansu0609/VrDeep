"""
Pick which person in the target each source face replaces.

Runs character_worker.py over frames spread across the chosen portion of the
target, shows one tile per person it found, and lets the user assign a person
to Face 1 and (when there is a second source face) another to Face 2. What
comes back is each face's reference crops; the swap then follows exactly that
person by identity (scripts/facefusion_addons/vrdeep_slots.py) instead of
whichever face happens to be largest in each frame.

A person can be split across two tiles when the sampled views differ a lot
(profile against frontal). Assigning both tiles to the same face is fine - it
just gives that face more reference views.
"""

import json
import os
import subprocess
import threading
from datetime import datetime

import customtkinter as ctk
from PIL import Image

from engine import FACEFUSION_DIR, PROJECT_DIR, PYTHON, SCRIPTS_DIR

THUMB = 120
FACE_COLORS = {1: "#2f80ed", 2: "#e0457b"}
SAMPLES = 12


class CharacterPickerDialog(ctk.CTkToplevel):
    """
    Modal. Read `.result` after the window closes:

      {1: [crop, ...], 2: [crop, ...]}   the picks (an empty list clears a face)
      None                               cancelled, leave everything alone
    """

    def __init__(self, parent, video_path, config, layout=None, start=None, end=None,
                 two_faces=False, current=None):
        super().__init__(parent)
        self.video_path = video_path
        self.config = config or {}
        self.layout = layout or "none"
        self.start = start
        self.end = end
        self.two_faces = two_faces
        self.result = None
        self.people = []
        self.assigned = {}          # person number -> face number
        self.current = current or {}
        self._proc = None
        self._closing = False
        self._images = []           # CTkImage refs, or Tk drops them
        self.tiles = {}

        self.title("Pick characters - " + os.path.basename(video_path))
        self.geometry("760x620")
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind("<Escape>", lambda _e: self._cancel())
        self._build_ui()
        self.after(150, self._make_modal)
        self._scan()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        faces = "Face 1 and Face 2" if self.two_faces else "Face 1"
        ctk.CTkLabel(
            self, justify="left", wraplength=720,
            text=(f"Assign a person to {faces}. Each face then replaces only that person, "
                  f"wherever they are in the frame. Anyone left unassigned is not touched."
                  + ("" if self.two_faces else
                     "  (Set a Second Face on the tab to swap a second person.)")),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 4))

        self.status = ctk.CTkLabel(self, text="", text_color="#888888")
        self.status.grid(row=1, column=0, sticky="w", padx=14)

        self.grid_frame = ctk.CTkScrollableFrame(self)
        self.grid_frame.grid(row=2, column=0, sticky="nsew", padx=14, pady=8)
        for column in range(4):
            self.grid_frame.grid_columnconfigure(column, weight=1)

        self.progress = ctk.CTkProgressBar(self)
        self.progress.set(0)
        self.progress.grid(row=3, column=0, sticky="ew", padx=14)

        self.summary = ctk.CTkLabel(self, text="", anchor="w")
        self.summary.grid(row=4, column=0, sticky="w", padx=14, pady=(6, 0))

        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.grid(row=5, column=0, sticky="e", padx=14, pady=12)
        self.rescan_btn = ctk.CTkButton(buttons, text="Rescan", width=90,
                                        fg_color="#444444", hover_color="#555555",
                                        command=self._scan)
        self.rescan_btn.grid(row=0, column=0, padx=4)
        ctk.CTkButton(buttons, text="Cancel", width=90, fg_color="#444444",
                      hover_color="#555555", command=self._cancel).grid(row=0, column=1, padx=4)
        self.save_btn = ctk.CTkButton(buttons, text="Use these", width=110, command=self._save)
        self.save_btn.grid(row=0, column=2, padx=4)
        self._sync_summary()

    def _make_modal(self):
        try:
            self.grab_set()
            self.focus_force()
        except Exception:
            pass

    # ---------------------------------------------------------------- scan

    def _scan(self):
        if self._proc is not None:
            return
        for child in self.grid_frame.winfo_children():
            child.destroy()
        self.people, self.assigned, self.tiles, self._images = [], {}, {}, []
        self.rescan_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        self.progress.set(0)
        self.status.configure(text=f"Looking for faces in {SAMPLES} frames across the clip "
                                   f"(the first scan loads the face models, ~20 s)...")
        stem = os.path.splitext(os.path.basename(self.video_path))[0][:40]
        out_dir = os.path.join(PROJECT_DIR, "output", "characters",
                               f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        c = self.config
        cmd = [
            PYTHON, os.path.join(SCRIPTS_DIR, "character_worker.py"),
            "--video", self.video_path, "--out-dir", out_dir, "--layout", self.layout,
            "--samples", str(SAMPLES),
            "--provider", str(c.get("provider") or "cuda"),
            "--detector-model", str(c.get("detector_model") or "retinaface"),
            "--detector-size", str(c.get("detector_size") or "640x640"),
            "--detector-score", str(c.get("detector_score", 0.5)),
            "--detector-angles", str(c.get("detector_angles") or "0"),
            "--landmarker-model", str(c.get("landmarker_model") or "2dfan4"),
            "--landmarker-score", str(c.get("landmarker_score", 0.5)),
        ]
        if self.start:
            cmd += ["--start", str(float(self.start))]
        if self.end:
            cmd += ["--end", str(float(self.end))]
        threading.Thread(target=self._run_worker, args=(cmd,), daemon=True).start()

    def _run_worker(self, cmd):
        result, error, tail = None, None, []
        try:
            self._proc = subprocess.Popen(
                cmd, cwd=FACEFUSION_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in self._proc.stdout:
                line = line.strip()
                if line.startswith("CHARS_PROGRESS "):
                    try:
                        done, total = (int(x) for x in line.split()[1:3])
                        self.after(0, self.progress.set, done / max(1, total))
                    except (ValueError, RuntimeError):
                        pass
                elif line.startswith("CHARS_RESULT "):
                    result = json.loads(line[len("CHARS_RESULT "):])
                elif line.startswith(("CHARS_ERROR ", "FACESRC_ERROR ")):
                    error = line.split(" ", 1)[1]
                elif line:
                    tail = (tail + [line])[-8:]
            self._proc.wait()
        except Exception as exc:  # the dialog must come back either way
            error = str(exc)
        finally:
            self._proc = None
        if self._closing:
            return
        if result is None and not error:
            error = "\n".join(tail) or "The face scan stopped without a result."
        try:
            self.after(0, self._show_result, result, error)
        except RuntimeError:
            pass

    def _show_result(self, result, error):
        self.rescan_btn.configure(state="normal")
        self.progress.set(1)
        if error:
            self.status.configure(text="Face scan failed: " + error, text_color="#e05252")
            return
        self.people = result.get("people") or []
        self.status.configure(
            text=f"{len(self.people)} person(s) found in {result.get('samples')} frames - "
                 f"most frequent first.", text_color="#888888")
        for index, person in enumerate(self.people):
            self._add_tile(index, person)
        self.save_btn.configure(state="normal")
        self._sync_summary()

    def _add_tile(self, index, person):
        number = person["person"]
        tile = ctk.CTkFrame(self.grid_frame, border_width=3, border_color="#333333")
        tile.grid(row=index // 4, column=index % 4, padx=6, pady=6, sticky="n")
        try:
            image = Image.open(person["thumb"])
            image.thumbnail((THUMB, THUMB))
            photo = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
            self._images.append(photo)
            ctk.CTkLabel(tile, image=photo, text="").grid(row=0, column=0, columnspan=2,
                                                          padx=6, pady=(6, 2))
        except Exception:
            ctk.CTkLabel(tile, text="(no preview)", width=THUMB, height=THUMB).grid(
                row=0, column=0, columnspan=2)
        details = " ".join(x for x in (person.get("gender"), person.get("age")) if x)
        ctk.CTkLabel(tile, font=ctk.CTkFont(size=11),
                     text=f"Person {number}\n{details}\nin {person['appearances']} frame(s)"
                     ).grid(row=1, column=0, columnspan=2)
        faces = [1, 2] if self.two_faces else [1]
        for column, face in enumerate(faces):
            ctk.CTkButton(tile, text=f"Face {face}", width=58, height=24,
                          fg_color=FACE_COLORS[face],
                          command=lambda n=number, f=face: self._toggle(n, f)).grid(
                row=2, column=column, columnspan=1 if self.two_faces else 2,
                padx=3, pady=(2, 6))
        self.tiles[number] = tile

    def _toggle(self, number, face):
        if self.assigned.get(number) == face:
            del self.assigned[number]
        else:
            self.assigned[number] = face
        for person, tile in self.tiles.items():
            face_for = self.assigned.get(person)
            tile.configure(border_color=FACE_COLORS[face_for] if face_for else "#333333")
        self._sync_summary()

    def _picked(self, face):
        return [p["person"] for p in self.people if self.assigned.get(p["person"]) == face]

    def _sync_summary(self):
        bits = []
        for face in ([1, 2] if self.two_faces else [1]):
            people = self._picked(face)
            if people:
                text = " + ".join(f"Person {n}" for n in people)
            elif self.current.get(face):
                text = "unchanged (picked earlier)"
            else:
                text = "automatic (largest face of its gender)"
            bits.append(f"Face {face} -> {text}")
        self.summary.configure(text="     ".join(bits))

    # --------------------------------------------------------------- close

    def _save(self):
        result = {}
        for face in ([1, 2] if self.two_faces else [1]):
            people = self._picked(face)
            if people:
                result[face] = [ref for p in self.people if p["person"] in people
                                for ref in p["refs"]]
        self.result = result
        self._close()

    def _cancel(self):
        self.result = None
        self._close()

    def _close(self):
        self._closing = True
        proc = self._proc
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()


def pick_characters(parent, video_path, config, layout=None, start=None, end=None,
                    two_faces=False, current=None):
    dialog = CharacterPickerDialog(parent, video_path, config, layout=layout, start=start,
                                   end=end, two_faces=two_faces, current=current)
    parent.wait_window(dialog)
    return dialog.result
