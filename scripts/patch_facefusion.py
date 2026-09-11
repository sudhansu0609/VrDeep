"""
Apply VrDeep's memory patches to the vendored FaceFusion checkout.

FaceFusion lives in facefusion/, which is a clone of an upstream repository and
is deliberately NOT tracked by this repo (see .gitignore). Anything we change in
there is therefore invisible to git and would be lost on the next `git pull` in
that clone - so the changes live here, as idempotent rewrites, instead of as
edits nobody can see.

Both patches address the same thing: at 8192x4096 source, one VR eye is a
4096x4096 frame, i.e. **48 MB of raw pixels per frame**, and FaceFusion's
video path was written for 1080p frames that are 1/16th of that.

  1. ffmpeg.py - the raw-frame readers ran with ffmpeg's default thread count.
     On a 28-core machine that is 16 frame-decoding threads, each with its own
     pool of decoded 48 MB frames. Measured on this machine during a left-eye
     pass: two readers and a writer holding 2.35 + 1.38 + 2.30 = 6.0 GB
     resident, out of 8.5 GB for the whole stage. Capping the reader threads
     cuts most of that, and the decode still runs an order of magnitude faster
     than the swapper consumes frames.

  2. workflows/to_video.py - process_memory_frames() submitted a future for
     EVERY frame of the video up front and only then started draining them.
     ThreadPoolExecutor's work queue is unbounded, so there was no backpressure
     at all: any stall on the writer let the worker threads run ahead, and each
     frame they ran ahead by held another 48 MB result alive. A 60s 90fps eye is
     5400 frames, so the worst case was several hundred GB. Submitting inside a
     bounded window keeps the queue at a few frames and cannot regress
     throughput - the workers were never the thing being starved.

  3. inference_manager.py - every InferenceSession was built with default
     SessionOptions, so onnxruntime gave each of FaceFusion's ~8 models its own
     intra-op pool of one thread per core (14 on this machine) and let those
     threads busy-spin while they waited for the GPU. Measured: the FaceFusion
     python process alone held 92% of a 28-core box, and the machine sat at 82%
     CPU, while the GPU was only 70% busy. Two intra-op threads per session with
     spinning turned off takes the machine to 20% CPU at identical speed
     (98.8 s against 99.3 s over a 183-frame 4096x4096 pass). Nothing about the
     swap gets faster; what changes is that there is now a machine left over to
     run a split, a merge or a second lane alongside it.

Usage:
    python scripts/patch_facefusion.py            apply (idempotent)
    python scripts/patch_facefusion.py --check    report status, change nothing
"""

import io
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACEFUSION_DIR = os.path.join(PROJECT_DIR, "facefusion")

MARKER = "VrDeep:"

# FaceFusion is indented with tabs. Keep it that way - a mixed-indent file is a
# syntax error in the middle of somebody else's codebase.
READER_OLD = """def create_video_reader(video_path : str, frame_number : int, video_metadata : VideoReaderMetadata) -> subprocess.Popen[bytes]:
\tcommands = ffmpeg_builder.chain(
\t\tffmpeg_builder.seek_to(frame_number / video_metadata.get('fps')),
\t\tffmpeg_builder.set_input(video_path),
"""

READER_NEW = """# VrDeep: decode threads for the raw-frame readers. ffmpeg's default is one per
# core, and every frame thread holds its own pool of decoded frames - 48 MB each
# at 4096x4096, which is one VR eye. Four threads still decode far faster than
# the swapper consumes, at roughly a quarter of the memory.
VRDEEP_READER_THREADS = int(os.environ.get('VRDEEP_FFMPEG_READER_THREADS', '4'))


def create_video_reader(video_path : str, frame_number : int, video_metadata : VideoReaderMetadata) -> subprocess.Popen[bytes]:
\tcommands = ffmpeg_builder.chain(
\t\tffmpeg_builder.set_thread_count(VRDEEP_READER_THREADS),
\t\tffmpeg_builder.seek_to(frame_number / video_metadata.get('fps')),
\t\tffmpeg_builder.set_input(video_path),
"""

WINDOW_OLD = """\t\t\twith ThreadPoolExecutor(max_workers = state_manager.get_item('execution_thread_count')) as executor:
\t\t\t\tfutures : Deque[Future[VisionFrame]] = deque()

\t\t\t\tfor frame_number in temp_frame_range:
\t\t\t\t\tfuture = executor.submit(process_memory_frame, frame_number, temp_video_resolution, output_video_resolution)
\t\t\t\t\tfutures.append(future)

\t\t\t\twhile futures:
\t\t\t\t\tfuture = futures.popleft()

\t\t\t\t\tif is_process_stopping():

\t\t\t\t\t\tfor pending_future in futures:
\t\t\t\t\t\t\tpending_future.cancel()

\t\t\t\t\t\tfutures.clear()

\t\t\t\t\telse:
\t\t\t\t\t\tvideo_manager.write_video_frame(video_writer, future.result())
\t\t\t\t\t\tprogress.update()
"""

WINDOW_NEW = """\t\t\twith ThreadPoolExecutor(max_workers = state_manager.get_item('execution_thread_count')) as executor:
\t\t\t\tfutures : Deque[Future[VisionFrame]] = deque()
\t\t\t\t# VrDeep: submit inside a bounded window instead of queueing every
\t\t\t\t# frame of the video up front. ThreadPoolExecutor's queue is
\t\t\t\t# unbounded, so the original loop had no backpressure - a slow writer
\t\t\t\t# let the workers run arbitrarily far ahead, and every frame of lead
\t\t\t\t# holds a full result frame alive (48 MB at 4096x4096, one VR eye).
\t\t\t\t# The window is deep enough that no worker ever waits for work.
\t\t\t\tframe_numbers = iter(temp_frame_range)
\t\t\t\twindow = state_manager.get_item('execution_thread_count') * 2 + 2

\t\t\t\tdef submit_next() -> bool:
\t\t\t\t\tframe_number = next(frame_numbers, None)

\t\t\t\t\tif frame_number is None:
\t\t\t\t\t\treturn False

\t\t\t\t\tfutures.append(executor.submit(process_memory_frame, frame_number, temp_video_resolution, output_video_resolution))
\t\t\t\t\treturn True

\t\t\t\tfor _ in range(window):
\t\t\t\t\tif not submit_next():
\t\t\t\t\t\tbreak

\t\t\t\twhile futures:
\t\t\t\t\tfuture = futures.popleft()

\t\t\t\t\tif is_process_stopping():

\t\t\t\t\t\tfor pending_future in futures:
\t\t\t\t\t\t\tpending_future.cancel()

\t\t\t\t\t\tfutures.clear()

\t\t\t\t\telse:
\t\t\t\t\t\tvideo_manager.write_video_frame(video_writer, future.result())
\t\t\t\t\t\tprogress.update()
\t\t\t\t\t\tsubmit_next()
"""

# The imports are a separate patch tuple from the session change so --check
# reports each half: the two live in the same file but nothing stops upstream
# from moving one and not the other.
IMPORT_OLD = """import importlib
import random
from functools import lru_cache
from time import sleep, time
from typing import List

from onnxruntime import InferenceSession
"""

IMPORT_NEW = """import importlib
import os
import random
from functools import lru_cache
from time import sleep, time
from typing import List

# VrDeep: the onnxruntime module itself, for SessionOptions, and os for the
# environment knob that sizes its thread pools.
import onnxruntime
from onnxruntime import InferenceSession
"""

SESSION_OLD = """def create_inference_session(model_path : str, inference_providers : List[InferenceProvider]) -> InferenceSession:
\tmodel_file_name = get_file_name(model_path)
\tstart_time = time()

\ttry:
\t\tinference_session = InferenceSession(model_path, providers = inference_providers)
"""

SESSION_NEW = """# VrDeep: intra-op threads per inference session, overridable from the child
# environment (engine.run_facefusion sets it from guard_config). Two is enough
# to keep the card fed - the work is on the GPU, not in the CPU kernels - and
# eight models at two threads is a fraction of what the defaults spin up.
VRDEEP_ORT_INTRA_THREADS = int(os.environ.get('VRDEEP_ORT_INTRA_THREADS', '2'))


def create_inference_session(model_path : str, inference_providers : List[InferenceProvider]) -> InferenceSession:
\tmodel_file_name = get_file_name(model_path)
\tstart_time = time()

\ttry:
\t\t# VrDeep: bounded, non-spinning thread pools. Default SessionOptions give
\t\t# every model its own intra-op pool of one thread per core (14 here) and
\t\t# let those threads busy-spin between GPU ops: 82% of the whole machine
\t\t# for one pass, of which 62 points were spin-waiting. intra_op 2 with
\t\t# spinning off is 20% at the same speed (98.8 s against 99.3 s over 183
\t\t# frames at 4096x4096), which is what leaves CPU for anything else.
\t\tsession_options = onnxruntime.SessionOptions()
\t\tsession_options.intra_op_num_threads = VRDEEP_ORT_INTRA_THREADS
\t\tsession_options.inter_op_num_threads = 1
\t\tsession_options.add_session_config_entry('session.intra_op.allow_spinning', '0')
\t\tsession_options.add_session_config_entry('session.inter_op.allow_spinning', '0')
\t\tinference_session = InferenceSession(model_path, sess_options = session_options, providers = inference_providers)
"""

PATCHES = [
    (os.path.join(FACEFUSION_DIR, "facefusion", "ffmpeg.py"),
     "capped ffmpeg reader decode threads", READER_OLD, READER_NEW),
    (os.path.join(FACEFUSION_DIR, "facefusion", "workflows", "to_video.py"),
     "bounded the in-flight frame window", WINDOW_OLD, WINDOW_NEW),
    (os.path.join(FACEFUSION_DIR, "facefusion", "inference_manager.py"),
     "onnxruntime imported for session options", IMPORT_OLD, IMPORT_NEW),
    (os.path.join(FACEFUSION_DIR, "facefusion", "inference_manager.py"),
     "bounded onnxruntime thread pools", SESSION_OLD, SESSION_NEW),
]


def status():
    """(applied, missing, broken) lists of human-readable patch descriptions."""
    applied, missing, broken = [], [], []
    for path, label, old, new in PATCHES:
        try:
            text = io.open(path, encoding="utf-8").read()
        except OSError:
            broken.append(f"{label} ({path} not found)")
            continue
        if new in text:
            applied.append(label)
        elif old in text:
            missing.append(label)
        else:
            # Neither form present: upstream changed underneath us, and blindly
            # rewriting would be worse than leaving it alone.
            broken.append(f"{label} (source no longer matches; needs re-doing by hand)")
    return applied, missing, broken


def apply():
    changed = []
    for path, label, old, new in PATCHES:
        text = io.open(path, encoding="utf-8").read()
        if new in text:
            continue
        if old not in text:
            continue
        io.open(path, "w", encoding="utf-8", newline="").write(text.replace(old, new, 1))
        changed.append(label)
    return changed


def main():
    if "--check" in sys.argv:
        applied, missing, broken = status()
        for item in applied:
            print(f"  applied : {item}")
        for item in missing:
            print(f"  MISSING : {item}")
        for item in broken:
            print(f"  BROKEN  : {item}")
        return 0 if not missing and not broken else 1

    changed = apply()
    applied, missing, broken = status()
    if changed:
        print("Patched FaceFusion:")
        for item in changed:
            print(f"  - {item}")
    else:
        print("FaceFusion already patched; nothing to do.")
    for item in broken:
        print(f"  WARNING: {item}")
    return 0 if not missing and not broken else 1


if __name__ == "__main__":
    sys.exit(main())
