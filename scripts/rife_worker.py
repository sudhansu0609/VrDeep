"""
RIFE frame interpolation, as a standalone worker process.

WHY A SEPARATE PROCESS
----------------------
The model holds ~13.8 GB of VRAM on a 4096x4096 frame, measured on the 16 GB
card in this machine. That cannot coexist with FaceFusion's own ~9 GB of ONNX
sessions, so interpolation runs as its own process, after the swap, and the
OS reclaims the VRAM wholesale when it exits. It is spawned into the guard's
Job Object like every other child, so the STOP button and the watchdog reach it.

WHY ONNX AND NOT PYTORCH
------------------------
The obvious RIFE implementations are PyTorch. That is a dead end here: this
machine has an RTX 5060 Ti (sm_120) and the installed torch is cu121, which
compiles for sm_50-sm_90 only - `torch.cuda.is_available()` returns True and
then every kernel launch fails. onnxruntime's CUDA provider works on the same
card today, which is what FaceFusion already runs on, so RIFE goes through
onnxruntime as well.

WHICH MODEL
-----------
vs-mlrt's `rife_v2` export of RIFE v4.15-lite. The v2 interface is the one
worth having: a single float32 input of shape (1, 7, H, W) - frame0 RGB,
frame1 RGB, then a constant timestep plane in (0,1) - and it does its own
internal padding, so an arbitrary frame size works with no rounding up to a
multiple of 32. Output is (1, 3, H, W) RGB in [0,1].

MEASURED ON THIS MACHINE
------------------------
Inference cost against frame size, one VR eye, CUDA:

    infer at      frame/s    peak VRAM (16.3 GB card)
    2048x2048        6.1      6.6 GB
    3072x3072        3.0      8.7 GB
    4096x4096        1.6     15.8 GB     <- 97% of the card

Quality, leave-one-out against real frames: predict frame n+1 from n and n+2 at
4096x4096 and score against the one that is really there. A linear crossfade is
the baseline, since it is what you get for nothing.

    crossfade                  42.73 dB
    infer at 2048, upscaled    45.05 dB    +2.32
    infer at 3072, upscaled    46.58 dB    +3.85
    infer at 4096 (full)       47.50 dB    +4.77

WHY THE DEFAULT IS NOT FULL RESOLUTION
--------------------------------------
Full resolution needs 15.8 GB of a 16.3 GB card. That leaves ~550 MB for the
desktop, the NVENC encoder this worker is feeding, and anything else on the
GPU - and when it does not fit, WDDM pages GPU memory into system RAM and the
rate collapses from 1.6 frame/s to about 0.03. That is not a theoretical risk;
it is what the first run of this worker did.

Inferring at 3/4 scale costs 0.92 dB, runs nearly twice as fast, and needs
8.7 GB, which leaves real headroom. Only the *synthesised* frames go through
it - every original frame is passed through untouched - so the cost applies to
half the frames of a 2x job and none of the source material.

Cost is still the thing to be clear-eyed about: doubling a 90 fps VR clip means
one new frame per existing frame, per eye. At 3 frame/s that is about an hour
per minute of finished video.
"""

import argparse
import os
import subprocess
import sys
import tempfile

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "rife")
DEFAULT_MODEL = os.path.join(MODEL_DIR, "rife_v4.15_lite.onnx")

# vs-mlrt publishes the ONNX exports as a 7-Zip archive of both interfaces.
MODEL_URL = ("https://github.com/AmusementClub/vs-mlrt/releases/download/"
             "external-models/rife_v4.15_lite.7z")
MODEL_MEMBER = "rife_v2/rife_v4.15_lite.onnx"
SEVENZIP_CANDIDATES = (
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
    "7z",
)


def emit(line):
    sys.stdout.write(line if line.endswith("\n") else line + "\n")
    sys.stdout.flush()


def fail(message):
    emit(f"RIFE_ERROR {message}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Model installation
# ---------------------------------------------------------------------------

def install_model(dest=DEFAULT_MODEL):
    """Download and extract the weights. Returns the path, or raises."""
    import shutil
    import urllib.request

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    seven = next((c for c in SEVENZIP_CANDIDATES
                  if c == "7z" or os.path.isfile(c)), None)
    if seven is None or (seven == "7z" and not shutil.which("7z")):
        raise RuntimeError(
            "7-Zip is needed to unpack the RIFE weights and was not found. "
            "Install it from https://www.7-zip.org/, or unpack "
            f"{MODEL_URL} by hand and put its {MODEL_MEMBER} at {dest}."
        )

    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "rife.7z")
        emit(f"RIFE_INFO downloading {os.path.basename(MODEL_URL)}")
        urllib.request.urlretrieve(MODEL_URL, archive)
        result = subprocess.run(
            [seven, "x", "-y", f"-o{tmp}", archive],
            capture_output=True, text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode != 0:
            raise RuntimeError(f"could not unpack the weights: {result.stdout[-300:]}")
        extracted = os.path.join(tmp, *MODEL_MEMBER.split("/"))
        if not os.path.isfile(extracted):
            raise RuntimeError(f"{MODEL_MEMBER} was not in the archive")
        shutil.copy(extracted, dest)
    return dest


# ---------------------------------------------------------------------------
# Video IO
# ---------------------------------------------------------------------------

def probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames:format=duration",
         "-of", "json", path],
        capture_output=True, text=True, timeout=120,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    import json
    data = json.loads(result.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    num, _, den = (stream.get("r_frame_rate") or "30/1").partition("/")
    fps = float(num) / float(den or 1)
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    frames = int(stream.get("nb_frames") or 0) or int(duration * fps)
    return {"width": int(stream.get("width") or 0),
            "height": int(stream.get("height") or 0),
            "fps": fps, "frames": frames}


def open_decoder(path, threads):
    cmd = ["ffmpeg", "-v", "error", "-threads", str(threads), "-i", path,
           "-an", "-sn", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    # stderr to a file: this process reads stdout for the whole run and would
    # never drain a second pipe, and a full stderr pipe stops ffmpeg dead.
    err = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return proc, err


def open_encoder(path, source, width, height, fps, encoder, crf, maxrate, threads):
    if encoder.endswith("_nvenc"):
        enc = ["-c:v", encoder, "-preset", "p7", "-rc", "vbr", "-cq", str(crf),
               "-maxrate", f"{maxrate}M", "-bufsize", f"{maxrate * 2}M"]
    else:
        enc = ["-c:v", encoder, "-preset", "faster", "-crf", str(crf),
               "-threads", str(threads)]
    cmd = (["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", f"{fps:.10f}", "-i", "pipe:0",
            "-threads", str(threads), "-i", source,
            "-map", "0:v:0", "-map", "1:a:0?"]
           # No -shortest. It ends the output at whichever stream runs out
           # first, so an audio track even slightly shorter than the video -
           # which is normal, audio is written in whole frames of its own -
           # silently truncates the video we just spent hours interpolating.
           # The video is the thing being produced; it decides the length.
           + enc + ["-c:a", "aac", "-b:a", "192k", path])
    err = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=err,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return proc, err


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RIFE frame interpolation")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--multi", type=int, default=2,
                        help="output frames per input frame: 2 doubles the fps")
    parser.add_argument("--encoder", default="hevc_nvenc")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--maxrate", type=int, default=120)
    parser.add_argument("--decode-threads", type=int, default=4)
    parser.add_argument("--provider", default="cuda")
    parser.add_argument("--infer-scale", type=float, default=0.75,
                        help="run the model at this fraction of the frame size "
                             "and upscale the result (1.0 = full resolution)")
    parser.add_argument("--install-model", action="store_true",
                        help="fetch the weights and exit")
    args = parser.parse_args()

    if args.install_model:
        try:
            path = install_model(args.model)
        except Exception as exc:
            fail(str(exc))
        emit(f"RIFE_INFO model installed at {path}")
        return 0

    if not args.input or not args.output:
        fail("--input and --output are required")
    if args.multi < 2:
        fail("--multi must be 2 or more; 2 doubles the frame rate")
    if not os.path.isfile(args.model):
        fail(f"the RIFE weights are not installed. Expected {args.model}. "
             f"Run: python scripts/rife_worker.py --install-model")

    import numpy
    import onnxruntime

    info = probe(args.input)
    width, height, fps = info["width"], info["height"], info["fps"]
    if not width or not height:
        fail(f"could not read a video stream from {args.input}")

    total_out = max(1, (info["frames"] - 1) * args.multi + 1) if info["frames"] else 0
    emit(f"RIFE_INFO {width}x{height} @ {fps:.3f} fps, {info['frames']} frames "
         f"-> {fps * args.multi:.3f} fps, {total_out} frames")

    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "cuda" in args.provider else ["CPUExecutionProvider"])
    try:
        session = onnxruntime.InferenceSession(args.model, options, providers=providers)
    except Exception as exc:
        fail(f"could not load the RIFE model: {type(exc).__name__}: {exc}")
    active = session.get_providers()[0]
    if "cuda" in args.provider and active != "CUDAExecutionProvider":
        # Silently landing on CPU turns hours into weeks. Same rule as the swap.
        fail("onnxruntime fell back to the CPU provider for RIFE, which would take "
             "weeks at this resolution. Check the CUDA install.")
    emit(f"RIFE_INFO running on {active}")

    input_name = session.get_inputs()[0].name
    frame_bytes = width * height * 3

    # Inference size. Rounded to a multiple of 8 purely so the resize is clean;
    # the v2 model pads internally and accepts any size.
    scale = min(1.0, max(0.25, args.infer_scale))
    infer_w = max(64, int(round(width * scale / 8)) * 8)
    infer_h = max(64, int(round(height * scale / 8)) * 8)
    reduced = (infer_w, infer_h) != (width, height)
    if reduced:
        import cv2
        emit(f"RIFE_INFO inferring at {infer_w}x{infer_h} and upscaling the "
             f"synthesised frames; originals pass through untouched")

    # One tensor, reused. At 4096x4096 this is 470 MB; reallocating it per frame
    # would churn half a gigabyte per interpolated frame.
    tensor = numpy.empty((1, 7, infer_h, infer_w), numpy.float32)

    decoder, dec_err = open_decoder(args.input, args.decode_threads)
    encoder, enc_err = open_encoder(args.output, args.input, width, height,
                                    fps * args.multi, args.encoder, args.crf,
                                    args.maxrate, args.decode_threads)

    def read_frame():
        buf = decoder.stdout.read(frame_bytes)
        if not buf or len(buf) < frame_bytes:
            return None
        return numpy.frombuffer(buf, numpy.uint8).reshape(height, width, 3)

    written = 0
    try:
        previous = read_frame()
        if previous is None:
            fail("no frames were decoded from the input video")
        previous_small = (cv2.resize(previous, (infer_w, infer_h),
                                     interpolation=cv2.INTER_AREA) if reduced else None)

        while True:
            current = read_frame()
            if current is None:
                break

            encoder.stdin.write(previous.tobytes())
            written += 1

            # Both source frames stay put for the whole group; only the
            # timestep plane changes between the frames we synthesise.
            if reduced:
                # Each frame is the right-hand side of one pair and then the
                # left-hand side of the next, so downscaling it twice is pure
                # waste - at 4096x4096 an INTER_AREA resize is not cheap.
                a = previous_small
                b = cv2.resize(current, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
                previous_small = b
            else:
                a, b = previous, current
            tensor[0, 0:3] = a.transpose(2, 0, 1)
            tensor[0, 3:6] = b.transpose(2, 0, 1)
            tensor[0, 0:6] /= 255.0

            for step in range(1, args.multi):
                tensor[0, 6] = step / float(args.multi)
                result = session.run(None, {input_name: tensor})[0]
                mid = numpy.clip(result[0].transpose(1, 2, 0) * 255.0, 0, 255)
                mid = mid.astype(numpy.uint8)
                if reduced:
                    mid = cv2.resize(mid, (width, height), interpolation=cv2.INTER_CUBIC)
                encoder.stdin.write(mid.tobytes())
                written += 1

            previous = current
            if written % 10 < args.multi:
                emit(f"RIFE_PROGRESS {written} {total_out}")

        # The last source frame has no successor to interpolate towards.
        encoder.stdin.write(previous.tobytes())
        written += 1
    except BrokenPipeError:
        fail("the encoder stopped early; the output was not written")
    finally:
        for pipe in (decoder.stdout, encoder.stdin):
            try:
                pipe.close()
            except Exception:
                pass
        decoder.wait()

    encoder.wait()
    enc_err.seek(0)
    message = (enc_err.read() or b"").decode(errors="replace")
    enc_err.close()
    dec_err.close()
    if encoder.returncode != 0:
        fail("ffmpeg encode failed: " + message[:400])

    emit(f"RIFE_PROGRESS {written} {written}")
    emit(f"RIFE_DONE {written} frames at {fps * args.multi:.3f} fps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
