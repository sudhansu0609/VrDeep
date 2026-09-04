"""
Real-ESRGAN upscale, in its own process.

This used to run inside the GUI process. That was the only stage in the whole
pipeline that did, and it cost three separate things:

  * torch never goes away. Importing it alone adds ~350 MB RSS to the GUI
    (measured 16 MB -> 364 MB) and Python cannot unload a native extension, so
    the app stayed that much heavier for the rest of the session after a single
    upscale.
  * the CUDA context it creates is process-lifetime too, and torch's caching
    allocator does not return freed blocks to the driver without an explicit
    empty_cache(), so VRAM stayed reserved as well.
  * the GUI process is not inside the guard's Job Object, so this stage ran
    with no memory cap, no CPU cap, and out of reach of the watchdog and the
    Stop button - exactly the stage most able to exhaust the card.

Running it here fixes all three at once: everything it allocates leaves with
the process, and because the parent spawns it through the guard it inherits
the job object, so the caps and the Stop button cover it like every other
stage. ffmpeg is started from in here and inherits job membership too.

Progress goes to stdout as `ESRGAN_PROGRESS <done> <total>` lines; fatal
problems go out as a single `ESRGAN_ERROR <message>` line so the parent can
show something better than a traceback.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

# realesr-general-x4v3 is an SRVGGNetCompact with 32 conv blocks, and it is a
# 4x model regardless of the output scale asked for - RealESRGANer takes the
# model's own scale, and enhance() takes the one we actually want.
MODEL_NUM_CONV = 32
MODEL_NATIVE_SCALE = 4


def fail(message):
    sys.stdout.write("ESRGAN_ERROR " + " ".join(str(message).split()) + "\n")
    sys.stdout.flush()
    sys.exit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--model-path", required=True)
    # Everything that decides whether Real-ESRGAN can run here at all - the
    # weights, and whether torch was built for this GPU - is checked below
    # before a single frame is read. Exposing that as a mode lets
    # engine.prepare_run reuse this exact code rather than keep a second copy
    # that can drift, and refuse the run before the swap instead of after it.
    parser.add_argument("--preflight", action="store_true",
                        help="Report whether Real-ESRGAN can run on this machine, "
                             "then exit without touching any video.")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--target-width", type=int, default=0)
    parser.add_argument("--tile", type=int, default=512)
    # JSON, not a plain string: the flags start with "-c:v" and argparse reads
    # any value beginning with a dash as the next option.
    parser.add_argument("--encoder-args",
                        help="Encoder flags for ffmpeg, as a JSON list.")
    args = parser.parse_args()

    encoder_args = None
    if not args.preflight:
        missing = [name for name, value in (("--input", args.input),
                                            ("--output", args.output),
                                            ("--encoder-args", args.encoder_args))
                   if not value]
        if missing:
            fail(f"Missing required argument(s): {', '.join(missing)}")
        try:
            encoder_args = [str(a) for a in json.loads(args.encoder_args)]
        except Exception as exc:
            fail(f"Could not parse --encoder-args: {type(exc).__name__}: {exc}")

    if not os.path.exists(args.model_path):
        fail(
            f"Real-ESRGAN weights not found at {args.model_path}. Either drop "
            f"realesr-general-x4v3.pth there, or set upscale_method to 'ffmpeg', or "
            f"use FaceFusion's ONNX frame enhancer instead "
            f"(frame_enhancer: 'real_esrgan_x2_fp16'), which runs on the onnxruntime "
            f"GPU stack that is already working."
        )

    import cv2

    try:
        import torch
        from realesrgan import RealESRGANer
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    except Exception as exc:
        fail(f"{type(exc).__name__}: {exc}")

    # torch.cuda.is_available() returns True even when the installed build has
    # no kernels for this GPU; the failure only shows up as "no kernel image is
    # available for execution on the device" once work is submitted. Check the
    # compiled architecture list instead, before anything is staged.
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        arch = f"sm_{capability[0]}{capability[1]}"
        if arch not in torch.cuda.get_arch_list():
            fail(
                f"The installed PyTorch ({torch.__version__}) has no kernels for "
                f"{torch.cuda.get_device_name(0)} ({arch}); it was built for "
                f"{', '.join(torch.cuda.get_arch_list())}. Either install a matching "
                f"build into facefusion/venv, or set upscale_method to 'ffmpeg', or "
                f"use FaceFusion's ONNX frame enhancer instead "
                f"(frame_enhancer: 'real_esrgan_x2_fp16'), which runs on the "
                f"onnxruntime GPU stack that is already working."
            )

    if args.preflight:
        sys.stdout.write("ESRGAN_PREFLIGHT_OK\n")
        sys.stdout.flush()
        return 0

    cap = cv2.VideoCapture(args.input)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0 or orig_w <= 0:
        cap.release()
        fail("No frames found in the input video.")

    out_w, out_h = orig_w * args.scale, orig_h * args.scale
    if args.target_width:
        aspect = orig_h / orig_w
        out_w = (args.target_width + 1) // 2 * 2
        out_h = (int(args.target_width * aspect) + 1) // 2 * 2

    try:
        # RealESRGANer does NOT build the architecture for you - without an
        # explicit `model` it leaves it as None and dies inside
        # load_state_dict. `scale` here is the checkpoint's own scale; the
        # output scale is enhance()'s outscale.
        upsampler = RealESRGANer(
            scale=MODEL_NATIVE_SCALE,
            model_path=args.model_path,
            model=SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                                  num_conv=MODEL_NUM_CONV,
                                  upscale=MODEL_NATIVE_SCALE, act_type="prelu"),
            tile=args.tile, tile_pad=10, pre_pad=0, half=True, device="cuda",
        )
    except Exception as exc:
        cap.release()
        fail(f"Could not load the Real-ESRGAN model: {type(exc).__name__}: {exc}")

    encoder_proc = None
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                # .enhance(), not __call__ - RealESRGANer is not callable.
                upscaled, _ = upsampler.enhance(rgb, outscale=args.scale)
            except Exception as exc:
                fail(f"Real-ESRGAN failed on frame {index}: {type(exc).__name__}: {exc}")
            bgr = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)

            if encoder_proc is None:
                # Size the raw pipe from what the model actually returned
                # rather than from what we assumed it would return.
                raw_h, raw_w = bgr.shape[:2]
                cmd = (
                    ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                     "-f", "rawvideo", "-pix_fmt", "bgr24",
                     "-s", f"{raw_w}x{raw_h}",
                     "-r", str(fps), "-i", "pipe:0",
                     "-i", args.input,
                     "-vf", f"scale={out_w}:{out_h}:flags=lanczos",
                     "-map", "0:v:0", "-map", "1:a:0?"]
                    + encoder_args
                    + ["-c:a", "aac", "-b:a", "192k", "-shortest", args.output]
                )
                # Raw frames straight into ffmpeg; staging them as PNG is what
                # used to fill the temp drive.
                # stderr to a temp FILE, not a pipe. This loop writes frames
                # into stdin for the whole run and only reads ffmpeg's output
                # at the end via communicate(), so an unread pipe that fills
                # would block ffmpeg and hang the encode permanently. A file
                # cannot fill.
                encoder_stderr = tempfile.TemporaryFile()
                encoder_proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                    stderr=encoder_stderr,
                    # This worker is itself spawned without a console, so its
                    # own children would each allocate one to write into.
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

            encoder_proc.stdin.write(bgr.tobytes())
            index += 1
            if index % 10 == 0 or index == total:
                sys.stdout.write(f"ESRGAN_PROGRESS {index} {total}\n")
                sys.stdout.flush()
    except BrokenPipeError:
        pass
    finally:
        cap.release()
        if encoder_proc is not None:
            try:
                encoder_proc.stdin.close()
            except Exception:
                pass

    if encoder_proc is None:
        fail("No frames were decoded from the input video.")

    encoder_proc.wait()
    encoder_stderr.seek(0)
    stderr = encoder_stderr.read()
    encoder_stderr.close()
    if encoder_proc.returncode != 0:
        fail("ffmpeg encode failed: " + (stderr or b"").decode(errors="replace")[:400])

    sys.stdout.write(f"ESRGAN_PROGRESS {index} {total}\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
