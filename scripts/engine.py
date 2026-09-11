import concurrent.futures
import subprocess
import sys
import os
import hashlib
import json
import time
import shutil
import threading
import re

import guardrails
import vr_metadata
from guardrails import GuardAbort, ResourceGuard, NULL_GUARD

# NOTE: this module deliberately imports nothing heavy. Every stage that needs
# a model - FaceFusion, the CUDA probe, Real-ESRGAN - runs in its own process,
# so torch, onnxruntime, cv2 and their CUDA contexts never enter the GUI and
# there is nothing left to release when a run ends. cv2 and numpy used to be
# imported here for the in-process Real-ESRGAN loop; that loop now lives in
# esrgan_worker.py, so they went with it.


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACEFUSION_DIR = os.path.join(PROJECT_DIR, "facefusion")
PYTHON = os.path.join(FACEFUSION_DIR, "venv", "Scripts", "python.exe")
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONFIG = {
    # hyperswap is the newest swapper in FaceFusion 3.8 and holds identity far
    # better than simswap_unofficial_512, which is soft and washes colour out.
    # It is also faster: measured 1.95 vs 1.39 frame/s on a 4K frame here.
    "swapper_model": "hyperswap_1a_256",
    # Pixel boost tiles the crop, so 512 on a 256 model is a batch of 4 and
    # 1024 is a batch of 16. hyperswap at 512x512 asked for a single 4.5 GB
    # tensor and died with 15 GB free, so it stays at native resolution.
    "pixel_boost": "256x256",
    # gpen_bfr_1024 costs ~2x gpen_bfr_512 per face for detail that a face
    # occupying a few hundred pixels of a VR frame cannot show.
    "enhancer_model": "gpen_bfr_512",
    # Above ~75 the enhancer's own face prior starts overriding the swapped
    # identity, which reads as waxy skin and per-frame flicker in a headset.
    "enhancer_blend": 70,
    # LivePortrait re-applies the ORIGINAL face's expression to the swapped
    # one. hyperswap carries identity well but flattens expression towards the
    # source photo's, so a talking head can end up subtly deadpan and slightly
    # out of sync with its own audio. Off by default: it is a second generative
    # model per face, so it is not cheap.
    "expression_restorer": False,
    # How much of the original expression to put back, 0-100.
    "expression_restorer_factor": 80,
    # 'upper-face' is brows and eyes, 'lower-face' is jaw and mouth. Restoring
    # only the lower face is the usual choice when the source and target have
    # very different eyes.
    "expression_restorer_areas": "upper-face lower-face",
    # Frame enhancers run on the FULL frame, not a 512px face crop. At 4K that
    # is several GB of VRAM per in-flight frame, so it is off unless asked for.
    "frame_enhancer": None,
    "frame_enhancer_blend": 85,
    # Identity strength. FaceFusion maps this to a blend between the source and
    # target ArcFace embeddings: 0.5 is neutral (pure source identity), below
    # 0.5 mixes the *original* face back in, above 0.5 extrapolates away from it
    # and exaggerates the source's identity. Raise it when the swap is clearly
    # the right person but reads as "generic"; lower it if the face stops
    # matching the head it is pasted on.
    "swapper_weight": 0.5,
    # Only models with a 'weight' input honour this (codeformer's fidelity
    # knob). gpen/gfpgan/restoreformer ignore it, so it is inert on the default.
    "enhancer_weight": 0.5,
    "detector_model": "retinaface",
    "detector_size": "640x640",
    # Grows the detection box by a percentage per side before landmarking.
    # A too-tight box clips the jaw/forehead and drags the landmarks with it.
    "detector_margin": "0 0 0 0",
    # 0.25 accepted detections that are not faces; each one then gets a face
    # composited onto it. On the test frame 0.25 returned 8 boxes where 0.5
    # returned 6.
    "detector_score": 0.5,
    "detector_angles": "0",
    # Landmarks decide the affine warp, so a bad set does not just soften the
    # swap, it makes it slide around between frames. 'many' runs 2dfan4 *and*
    # peppa_wutz on every face and keeps whichever scores higher (2dfan4 wins
    # ties by 0.2) - but the second pass is NOT cheap, and on ordinary
    # front-facing footage 2dfan4 wins every frame anyway:
    #
    #   2dfan4            39.0 s / 100 frames
    #   many              72.0 s / 100 frames, byte-identical output
    #
    # So it stays at 2dfan4. Switch to 'many' for footage that actually needs
    # it - hard profiles, heavy head roll, faces looking down - where the extra
    # 85% buys landmarks 2dfan4 cannot find.
    "landmarker_model": "2dfan4",
    "landmarker_score": 0.5,
    # 'one' swaps a single face per frame. 'many' replaces every face in frame,
    # including bystanders and anything the detector got wrong; 'reference'
    # matches a specific face by embedding.
    "selector_mode": "one",
    # Which face 'one' picks: largest in frame by default.
    "selector_order": "large-small",
    # 'auto' | 'male' | 'female'. Restricts which faces are eligible at all.
    "selector_gender": "auto",
    # Only used by selector_mode 'reference': how close a face has to be to the
    # reference embedding to count as the same person, and which frame the
    # reference is taken from.
    "reference_face_distance": 0.3,
    "reference_frame_number": 0,
    # Recovers frames the detector drops. At 0 - the FaceFusion default, and
    # what this pipeline used to run - a frame with no detection is simply left
    # unswapped, so the original face flashes back for one frame. Above 0,
    # faces are matched across a window of +/- target_frame_amount frames by
    # bounding-box overlap and the geometry for a missed frame is interpolated
    # from its neighbours, so the swap survives the gap.
    #
    # Be clear about what this is NOT: it does not smooth landmarks. When the
    # detector succeeds on every frame, tracked output is byte-identical to
    # untracked output - verified on two 100-frame clips, one of them with
    # synthetic head roll, jitter and 4-frame motion blur. It only earns its
    # keep where detection actually fails, and there it earns it outright:
    #
    #   100-frame clip, 2 frames dropped by the detector
    #     tracker off  frames 82-83 unswapped (original face flashes back)
    #     tracker 0.2  frames 82-83 swapped
    #
    # The value is the minimum bounding-box overlap that counts as the same
    # face; FaceFusion accepts 0.0-0.5. Detections are cached per frame, so the
    # extra cost is only decoding the window: measured 39.5 -> 42.3 s per 100
    # frames, +7%. Cheap insurance, so it is on.
    "face_tracker_score": 0.2,
    # Half-width of that window, in frames. 2 = a 5-frame window, so a gap of
    # up to 2 consecutive dropped frames can be bridged. 4 bridges longer gaps
    # and cost 46.0 s against 42.3 s for 2 on the same clip; it changed nothing
    # there, because the gaps were single frames. Raise it only for footage
    # that loses the face for several frames at a time.
    "target_frame_amount": 2,
    "encoder": "hevc_nvenc",
    "quality": 95,
    # FaceFusion maps this onto the encoder's own preset: for NVENC anything
    # from 'ultrafast' to 'fast' collapses to nvenc 'fast', 'medium' to
    # 'medium', and 'slow'/'slower'/'veryslow' to nvenc 'slow'. The default was
    # 'veryfast', i.e. the *lowest* NVENC quality tier. NVENC runs on dedicated
    # silicon, so the slow tier costs GPU time the swapper is not using anyway.
    "output_video_preset": "slow",
    "provider": "cuda",
    # Six, not three: with onnxruntime's thread pools bounded (see
    # scripts/patch_facefusion.py) a pass is Python-serial-bound rather than
    # CPU-bound, and 3 -> 6 threads took a 183-frame 4096x4096 pass from 98.8 s
    # to 89.2 s for 0.9 GB of VRAM and 1.1 GB of commit. The guard clamps it
    # back down when the machine cannot hold it.
    "threads": 6,
    "mask_types": "box occlusion region",
    # Decides what counts as "in front of" the face - a hand, hair, a mic - and
    # is therefore left alone by the swap. 'many' runs all three xseg models
    # and intersects them, which is stricter but slower.
    "occluder_model": "xseg_1",
    # Only consulted when mask_types includes 'region'.
    "parser_model": "bisenet_resnet_34",
    "mask_blur": 0.3,
    # Padding shrinks the box mask inward as a percentage per side, so a
    # non-zero value leaves a rim of the original face around the swap.
    "mask_padding": "0 0 0 0",
    # --- Source face from a video -------------------------------------------
    # Used only when the source is a clip rather than photos. FaceFusion
    # averages the ArcFace embeddings of every source image it is given, so
    # these decide what goes into that average. See extract_source_faces().
    #
    # More stills is not better past a point: the average converges after a
    # handful of genuinely different views, and every extra near-duplicate just
    # re-weights a view already in there. Six is where measured returns stop.
    "source_video_faces": 6,
    # How densely the clip is sampled. Two per second is well below any real
    # rate of change in a face and keeps a 30 s clip to 60 detections.
    "source_video_sample_fps": 2.0,
    # Hard ceiling on decoded frames however long the clip is; the sample rate
    # is lowered to fit rather than the clip being truncated, so a long clip is
    # still covered end to end.
    "source_video_max_samples": 400,
    # Below this the face has fewer pixels than the 112x112 the recogniser
    # wants, and the embedding is mostly upscaling artefacts.
    "source_video_min_face_px": 160,
    # Cosine similarity above which two frames are taken to be the same person.
    # ArcFace puts same-identity pairs around 0.5-0.9 and different-identity
    # pairs near 0, so 0.45 separates them with room on both sides.
    "source_video_identity_threshold": 0.45,
    "fps": None,
    "upscale_method": None,
    "upscale_scale": 2,
    "upscale_target_width": None,
    "temporal_smoothing": False,
    # RIFE frame interpolation. 1 is off; 2 doubles the frame rate. Off by
    # default because it is the most expensive thing in the pipeline by a wide
    # margin - measured 0.96 interpolated frame/s on a 4096x4096 eye, i.e.
    # about 3 hours for both eyes of a 60-second 90 fps VR clip. Worth it for
    # 24-30 fps footage; rarely worth it on source that is already 90.
    "interpolate_multi": 1,
    # Fraction of the frame the model actually runs on; the synthesised frames
    # are upscaled back. 0.75 costs 0.92 dB against full resolution and needs
    # 8.7 GB of VRAM instead of 15.8 on a 16.3 GB card - and going over that
    # ceiling collapses the rate by ~50x. See scripts/rife_worker.py.
    "interpolate_scale": 0.75,
    # 'tolerant' never frees models from VRAM; on a 16 GB card that overflows
    # into system RAM via WDDM and takes the whole desktop with it.
    "video_memory_strategy": "moderate",
    # NOTE: only used by the 'disk' workflow strategy. In 'memory' (below)
    # FaceFusion never writes frames out at all - it decodes, processes and
    # pipes straight to the encoder - so there is no image round-trip to lose
    # quality to, and this key is inert.
    "temp_frame_format": "jpeg",
    # Pinned rather than inherited from facefusion.ini. 'disk' extracts every
    # frame as an image first, which is both hundreds of GB at 4K and an extra
    # lossy generation when temp_frame_format is jpeg.
    "workflow_strategy": "memory",
    # Length, in seconds, of the pieces a selected range is processed in. Each
    # piece goes through the whole pipeline and lands as a playable file in
    # <output>_chunks/ before the next one starts; the pieces are joined at
    # the end. A run that dies part-way keeps every finished piece, and temp
    # usage is one piece's worth rather than the whole clip's. 0 turns it off.
    # See run_chunked_pipeline().
    "chunk_seconds": 60,
}


# FaceFusion decides image-vs-video from the extension of the *output* path, so
# these have to agree with facefusion/choices.py image_type_set.
IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

# Used to spot a *source* that is a clip rather than a photo. FaceFusion's
# --source-paths takes images only, so one of these has to be turned into
# stills first - see extract_source_faces().
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".wmv",
              ".mpg", ".mpeg", ".m2ts", ".ts", ".flv"}


def is_image_path(path):
    return os.path.splitext(str(path))[1].lower() in IMAGE_EXTS


def is_video_path(path):
    return os.path.splitext(str(path))[1].lower() in VIDEO_EXTS


class PipelineError(Exception):
    """
    Anything that stops a run, with a flag for whether it might clear on its own.

    `transient` is False for every raise in this file except a preflight
    refusal that was only about committed memory or free RAM. Those two are
    facts about what the machine happens to be holding this minute - another
    program's 8 GB of commit, a browser's working set - and they are gone by
    themselves an hour later. Everything else (disk, VRAM, the job cap, a
    missing file, a child that died) will read exactly the same way at 04:00,
    so a queued job refused for one of those is a real failure and retrying it
    only burns the night. app._launch_job is what acts on the difference.

    Declared as a class attribute so every existing `raise PipelineError(...)`
    keeps working unchanged and simply means "not transient".
    """
    transient = False


def load_config(config_path=None):
    config = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config.update(json.load(f))
    return config


_GPU_PROBE_CACHE = None

_GPU_PROBE_SRC = """
import sys, json
try:
    import numpy, onnx, onnxruntime as ort
    from onnx import helper, TensorProto
    ort.set_default_logger_severity(3)
    avail = ort.get_available_providers()
    result = {"ort": ort.__version__, "available": avail, "active": None, "error": None}
    if "CUDAExecutionProvider" in avail:
        node = helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
        graph = helper.make_graph(
            [node], "probe",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 16, 16])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 16, 16])],
            [helper.make_tensor("w", TensorProto.FLOAT, [4, 4, 3, 3], [0.01] * 144)],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
        sess.run(None, {"x": numpy.zeros((1, 4, 16, 16), dtype=numpy.float32)})
        result["active"] = sess.get_providers()[0]
    print("VRDEEP_GPU_PROBE " + json.dumps(result))
except Exception as exc:
    print("VRDEEP_GPU_PROBE " + json.dumps({"ort": None, "available": [], "active": None,
                                            "error": type(exc).__name__ + ": " + str(exc)[:400]}))
"""


def _cuda_child_env(guard_config=None):
    """
    Environment for a child process that is going to initialise CUDA.

    Two knobs, both measured on this machine against a 183-frame 4096x4096 pass:

      * CUDA_MODULE_LOADING=LAZY. Without it the CUDA runtime materialises every
        kernel image of every loaded module into the process when the context is
        created, whether the run ever calls them or not; onnxruntime's CUDA
        provider carries a lot of them. Lazy loading defers that to first use and
        took the job's peak commit from 14.5 GB to 13.7 GB, and system VRAM in
        use from 11.2 GB to 8.4 GB, at the same wall time. setdefault rather than
        a plain assignment, so a shell that deliberately asked for EAGER (a
        profiler that wants the symbols up front) is left alone.

      * VRDEEP_ORT_INTRA_THREADS. Read by the inference_manager patch in
        scripts/patch_facefusion.py, which builds every onnxruntime session with
        that many intra-op threads and spinning turned off. FaceFusion's default
        sessions spin one thread per core per model and held 82% of this machine;
        two threads hold 20% at the same frame rate. Only FaceFusion reads it,
        and it is inert in the other workers.
    """
    env = dict(os.environ)
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    env["VRDEEP_ORT_INTRA_THREADS"] = str(
        int((guard_config or {}).get("ort_intra_op_threads", 2) or 2))
    return env


def check_gpu(log_cb=None, force_refresh=False):
    """
    Verify the FaceFusion venv can actually *run* on the GPU.

    onnxruntime-gpu ships no CUDA libraries. If `onnxruntime_providers_cuda.dll`
    cannot resolve cuDNN/cuBLAS, onnxruntime logs one line at startup and then
    quietly executes the whole graph on the CPU - roughly 10x slower for the
    swapper and ~9x for the enhancer. Listing 'CUDAExecutionProvider' in
    get_available_providers() is NOT proof it works, so this actually builds a
    session and runs one convolution on it.

    Returns a dict: {"ok": bool, "ort": str, "active": str, "error": str}.
    """
    global _GPU_PROBE_CACHE
    if _GPU_PROBE_CACHE is not None and not force_refresh:
        return _GPU_PROBE_CACHE

    info = {"ok": False, "ort": None, "active": None, "error": None}
    try:
        result = subprocess.run(
            [PYTHON, "-c", _GPU_PROBE_SRC], cwd=FACEFUSION_DIR,
            capture_output=True, text=True, timeout=180, errors="replace",
            # The probe builds a real CUDA context, so it pays the same module
            # loading cost as a full pass unless it is told not to.
            env=_cuda_child_env(),
            # Without this the probe opens a console window of its own when the
            # GUI itself was started without one.
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        for line in (result.stdout or "").splitlines():
            if line.startswith("VRDEEP_GPU_PROBE "):
                info.update(json.loads(line[len("VRDEEP_GPU_PROBE "):]))
                break
        else:
            info["error"] = (result.stderr or result.stdout or "probe produced no output")[-400:]
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"

    info["ok"] = info.get("active") == "CUDAExecutionProvider"

    if log_cb:
        if info["ok"]:
            log_cb(f"[GPU] CUDA active (onnxruntime {info.get('ort')}).")
        else:
            log_cb("[GPU] CUDA is NOT usable - onnxruntime would fall back to the CPU.")
            if info.get("error"):
                log_cb(f"[GPU] {info['error']}")

    _GPU_PROBE_CACHE = info
    return info


def probe_video(path):
    # duration comes along because the stages that report progress need a
    # frame count to report against, and a second ffprobe per stage to get it
    # is a process spawn for information this call already had to ask for.
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate:format=duration",
        "-of", "json", path
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode == 0 and result.stdout:
        try:
            data = json.loads(result.stdout)
            stream = data["streams"][0]
            w = int(stream["width"])
            h = int(stream["height"])
            r_rate = stream.get("r_frame_rate", "30/1")
            num, den = r_rate.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 30.0
            duration = float(data.get("format", {}).get("duration") or 0.0)
            return {"width": w, "height": h, "fps": fps, "duration": duration}
        except Exception:
            pass
    return {"width": 1920, "height": 1080, "fps": 30.0, "duration": 0.0}


def quality_to_cq(quality):
    """
    Map the GUI's 0-100 Quality onto a -cq / -crf value the same way FaceFusion
    maps --output-video-quality internally (numpy.interp over [51, 0]), so the
    merge does not silently re-encode at a different quality than the swap.
    """
    quality = max(0, min(100, int(quality if quality is not None else 80)))
    return int(round(51 - (quality / 100.0) * 51))


def _decode_args(guard_config=None):
    """
    Decode-thread cap, to be placed immediately before an `-i`.

    ffmpeg defaults to one frame-decoding thread per core, and every one of
    them holds its own decoded frames. That default was written for 1080p; here
    a frame is 4096x4096 (one VR eye, 50 MB raw) or 8192x4096 for the stereo
    source, so the thread count is what decides how many gigabytes a plain
    crop-and-encode holds. Measured on this machine, peak committed memory:

        split  8192x4096 -> one eye   4.58 GB default, 3.25 GB at 4, 2.88 GB at 2
        merge  two eyes  -> 8192x4096 6.86 GB default, 5.33 GB at 4, 4.86 GB at 2

    with the wall-clock time identical in every case (36.3 vs 36.1 s; 24.5 vs
    24.3 s) - the decoders were never the bottleneck, NVENC is. So this is a
    straight 1.3-1.5 GB refund per ffmpeg process.

    Shares `ffmpeg_reader_threads` with FaceFusion's own raw-frame readers
    (see scripts/patch_facefusion.py), because it is the same trade-off.
    """
    threads = int((guard_config or {}).get("ffmpeg_reader_threads", 4) or 4)
    return ["-threads", str(max(1, threads))]


def _encoder_args(encoder, quality_crf, guard_config=None, preset="p6"):
    """
    Build encoder args, preferring NVENC.

    Software x264/x265 at 'slow'/'medium' saturates every logical core at
    normal priority, which is what made the desktop stop responding. NVENC
    runs on a dedicated block of the GPU and costs almost no CPU.

    The -maxrate/-bufsize pair is not a size limit, it is what makes -cq work.
    Under '-rc vbr' with no explicit rate, NVENC falls back to its own default
    maximum bitrate, and that cap - not -cq - is what ends up deciding quality.
    Measured on this card, 3 s of 4K, hevc_nvenc:

        -rc vbr -cq 1 ..20   ->  8.3 MB   (identical output for every value)
        -rc vbr -cq 10 -maxrate 200M -> 30.1 MB, +0.15 dB PSNR

    i.e. every -cq below ~20 - which is the entire range this pipeline uses -
    was collapsing onto the same operating point. Above cq 20 the cap stops
    binding and -cq starts behaving normally again.
    """
    guard_config = guard_config or {}
    if encoder and encoder.endswith("_nvenc"):
        maxrate = int(guard_config.get("nvenc_maxrate_mbps") or 120)
        return ["-c:v", encoder, "-preset", preset, "-rc", "vbr",
                "-cq", str(quality_crf),
                "-maxrate", f"{maxrate}M", "-bufsize", f"{maxrate * 2}M"]

    # Software fallback: cap threads so ffmpeg cannot claim the whole machine.
    # `preset` is an NVENC p-level and means nothing to x264/x265, so pick a
    # software one rather than passing it through.
    cores = os.cpu_count() or 8
    threads = max(2, min(8, cores // 3))
    sw_preset = "veryfast" if encoder == "libx265" else "faster"
    return ["-c:v", encoder or "libx264", "-preset", sw_preset,
            "-crf", str(quality_crf), "-threads", str(threads)]


def split_video(input_path, output_dir, layout="sbs", fps=None, trim_start=None, trim_end=None,
                log_cb=None, guard=None, encoder="h264_nvenc", guard_config=None, skip=(),
                progress_cb=None, start_pct=0, end_pct=0, lane=None):
    """
    Crop a stereo pair into two per-eye files.

    `skip` names eyes ("left"/"right") the caller has already accounted for -
    either their split survived an earlier attempt, or their swap did and the
    split is no longer needed at all. Decoding 8K source twice is minutes of
    work, so this is not a micro-optimisation.
    """
    guard = guard or NULL_GUARD
    skip = set(skip)
    if log_cb:
        log_cb(f"Splitting {layout.upper()} video: {os.path.basename(input_path)}")

    os.makedirs(output_dir, exist_ok=True)
    left_out = os.path.join(output_dir, "left.mp4")
    right_out = os.path.join(output_dir, "right.mp4")

    probe = probe_video(input_path)

    if layout == "sbs":
        half_w = probe["width"] // 2
        fl = f"crop={half_w}:{probe['height']}:0:0"
        fr = f"crop={half_w}:{probe['height']}:{half_w}:0"
    else:
        half_h = probe["height"] // 2
        fl = f"crop={probe['width']}:{half_h}:0:0"
        fr = f"crop={probe['width']}:{half_h}:0:{half_h}"

    # crf 6 + preset slow produced near-lossless intermediates that ran the
    # temp drive out of space. This is the *input* to the swap, so anything it
    # throws away is gone for good - cq 16 with the NVENC rate cap lifted (see
    # _encoder_args) is visually transparent and still an order of magnitude
    # smaller than the near-lossless version. The cap keeps the disk estimate
    # bounded.
    enc_args = _encoder_args(encoder, 16, guard_config, preset="p6")

    # How many frames each eye will write, so the split can report progress
    # rather than being a silent hour on a long job.
    duration = probe.get("duration") or 0.0
    span_start = max(0.0, trim_start or 0.0)
    span_end = duration if trim_end is None else min(duration, trim_end)
    eye_frames = int(max(0.0, span_end - span_start) * (fps or probe["fps"]))

    todo = [eye for eye in ("left", "right") if eye not in skip]
    slot = 0

    for eye, label, filt, out in [("left", "Left eye", fl, left_out),
                                  ("right", "Right eye", fr, right_out)]:
        guard.check()
        if eye in skip:
            if log_cb:
                log_cb(f"  {label}: already split by an earlier attempt, skipping.")
            continue
        if log_cb:
            log_cb(f"  Processing {label}...")

        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        # Either end may be left open - a start with no end runs to the end of
        # the file, an end with no start begins at zero. Both are input options
        # so -to stays absolute on the source timeline, which is what the two
        # boxes in the GUI mean.
        if trim_start is not None:
            cmd += ["-ss", str(trim_start)]
        if trim_end is not None:
            cmd += ["-to", str(trim_end)]
        cmd += _decode_args(guard_config) + ["-i", input_path, "-vf", filt]
        if fps:
            cmd += ["-r", str(fps)]
        cmd += enc_args + ["-c:a", "aac", "-b:a", "256k", out]

        on_frame = None
        if progress_cb and end_pct > start_pct and todo:
            width = (end_pct - start_pct) / float(len(todo))
            base = start_pct + slot * width

            def on_frame(done, total, _base=base, _width=width, _label=label):
                progress_cb(int(_base + min(1.0, done / float(total)) * _width),
                            f"Splitting {_label.lower()}: frame {done}/{total}")

        code, err = _run_ffmpeg_watched(cmd, guard, on_frame=on_frame,
                                        total_frames=eye_frames, lane=lane)
        if code != 0:
            guard.check()
            raise PipelineError(f"Split failed for {label}: {_ffmpeg_error_excerpt(err)}")
        slot += 1

    return left_out, right_out


SOURCE_FACES_DIR = os.path.join(PROJECT_DIR, "output", "source_faces")

# Everything that changes what comes out of the worker. Anything not in here
# does not invalidate a cached extraction.
SOURCE_FACE_KEYS = (
    "source_video_faces", "source_video_sample_fps", "source_video_max_samples",
    "source_video_min_face_px", "source_video_identity_threshold",
    "detector_model", "detector_size", "detector_score", "detector_angles",
    "landmarker_model", "landmarker_score", "selector_gender",
)


def split_sources(source):
    """
    Normalise however the caller expressed the source into a list of paths.

    The GUI joins multiple picks with '; ', the CLI may pass a comma-separated
    string or a real list, and a single path arrives bare.
    """
    if isinstance(source, (list, tuple)):
        return [str(p).strip() for p in source if str(p).strip()]
    source = str(source)
    if ";" in source:
        return [p.strip() for p in source.split(";") if p.strip()]
    if "," in source:
        return [p.strip() for p in source.split(",") if p.strip()]
    return [source] if source.strip() else []


def _source_cache_dir(video_path, config):
    """
    A directory that is unique to this clip *and* these settings.

    Keyed on the file's identity rather than its contents - hashing a 200 MB
    clip to save re-running a 20 s extraction is the wrong trade - so a clip
    edited in place under the same name and timestamp would be missed. That is
    what the Re-extract button is for.
    """
    stat = os.stat(video_path)
    parts = [os.path.abspath(video_path), str(stat.st_mtime_ns), str(stat.st_size)]
    parts += [f"{key}={config.get(key)!r}" for key in SOURCE_FACE_KEYS]
    digest = hashlib.sha1("|".join(parts).encode("utf-8", "replace")).hexdigest()[:10]
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(os.path.basename(video_path))[0])[:40]
    return os.path.join(SOURCE_FACES_DIR, f"{stem}_{digest}")


def _read_cached_faces(cache_dir):
    """Return the cached still list, or None if the cache is absent or stale."""
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path) as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return None
    paths = [face["path"] for face in manifest.get("faces", [])]
    if not paths or not all(os.path.exists(p) for p in paths):
        return None
    return paths, manifest


def _log_manifest(manifest, log_cb):
    if not log_cb:
        return
    faces = manifest.get("faces", [])
    log_cb(f"  Sampled {manifest.get('sampled')} frame(s) at "
           f"{manifest.get('sample_fps')} fps; {manifest.get('candidates')} had a usable face.")

    rejected = [
        (manifest.get("rejected_no_face"), "no face detected"),
        (manifest.get("rejected_too_small"), "face too small"),
        (manifest.get("rejected_blurred"), "too blurred"),
        (manifest.get("rejected_exposure"), "too dark or blown out"),
        (manifest.get("rejected_pose"), "turned too far away"),
        (manifest.get("rejected_gender"), "wrong gender"),
        (manifest.get("rejected_other_identity"), "a different person"),
    ]
    dropped = ", ".join(f"{count} {why}" for count, why in rejected if count)
    if dropped:
        log_cb(f"  Rejected: {dropped}.")
    if manifest.get("rejected_other_identity"):
        log_cb("  NOTE: more than one person appears in this clip. Only the most "
               "frequent face was kept - averaging two identities produces neither.")

    log_cb(f"  Kept {len(faces)} still(s):")
    for index, face in enumerate(faces, start=1):
        log_cb(f"    {index}. t={face['time']}s  quality={face['quality']}  "
               f"face={face['size_px']}px  yaw={face['yaw']}")


def extract_source_faces(video_path, config, log_cb=None, progress_cb=None, guard=None,
                         trim_start=None, trim_end=None, refresh=False):
    """
    Turn a short source *video* into the stills FaceFusion needs.

    FaceFusion accepts only images in --source-paths. It reads each one, takes
    the largest face, and averages the ArcFace embeddings across all of them
    (face_creator.average_face_identity), which is why several stills beat a
    single photo: the mean of a few genuinely different views of a face is a
    better-conditioned identity than any one view.

    So a clip is not a new capability so much as a better way of filling the
    input FaceFusion already has - provided the wrong frames are thrown away
    first. Frames that are blurred, badly exposed, too small or turned away
    carry no identity, and averaging them in moves the mean away from the
    person; a second face appearing even briefly is worse still. That selection
    is the whole job, and it happens in face_source_worker.py.

    Results are cached under output/source_faces/ and keyed on the clip plus
    the settings that change the outcome, because the same source is normally
    reused across many runs - and in VR mode run_facefusion is called twice per
    run, once per eye, off the same source.
    """
    guard = guard or NULL_GUARD
    guard.check()

    if not os.path.exists(video_path):
        raise PipelineError(f"Source video not found: {video_path}")

    cache_dir = _source_cache_dir(video_path, config)
    if not refresh:
        cached = _read_cached_faces(cache_dir)
        if cached:
            paths, manifest = cached
            if log_cb:
                log_cb(f"[Source] Reusing {len(paths)} face still(s) already extracted from "
                       f"{os.path.basename(video_path)}.")
                log_cb(f"  Cache: {cache_dir}  (delete it, or tick Re-extract, to redo)")
            return paths, manifest

    # A stale or half-written extraction would otherwise be picked up as if it
    # were complete.
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)

    count = int(config.get("source_video_faces") or 6)
    if log_cb:
        log_cb(f"[Source] Extracting up to {count} face still(s) from "
               f"{os.path.basename(video_path)}...")

    cmd = [
        PYTHON, os.path.join(SCRIPTS_DIR, "face_source_worker.py"),
        "--video", video_path,
        "--out-dir", cache_dir,
        "--count", str(count),
        "--sample-fps", str(float(config.get("source_video_sample_fps") or 2.0)),
        "--max-samples", str(int(config.get("source_video_max_samples") or 400)),
        "--min-face-px", str(int(config.get("source_video_min_face_px") or 160)),
        "--identity-threshold", str(float(config.get("source_video_identity_threshold") or 0.45)),
        "--provider", str(config.get("provider") or "cuda"),
        "--detector-model", str(config.get("detector_model") or "retinaface"),
        "--detector-size", str(config.get("detector_size") or "640x640"),
        "--detector-score", str(config.get("detector_score", 0.5)),
        "--detector-angles", str(config.get("detector_angles") or "0"),
        "--landmarker-model", str(config.get("landmarker_model") or "2dfan4"),
        "--landmarker-score", str(config.get("landmarker_score", 0.5)),
        # 'auto' would have the worker keep whichever gender the classifier
        # happens to return per frame, which is not a filter at all - it is the
        # same default FaceFusion uses for the swap itself.
        "--gender", str(config.get("selector_gender") or "auto"),
        # Same knob as every other decode in the pipeline; read off the guard
        # rather than threading guard_config through two more signatures.
        "--decode-threads",
        str(int((getattr(guard, "config", None) or {}).get("ffmpeg_reader_threads", 4) or 4)),
    ]
    if trim_start:
        cmd += ["--trim-start", str(float(trim_start))]
    if trim_end:
        cmd += ["--trim-end", str(float(trim_end))]

    proc = guard.spawn(cmd, cwd=FACEFUSION_DIR, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")

    manifest = None
    error_message = None
    tail = []
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        if line.startswith("FACESRC_ERROR "):
            error_message = line[len("FACESRC_ERROR "):]
            continue
        if line.startswith("FACESRC_RESULT "):
            try:
                manifest = json.loads(line[len("FACESRC_RESULT "):])
            except ValueError:
                error_message = "The worker returned a result that could not be parsed."
            continue
        if line.startswith("FACESRC_PROGRESS "):
            try:
                done, total = (int(x) for x in line.split()[1:3])
            except ValueError:
                continue
            if progress_cb and total:
                progress_cb(int(done / total * 100),
                            f"Analysing source video, frame {done}/{total}")
            continue
        tail.append(line)
        del tail[:-20]

    proc.wait()
    guard.check()

    if proc.returncode != 0 or not manifest or not manifest.get("faces"):
        shutil.rmtree(cache_dir, ignore_errors=True)
        detail = error_message or ("\n".join(tail) if tail else "no output")
        raise PipelineError(
            f"Could not build source faces from {os.path.basename(video_path)}.\n\n{detail}"
        )

    _log_manifest(manifest, log_cb)
    return [face["path"] for face in manifest["faces"]], manifest


def resolve_source_faces(source, config, log_cb=None, progress_cb=None, guard=None,
                         trim_start=None, trim_end=None, refresh=False):
    """
    Expand a source specification into the image paths FaceFusion will be given.

    Photos pass straight through; any video entry is replaced by the stills
    extracted from it. Mixing the two is allowed and is often the best input a
    clip can be part of - a studio portrait plus a handful of frames from a
    clip covers both detail and pose.

    Called once per run rather than per FaceFusion invocation, because VR mode
    calls run_facefusion twice - once per eye - and both eyes must be swapped
    against exactly the same identity or they will not agree in the headset.
    """
    entries = split_sources(source)
    if not entries:
        raise PipelineError("No source face was given.")

    resolved = []
    for entry in entries:
        if not os.path.exists(entry):
            raise PipelineError(f"Source not found: {entry}")
        if is_video_path(entry):
            paths, _ = extract_source_faces(
                entry, config, log_cb=log_cb, progress_cb=progress_cb, guard=guard,
                trim_start=trim_start, trim_end=trim_end, refresh=refresh,
            )
            resolved.extend(paths)
        elif is_image_path(entry):
            resolved.append(entry)
        else:
            raise PipelineError(
                f"{os.path.basename(entry)} is neither an image nor a video this "
                f"pipeline recognises.\n\nImages: {', '.join(sorted(IMAGE_EXTS))}\n"
                f"Videos: {', '.join(sorted(VIDEO_EXTS))}"
            )

    return resolved


def run_facefusion(source_path, target_path, output_path, temp_path, config, log_cb=None,
                   progress_cb=None, start_pct=10, end_pct=90, stage_name="FaceFusion",
                   guard=None, guard_config=None, lane=None):
    """
    One FaceFusion pass, start to finish.

    `lane` names the part of the run this pass belongs to, and is only used
    when two of them are in flight at once (the both-eyes-at-once path in
    run_vr_pipeline). It goes to guard.spawn, which is what lets the watchdog
    freeze this pass - and only this pass - when the machine runs short of
    memory, instead of freezing the eye that is nearly finished as well. A pass
    with no lane behaves exactly as it always has.
    """
    guard = guard or NULL_GUARD
    guard.check()

    # This function is called by the pipelines, which have a guard config in
    # hand, and directly by scripts that do not. Prefer what the caller passed,
    # fall back to the config the guard is already running under, and only read
    # the file when there is neither - it is one read per pass either way.
    guard_config = (guard_config if guard_config is not None
                    else getattr(guard, "config", None) or guardrails.load_guard_config())

    src_list = split_sources(source_path)

    # The pipelines resolve videos to stills before they get here, but this
    # function is also called directly (run_facefusion.py, ad-hoc scripts), and
    # FaceFusion's failure for a video source is an unhelpful one - it filters
    # non-images out of --source-paths silently and then swaps nothing.
    videos = [p for p in src_list if is_video_path(p)]
    if videos:
        raise PipelineError(
            f"{os.path.basename(videos[0])} is a video, and FaceFusion's --source-paths "
            f"takes images only.\n\nCall engine.resolve_source_faces(source, config) first "
            f"- it extracts the best stills from a clip and returns the paths to pass here. "
            f"run_vr_pipeline() and run_normal_pipeline() already do this."
        )

    if log_cb:
        src_names = ", ".join([os.path.basename(p) for p in src_list])
        log_cb(f"[{stage_name}] Starting processing stage...")
        log_cb(f"  Source Face(s): {src_names} ({len(src_list)} photo(s))")
        log_cb(f"  Target Video: {os.path.basename(target_path)}")
        log_cb(f"  Swapper Model: {config['swapper_model']}")
        log_cb(f"  Enhancer Model: {config['enhancer_model']} (blend={config['enhancer_blend']}%)")
        log_cb(f"  Identity Weight: {config.get('swapper_weight', 0.5)} | "
               f"Landmarker: {config.get('landmarker_model', '2dfan4')}")
        log_cb(f"  Face Tracker: {config.get('face_tracker_score', 0.2)} "
               f"(window +/-{int(config.get('target_frame_amount', 2))} frames)")
        log_cb(f"  Encoder: {config['encoder']} | Provider: {config['provider']}")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    os.makedirs(temp_path, exist_ok=True)

    # Build processors list
    # Order is execution order. The expression has to be put back onto the
    # swapped face, so the restorer runs after the swapper; the enhancer runs
    # last so it is cleaning up the final geometry rather than something the
    # restorer is about to move.
    processors = ["face_swapper"]
    if config.get("expression_restorer"):
        processors.append("expression_restorer")
    if config["enhancer_model"]:
        processors.append("face_enhancer")
    if config.get("frame_enhancer"):
        processors.append("frame_enhancer")

    cmd = [
        PYTHON, "facefusion.py", "headless-run",
        "--source-paths", *src_list,
        "--target-path", target_path,
        "--output-path", output_path,
        "--temp-path", temp_path,
        # Its own job store, inside its own temp directory. FaceFusion's
        # headless-run names the job after the clock to the second
        # (job_helper.suggest_job_id) and refuses to create one that already
        # exists - and process_headless() returns 1 for that WITHOUT logging
        # anything, so it surfaces as "FaceFusion failed: Unknown error" with
        # no output at all. Two lanes started in the same second (B10) hit that
        # every time, and it cost the second eye its whole pass. Measured on
        # Sep 10: both eyes launched at 12:48:22, one ran, the other died
        # instantly. The default is a shared '.jobs' in the FaceFusion
        # directory; per pass, they cannot collide.
        "--jobs-path", os.path.join(temp_path, "jobs"),
        "--face-detector-model", config["detector_model"],
        "--face-detector-size", config["detector_size"],
        "--face-detector-score", str(config["detector_score"]),
        # Every extra angle is another pass over the frame *and* another chance
        # to return a rotated false positive, which then gets a face pasted onto
        # it. Upright footage only needs 0; add 90/180/270 for footage where the
        # camera itself rolls.
        "--face-detector-angles", *str(config.get("detector_angles") or "0").split(),
        "--face-detector-margin", *str(config.get("detector_margin") or "0 0 0 0").split(),
        "--face-landmarker-model", config.get("landmarker_model", "2dfan4"),
        "--face-landmarker-score", str(config.get("landmarker_score", 0.5)),
        "--face-swapper-model", config["swapper_model"],
        "--face-swapper-pixel-boost", config.get("pixel_boost", "512x512"),
        # 0.5 is neutral. Higher pushes the result further from the original
        # face's embedding, i.e. stronger resemblance to the source photo.
        "--face-swapper-weight", str(config.get("swapper_weight", 0.5)),
        # FaceFusion sorts by order first, then filters by gender/race/age, then
        # (in 'one' mode) takes the first survivor. So 'one' + 'large-small' +
        # gender 'female' means "the largest female face in the frame".
        "--face-selector-mode", config.get("selector_mode", "one"),
        "--face-selector-order", config.get("selector_order", "large-small"),
        # 'auto' copies the gender of the source photo, so a female source only
        # ever replaces female faces. 'male'/'female' pin it explicitly, which is
        # what you want when the source is ambiguous or misclassified.
        "--face-selector-gender", config.get("selector_gender", "auto"),
        # Without this, a frame the detector loses is left unswapped and the
        # original face flashes back for one frame. With it, the geometry for
        # that frame is interpolated from the neighbours inside a
        # +/- target_frame_amount window.
        "--face-tracker-score", str(config.get("face_tracker_score", 0.2)),
        "--target-frame-amount", str(int(config.get("target_frame_amount", 2))),
        "--face-occluder-model", config.get("occluder_model", "xseg_1"),
        "--face-parser-model", config.get("parser_model", "bisenet_resnet_34"),
        "--face-mask-types", *config["mask_types"].split(),
        "--face-mask-blur", str(config["mask_blur"]),
        "--face-mask-padding", *config["mask_padding"].split(),
        "--output-video-encoder", config["encoder"],
        "--output-video-quality", str(config["quality"]),
        # Default is 'veryfast', which FaceFusion maps to the lowest NVENC
        # quality tier. 'slow' maps to the highest one and costs GPU blocks the
        # inference is not using.
        "--output-video-preset", config.get("output_video_preset", "slow"),
        # 'cpu' is deliberately NOT appended as a fallback. onnxruntime drops
        # to the CPU provider *silently* when the CUDA provider fails to load,
        # which turns a 20-minute job into a 12-hour one with nothing in the log
        # to say why. check_gpu() verifies CUDA for real before we get here.
        "--execution-providers", *config["provider"].split(),
        "--execution-thread-count", str(int(config.get("threads") or 3)),
        "--video-memory-strategy", config.get("video_memory_strategy", "moderate"),
        "--temp-frame-format", config.get("temp_frame_format", "jpeg"),
        # 'memory' decodes, processes and pipes straight to the encoder. The
        # 'disk' path writes every frame out as an image first, which is an
        # extra lossy generation on top of hundreds of GB of temp files. Pinned
        # here so an edited facefusion.ini cannot quietly switch it.
        "--workflow-strategy", config.get("workflow_strategy", "memory"),
        "--log-level", "info",
    ]

    if processors:
        cmd.extend(["--processors"] + processors)

    if config["enhancer_model"]:
        cmd.extend(["--face-enhancer-model", config["enhancer_model"]])
        cmd.extend(["--face-enhancer-blend", str(config["enhancer_blend"])])
        # Ignored by any enhancer whose ONNX graph has no 'weight' input, which
        # is all of them except codeformer.
        cmd.extend(["--face-enhancer-weight", str(config.get("enhancer_weight", 0.5))])

    if config.get("expression_restorer"):
        cmd.extend(["--expression-restorer-model", "live_portrait"])
        cmd.extend(["--expression-restorer-factor",
                    str(int(config.get("expression_restorer_factor", 80)))])
        cmd.extend(["--expression-restorer-areas"]
                   + str(config.get("expression_restorer_areas")
                         or "upper-face lower-face").split())

    if config.get("selector_mode") == "reference":
        cmd.extend(["--reference-face-distance", str(config.get("reference_face_distance", 0.3))])
        cmd.extend(["--reference-frame-number", str(int(config.get("reference_frame_number", 0)))])

    if config.get("frame_enhancer"):
        cmd.extend(["--frame-enhancer-model", config["frame_enhancer"]])
        cmd.extend(["--frame-enhancer-blend", str(config.get("frame_enhancer_blend", 80))])

    if config.get("fps"):
        cmd.extend(["--output-video-fps", str(config["fps"])])

    # For a still target the video encoder settings above are inert and the
    # quality knob is a different flag entirely.
    if is_image_path(output_path):
        cmd.extend(["--output-image-quality", str(config["quality"])])

    proc = guard.spawn(
        cmd, lane=lane, cwd=FACEFUSION_DIR, env=_cuda_child_env(guard_config),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, errors="replace",
    )

    tqdm_regex = re.compile(r'(processing|analysing):\s*(\d+)%.*?(\d+)/(\d+)\s*\[.*?(?:,\s*([\d\.]+)frame/s)?')
    last_logged_milestone = -1
    output_lines = []

    # tqdm rewrites its progress bar with \r, so read in chunks and split on
    # both terminators. The old version read one character at a time and
    # spun on empty reads, which burned a full core for the entire run.
    def iter_lines(stream):
        buffer = ""
        while True:
            chunk = stream.read(256)
            if not chunk:
                if buffer.strip():
                    yield buffer.strip()
                return
            buffer += chunk
            while True:
                index = min(
                    (buffer.find(c) for c in ("\r", "\n") if c in buffer),
                    default=-1,
                )
                if index < 0:
                    break
                line, buffer = buffer[:index], buffer[index + 1:]
                if line.strip():
                    yield line.strip()

    for line in iter_lines(proc.stdout):
        if guard.abort_reason:
            break

        output_lines.append(line)
        # Keep only a tail; a long run emits hundreds of thousands of lines and
        # the old code held every one of them in memory until the run finished.
        if len(output_lines) > 400:
            del output_lines[:200]

        match = tqdm_regex.search(line)
        if match:
            action = match.group(1)
            frame_pct = int(match.group(2))
            curr_frame = int(match.group(3))
            total_frames = int(match.group(4))
            fps = float(match.group(5)) if match.group(5) else 0.0

            if action == "processing":
                scaled_pct = int(start_pct + (frame_pct / 100.0) * (end_pct - start_pct))
                fps_str = f" @ {fps:.1f} fps" if fps > 0 else ""
                stage_str = f"{stage_name}: Frame {curr_frame}/{total_frames} ({frame_pct}%){fps_str}"

                if progress_cb:
                    progress_cb(scaled_pct, stage_str)

                milestone = (frame_pct // 10) * 10
                if milestone > last_logged_milestone:
                    last_logged_milestone = milestone
                    if log_cb:
                        rem_frames = total_frames - curr_frame
                        eta_sec = int(rem_frames / fps) if fps > 0 else 0
                        eta_str = f" (ETA: ~{eta_sec//60}m {eta_sec%60}s)" if eta_sec > 0 else ""
                        log_cb(f"  [{stage_name}] Progress: {frame_pct}% ({curr_frame}/{total_frames} frames){fps_str}{eta_str}")

            elif action == "analysing":
                if progress_cb:
                    progress_cb(start_pct, f"{stage_name}: Analyzing frames ({curr_frame}/{total_frames})...")
        else:
            if any(tag in line for tag in ["[FACEFUSION", "error", "warning", "downloading", "extracting", "merging"]):
                if log_cb and not ("frame/s" in line or "%|" in line):
                    log_cb(f"  [FaceFusion] {line}")

    proc.wait()
    guard.check()

    # A lane the watchdog gave up on was killed by us, not by FaceFusion. Its
    # exit code is whatever TerminateProcess left behind, and reporting that as
    # "FaceFusion failed" would send the user looking for a bug in the swap.
    lane_reason = (getattr(guard, "lane_failed", {}) or {}).get(lane) if lane else None
    if lane_reason:
        error = PipelineError(
            f"The {lane} lane was stopped by the guard rails: {lane_reason}.\n\n"
            f"This is not a failure of the swap itself - the machine could not hold "
            f"two lanes at these settings, so this eye is re-run on its own."
        )
        error.lane = lane
        raise error

    if proc.returncode != 0:
        clean_err_lines = [
            l for l in output_lines
            if l.strip() and not ("frame/s" in l or "%|" in l or "analysing:" in l or "processing:" in l)
        ]
        err_msg = "\n".join(clean_err_lines[-10:]) if clean_err_lines else "Unknown error"

        # Once the job object's committed-memory cap is reached, every
        # allocation inside it fails and what comes back is an onnxruntime or
        # numpy memory error with nothing in it that points at the guard rails.
        # Say what actually happened instead.
        blob = "\n".join(clean_err_lines).lower()
        alloc_failure = any(
            tag in blob for tag in
            ("memoryerror", "bad allocation", "bad_alloc", "out of memory",
             "cannot allocate", "allocation failed", "alloc_failed")
        )
        if alloc_failure or guard.hit_memory_cap():
            cap_gb = guard.memory_cap_bytes() / (1024 ** 3)
            peak_gb = guard.peak_memory_bytes() / (1024 ** 3)

            if guard.hit_memory_cap():
                raise PipelineError(
                    "FaceFusion ran out of memory.\n\n"
                    f"The guard rails cap the whole pipeline at {cap_gb:.1f} GB of committed "
                    f"memory and it peaked at {peak_gb:.1f} GB. That cap is on *committed* "
                    "memory, not RAM in use - a CUDA process reserves several times what it "
                    "makes resident, so this happens with plenty of free RAM.\n\n"
                    "Fix, in order of preference:\n"
                    "  - use a lighter quality preset: 'Best' costs about 6 GB more than "
                    "'Good', nearly all of it Pixel Boost 512 plus gpen_bfr_1024\n"
                    "  - lower Threads in the Settings tab\n"
                    "  - raise job_memory_cap_gb in scripts/guardrails.json - but only as far "
                    "as this machine can actually back it: the commit limit is "
                    f"{guardrails.fmt_gb(guardrails.commit_limit_bytes())} in total, shared "
                    "with everything else running.\n\n"
                    f"FaceFusion said:\n{err_msg}"
                )

            # The cap was NOT reached, so blaming the guard rails would be a
            # lie - and the fix that message implies ("raise the cap") is the
            # opposite of what helps. What ran out is the machine's own commit
            # limit, which the job cap knows nothing about: the cap counts what
            # the children commit, while the GPU driver's system-memory
            # reservations and every other process on the box come out of the
            # same pool. When that pool empties the failure lands on whichever
            # allocation asked next, anywhere on the machine, with the job
            # sitting at a fraction of its cap - which is exactly what a 48 MB
            # frame failing at 16.9 GB of a 51 GB cap looks like.
            # guardrails already knows where the pagefile is and how big it
            # is, and "enlarge the pagefile" is much easier to act on when it
            # comes with the number it is starting from.
            try:
                pagefile_mb = sum(pf["allocated_mb"] for pf in guardrails.pagefiles())
            except Exception:
                pagefile_mb = 0
            pagefile_note = (
                f" The pagefile is {pagefile_mb / 1024:.0f} GB of that today, and it is the "
                f"half that can be grown."
                if pagefile_mb else ""
            )

            low_commit = guard.min_available_commit_bytes()
            low_note = (
                f", of which only {guardrails.fmt_gb(low_commit)} was still free at the "
                f"lowest point"
                if low_commit else ""
            )
            raise PipelineError(
                "FaceFusion ran out of memory - but not because of the guard rails.\n\n"
                f"The pipeline peaked at {peak_gb:.1f} GB of committed memory against its "
                f"{cap_gb:.1f} GB cap, so the cap was never the ceiling, and a lighter "
                "preset or fewer threads is not the fix here.\n\n"
                "What ran out is this machine's *system* commit limit - physical RAM plus "
                "the pagefile, "
                f"{guardrails.fmt_gb(guardrails.commit_limit_bytes())} in total, shared with "
                f"every other program running{low_note}.{pagefile_note} Once that is gone "
                "every allocation "
                "on the machine fails, and the one that lost here happened to be a video "
                "frame.\n\n"
                "Raising job_memory_cap_gb will not help: the cap is already higher than the "
                "system can hand out.\n\n"
                "Fix, in order of preference:\n"
                "  - enlarge the pagefile - the commit limit is made of it, and this is the "
                "only fix here that adds headroom rather than using less\n"
                "  - close other programs before starting a long run\n"
                "  - use a lighter quality preset: 'Best' costs about 6 GB more than "
                "'Good', nearly all of it Pixel Boost 512 plus gpen_bfr_1024\n"
                "  - lower Threads in the Settings tab\n\n"
                f"FaceFusion said:\n{err_msg}"
            )

        raise PipelineError(f"FaceFusion failed:\n{err_msg}")

    if log_cb:
        log_cb(f"[{stage_name}] Processing completed successfully!")

    return output_path


# Measured on this machine (RTX 5060 Ti, driver 595.97) by encoding one black
# frame at each size rather than looked up:
#
#   h264_nvenc   4096x4096 OK    4098x2160 FAIL   4096x4098 FAIL   5120x2880 FAIL
#   hevc_nvenc   4096x4096 OK    8192x4096 OK     8192x8192 OK     5120x2880 OK
#   av1_nvenc    4096x4096 OK    8192x4096 OK     8192x8192 OK
#
# So the cap is per dimension rather than on total pixels, and H.264 is half of
# HEVC - which matters because h264_nvenc used to be this function's default.
# Software encoders have no practical limit and are deliberately absent.
ENCODER_MAX_DIMENSION = {
    "h264_nvenc": 4096,
    "hevc_nvenc": 8192,
    "av1_nvenc": 8192,
}


def upscale_output_size(probe, scale=2, target_width=None):
    """
    The size an upscale will actually produce.

    Shared by the preflight and by the upscale itself, because a check that
    validates a different number from the one that reaches the encoder is
    worse than no check at all.
    """
    if target_width:
        target_width = int(target_width)
        aspect = probe["height"] / probe["width"]
        width, height = target_width, int(round(target_width * aspect))
    else:
        width, height = probe["width"] * scale, probe["height"] * scale
    # Both NVENC and the yuv420p chroma subsampling need even dimensions.
    return (int(width) + 1) // 2 * 2, (int(height) + 1) // 2 * 2


def check_upscale_encoder(width, height, encoder):
    """
    Refuse an upscale the encoder cannot perform, while it is still cheap.

    Left unchecked this surfaces from inside ffmpeg as

        [hevc_nvenc] No capable devices found
        Error while opening encoder - maybe incorrect parameters such as
        bit_rate, rate, width or height

    which names neither the real cause nor any way out - and in the pipeline it
    surfaces *after* the swap, half an hour into the run.
    """
    limit = ENCODER_MAX_DIMENSION.get(encoder)
    if not limit or (width <= limit and height <= limit):
        return

    ways_out = [f"lower the upscale target so neither side exceeds {limit}"]
    if limit < 8192:
        ways_out.append("switch the encoder to hevc_nvenc or av1_nvenc, which reach 8192")
    ways_out.append(
        "set the encoder to libx265, which encodes this in software with no "
        "practical size limit - but it is far slower, and anything wider than "
        "8192 will not hardware-decode afterwards, here or in a headset"
    )
    raise PipelineError(
        f"{encoder} cannot encode {width}x{height}.\n\n"
        f"NVENC on this GPU stops at {limit} pixels in either dimension, "
        f"measured. Ways out:\n"
        + "".join(f"  - {w}.\n" for w in ways_out)
    )


def upscale_video_ffmpeg(input_path, output_path, scale=2, target_width=None, encoder="hevc_nvenc",
                         crf=20, log_cb=None, guard=None, guard_config=None):
    """Option B: FFmpeg Lanczos upscale to target resolution."""
    guard = guard or NULL_GUARD
    guard.check()
    if log_cb:
        log_cb(f"Upscaling via FFmpeg Lanczos (scale={scale}x)")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    probe = probe_video(input_path)
    new_w, new_h = upscale_output_size(probe, scale, target_width)
    check_upscale_encoder(new_w, new_h, encoder)
    if log_cb:
        log_cb(f"  {probe['width']}x{probe['height']} -> {new_w}x{new_h}")

    cmd = (
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        + _decode_args(guard_config)
        + ["-i", input_path, "-vf", f"scale={new_w}:{new_h}:flags=lanczos"]
        + _encoder_args(encoder, crf, guard_config, preset="p7")
        + ["-c:a", "copy", output_path]
    )
    result = guard.run(cmd, capture_output=True)
    if result.returncode != 0:
        guard.check()
        raise PipelineError(f"FFmpeg upscale failed: {(result.stderr or b'').decode(errors='replace')[:400]}")

    return output_path


def upscale_video_realesrgan(input_path, output_path, scale=2, target_width=None, encoder="hevc_nvenc",
                             crf=20, log_cb=None, guard=None, tile=512, progress_cb=None,
                             guard_config=None):
    """
    Option A: Real-ESRGAN frame-by-frame upscale, run in a separate process.

    The model used to be loaded here, in the caller's process - which for the
    GUI meant torch, a CUDA context and torch's caching allocator all stayed
    resident for the rest of the session, because Python cannot unload a native
    extension and torch does not return freed device blocks to the driver on
    its own. Measured: `import torch` alone took the process from 39 MB RSS to
    369 MB, and none of it came back.

    It was also the only stage running outside the guard's Job Object, so it
    had no memory cap, no CPU cap, and the watchdog and Stop button could not
    reach it.

    Both are fixed by spawning scripts/esrgan_worker.py through the guard: it
    inherits the job object, and everything it allocates - host and device - is
    returned by the OS when it exits. See esrgan_worker.py.
    """
    guard = guard or NULL_GUARD
    guard.check()

    model_path = os.path.join(FACEFUSION_DIR, "weights", "realesr-general-x4v3.pth")
    if not os.path.exists(model_path):
        # Checked here rather than in the worker so the run dies before the
        # merge output has been consumed, and with something readable.
        raise PipelineError(
            f"Real-ESRGAN is selected but its weights are not installed.\n\n"
            f"Expected: {model_path}\n\n"
            f"Either drop realesr-general-x4v3.pth there, or pick a different "
            f"Output Upscaling method ('ffmpeg' for a plain Lanczos resize), or "
            f"use FaceFusion's own frame enhancer instead - set Frame Enhancer to "
            f"real_esrgan_x2_fp16, which runs on the onnxruntime GPU stack that is "
            f"already working and needs no extra download."
        )

    if log_cb:
        log_cb(f"Upscaling via Real-ESRGAN (scale={scale}x, tile={tile}) in a worker process")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    cmd = [
        PYTHON, os.path.join(SCRIPTS_DIR, "esrgan_worker.py"),
        "--input", input_path,
        "--output", output_path,
        "--model-path", model_path,
        "--scale", str(int(scale)),
        "--target-width", str(int(target_width or 0)),
        "--tile", str(int(tile)),
        # JSON: these start with "-c:v", and argparse in the worker would read
        # a bare value beginning with a dash as the next option.
        "--encoder-args", json.dumps(_encoder_args(encoder, crf, guard_config, preset="p7")),
    ]

    proc = guard.spawn(cmd, cwd=FACEFUSION_DIR, env=_cuda_child_env(guard_config),
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")

    error_message = None
    tail = []
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        if line.startswith("ESRGAN_ERROR "):
            error_message = line[len("ESRGAN_ERROR "):]
            continue
        if line.startswith("ESRGAN_PROGRESS "):
            try:
                done, total = (int(x) for x in line.split()[1:3])
            except ValueError:
                continue
            if progress_cb and total:
                progress_cb(int(done / total * 100), f"Upscaling frame {done}/{total}")
            if log_cb and total and done % 60 == 0:
                log_cb(f"  Processing frame {done}/{total}")
            continue
        tail.append(line)
        del tail[:-20]

    proc.wait()
    guard.check()

    if proc.returncode != 0:
        detail = error_message or ("\n".join(tail) if tail else "no output")
        raise PipelineError(f"Real-ESRGAN upscale failed: {detail}")

    return output_path


def interpolate_video(input_path, output_path, multi=2, infer_scale=0.75,
                     encoder="hevc_nvenc", crf=18, log_cb=None, guard=None,
                     progress_cb=None, guard_config=None, stage_name="Interpolation",
                     start_pct=0, end_pct=100):
    """
    Raise a clip's frame rate with RIFE, in a worker process.

    Runs per eye rather than on the merged stereo frame, and that is not a
    preference: the model needs about 8.7 GB of VRAM on a 4096x4096 eye, and
    the merged 8192x4096 frame is four times the pixels. It would not fit.

    Measured end to end on one VR eye at 4096x4096: 0.96 interpolated frame/s,
    so doubling a 60-second 90 fps clip costs roughly 1.5 hours per eye. The
    caller is expected to have said so before starting.
    """
    guard = guard or NULL_GUARD
    guard.check()
    if multi < 2:
        raise PipelineError("Frame interpolation needs a multiplier of 2 or more.")

    guard_config = guard_config or {}
    probe = probe_video(input_path)
    if log_cb:
        log_cb(f"[{stage_name}] {probe['fps']:.0f} -> {probe['fps'] * multi:.0f} fps "
               f"at {probe['width']}x{probe['height']} (RIFE, {multi}x)")

    cmd = [
        PYTHON, os.path.join(SCRIPTS_DIR, "rife_worker.py"),
        "--input", input_path,
        "--output", output_path,
        "--multi", str(int(multi)),
        "--infer-scale", str(float(infer_scale)),
        "--encoder", encoder,
        "--crf", str(int(crf)),
        "--maxrate", str(int(guard_config.get("nvenc_maxrate_mbps") or 120)),
        "--decode-threads", str(int(guard_config.get("ffmpeg_reader_threads", 4) or 4)),
    ]

    proc = guard.spawn(cmd, cwd=FACEFUSION_DIR, env=_cuda_child_env(guard_config),
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")

    error_message = None
    tail = []
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        if line.startswith("RIFE_ERROR "):
            error_message = line[len("RIFE_ERROR "):]
            continue
        if line.startswith("RIFE_INFO "):
            if log_cb:
                log_cb(f"  {line[len('RIFE_INFO '):]}")
            continue
        if line.startswith("RIFE_PROGRESS "):
            try:
                done, total = (int(x) for x in line.split()[1:3])
            except ValueError:
                continue
            if progress_cb and total:
                progress_cb(int(start_pct + (done / total) * (end_pct - start_pct)),
                            f"{stage_name}: frame {done}/{total}")
            if log_cb and total and done % 500 < multi:
                log_cb(f"  {stage_name}: {done}/{total} frames")
            continue
        if line.startswith("RIFE_DONE "):
            if log_cb:
                log_cb(f"  {line[len('RIFE_DONE '):]}")
            continue
        tail.append(line)
        del tail[:-20]

    proc.wait()
    guard.check()

    if proc.returncode != 0:
        detail = error_message or ("\n".join(tail) if tail else "no output")
        raise PipelineError(f"Frame interpolation failed: {detail}")

    return output_path


def upscale_video(input_path, output_path, method="realesrgan", scale=2, target_width=None,
                  encoder="hevc_nvenc", crf=20, log_cb=None, guard=None, tile=512,
                  progress_cb=None, guard_config=None):
    """Dispatch to upscaling method."""
    # Both methods end at the same encoder, so the ceiling is checked once here
    # rather than in each of them - and before Real-ESRGAN spends a worker
    # process and a model load on a frame that can never be written.
    check_upscale_encoder(
        *upscale_output_size(probe_video(input_path), scale, target_width), encoder)

    if method == "realesrgan":
        return upscale_video_realesrgan(input_path, output_path, scale, target_width, encoder,
                                        crf, log_cb, guard=guard, tile=tile,
                                        progress_cb=progress_cb, guard_config=guard_config)
    elif method == "ffmpeg":
        return upscale_video_ffmpeg(input_path, output_path, scale, target_width, encoder,
                                    crf, log_cb, guard=guard, guard_config=guard_config)
    else:
        raise PipelineError(f"Unknown upscale method: {method}")


# ffmpeg's minterpolate allocates three per-pixel tables for the whole frame in
# single blocks, and av_calloc refuses any one allocation over INT_MAX. Measured
# ceiling on this build: 4064x4064 (16.52 MP) configures, 4096x4096 (16.78 MP)
# fails instantly with "Cannot allocate memory" - and it is not a RAM shortage,
# it failed with 17 GB free. One eye of an 8K VR clip is 16.78 MP and the merged
# stereo frame is 33.55 MP, so both are out of reach.
MINTERPOLATE_MAX_PIXELS = 16.5e6


def apply_temporal_smoothing(input_path, output_path, encoder="hevc_nvenc", crf=18,
                             log_cb=None, guard=None, guard_config=None):
    """Apply motion-compensated interpolation for temporal smoothing to reduce flickering."""
    guard = guard or NULL_GUARD
    guard.check()
    if log_cb:
        log_cb("Applying temporal smoothing to reduce flickering...")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    probe = probe_video(input_path)
    fps = probe.get("fps", 30)

    pixels = probe["width"] * probe["height"]
    if pixels > MINTERPOLATE_MAX_PIXELS:
        # Without this the run spends minutes copying tens of GB, only to hand
        # back the file unchanged: ffmpeg would fail during filter setup and the
        # fallback below would copy the input verbatim. Say so and move it.
        if log_cb:
            log_cb(
                f"Temporal smoothing skipped: ffmpeg's minterpolate cannot handle a "
                f"{probe['width']}x{probe['height']} frame ({pixels / 1e6:.1f} MP). Its "
                f"per-pixel tables exceed the 2 GB limit on a single allocation above "
                f"about {MINTERPOLATE_MAX_PIXELS / 1e6:.1f} MP, whatever the machine has "
                f"free.\n"
                f"    The video is unaffected - this step only ever reduced flicker. To "
                f"use it, render at 4064x4064 per frame or less."
            )
        # The callers treat the input as dead the moment this returns, so moving
        # beats copying by tens of GB.
        shutil.move(input_path, output_path)
        return output_path

    # minterpolate is single-threaded, extremely slow, and holds several full
    # frames in memory. Cap ffmpeg's thread use so it cannot spread out.
    cmd = (
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        + _decode_args(guard_config)
        + ["-i", input_path,
           "-vf", f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
           "-threads", str(max(2, min(6, (os.cpu_count() or 8) // 4)))]
        + _encoder_args(encoder, crf, guard_config, preset="p7")
        + ["-c:a", "copy", output_path]
    )
    result = guard.run(cmd, capture_output=True)
    if result.returncode != 0:
        guard.check()
        if log_cb:
            log_cb(f"Temporal smoothing failed, using original: {(result.stderr or b'').decode(errors='replace')[:200]}")
        shutil.copy(input_path, output_path)
    return output_path


def merge_videos(left_path, right_path, output_path, layout="sbs", encoder="hevc_nvenc", crf=20,
                 preset=None, add_vr_metadata=True, fps=None, log_cb=None, guard=None,
                 guard_config=None, progress_cb=None, start_pct=0, end_pct=0, lane=None):
    guard = guard or NULL_GUARD
    guard.check()
    if log_cb:
        maxrate = int((guard_config or {}).get("nvenc_maxrate_mbps") or 120)
        log_cb(f"Merging into {layout.upper()} video (cq {crf}, cap {maxrate} Mbit/s)")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    filt = "[0:v][1:v]hstack=2[vout]" if layout == "sbs" else "[0:v][1:v]vstack=2[vout]"

    dec = _decode_args(guard_config)
    cmd = (["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
           + dec + ["-i", left_path]
           + dec + ["-i", right_path]
           + ["-filter_complex", filt])
    if fps:
        cmd += ["-r", str(fps)]
    cmd += ["-map", "[vout]", "-map", "0:a?"]
    # p7 is the highest NVENC quality tier. This is the one encode whose output
    # the user actually keeps, and it runs once, so there is no reason to be
    # anywhere but the top of the ladder.
    cmd += _encoder_args(encoder, crf, guard_config, preset="p7")
    cmd += ["-c:a", "aac", "-b:a", "192k"]
    if add_vr_metadata:
        # Also written here so a merge run on its own is tagged; the pipeline
        # re-applies it at the end via apply_stereo_metadata(), because the
        # optional smoothing and upscaling passes re-encode and lose it.
        cmd += ["-metadata:s:v:0", f"stereo_mode={'left_right' if layout == 'sbs' else 'top_bottom'}"]
    cmd += [output_path]

    on_frame = None
    total_frames = 0
    if progress_cb and end_pct > start_pct:
        eye = probe_video(left_path)
        total_frames = int((eye.get("duration") or 0.0) * (fps or eye["fps"]))

        def on_frame(done, total):
            progress_cb(int(start_pct + min(1.0, done / float(total)) * (end_pct - start_pct)),
                        f"Merging: frame {done}/{total}")

    code, err = _run_ffmpeg_watched(cmd, guard, on_frame=on_frame,
                                    total_frames=total_frames, lane=lane)
    if code != 0:
        guard.check()
        raise PipelineError(f"Merge failed: {_ffmpeg_error_excerpt(err)}")

    return output_path


# stereo_mode is a Matroska element. ffmpeg accepts -metadata:s:v:0
# stereo_mode=... for any container and silently drops it for the ones that
# have nowhere to put it.
STEREO_METADATA_FORMATS = {".mkv", ".webm"}

# MP4 says the same thing, and more, in Google's Spherical Video V2 boxes -
# which ffmpeg can read and write but has no CLI option to originate, so
# scripts/vr_metadata.py writes them directly.
SPHERICAL_METADATA_FORMATS = {".mp4", ".mov", ".m4v"}


def apply_stereo_metadata(path, layout, log_cb=None, guard=None):
    """
    Tell the headset what this file is, in place, without re-encoding.

    This runs last rather than as part of the merge because every optional step
    after the merge - temporal smoothing, upscaling - re-encodes into a fresh
    file and would drop the tag again.

    Two containers, two mechanisms, and only one of them describes the
    projection:

      * MP4 gets `st3d` (eye packing) and `sv3d` (equirectangular, 180 or 360).
        This is the one that matters - without it a headset has no reason to
        think the file is anything but flat 2D video, and plays it on a curved
        virtual screen.
      * MKV gets Matroska's `stereo_mode`, which only ever describes the eye
        packing. A tagged MKV is still not a sphere.
    """
    guard = guard or NULL_GUARD
    ext = os.path.splitext(path)[1].lower()
    mode = "left_right" if layout == "sbs" else "top_bottom"

    if ext in SPHERICAL_METADATA_FORMATS:
        try:
            probe = probe_video(path)
            half = vr_metadata.half_sphere_for(probe["width"], probe["height"], layout)
            vr_metadata.inject(path, layout=layout, half_sphere=half, log_cb=log_cb)
        except Exception as exc:
            if log_cb:
                log_cb(f"Could not write the VR metadata ({exc}); the file is fine but "
                       f"a player may show it as flat 2D. Pick .mkv, or rename the file "
                       f"with a {vr_metadata.suffix_for(layout)} suffix, to work around it.")
            return path

        if log_cb and not os.path.basename(path).upper().endswith(
                vr_metadata.suffix_for(layout).upper() + ext.upper()):
            log_cb(
                f"Tip: DeoVR, Skybox and Pigasus decide the layout from the filename "
                f"before they read any metadata. Renaming this to "
                f"'{os.path.basename(vr_metadata.name_with_convention(path, layout))}' "
                f"makes it work in those too."
            )
        return path

    if ext not in STEREO_METADATA_FORMATS:
        if log_cb:
            log_cb(f"Note: {ext or 'this container'} cannot store VR metadata, "
                   f"so the output is not self-describing - the headset or player has to "
                   f"be told it is {layout.upper()}. Use .mp4 or .mkv to have it tagged.")
        return path

    stamped = path + ".stereo" + ext
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", path,
           "-map", "0", "-c", "copy", "-metadata:s:v:0", f"stereo_mode={mode}", stamped]
    result = guard.run(cmd, capture_output=True)

    if result.returncode != 0 or not os.path.exists(stamped):
        if log_cb:
            log_cb("Could not write the stereo_mode tag; leaving the file untagged.")
        if os.path.exists(stamped):
            os.remove(stamped)
        return path

    os.replace(stamped, path)
    if log_cb:
        log_cb(f"Tagged as stereo_mode={mode}")
        # Worth saying plainly, because the older notes in PROGRESS.md
        # recommended .mkv for exactly this and that advice is now backwards.
        log_cb(
            "Note: stereo_mode describes how the eyes are packed and nothing else - "
            "it cannot say the video is a 180-degree sphere, so a player may still "
            "show this as a flat stereo pair. Render to .mp4 to get the full VR "
            "tagging (st3d + sv3d)."
        )
    return path


def cleanup_temp(temp_dir, items=None):
    if items is None:
        # Prefixes, not names: the merged and smoothed intermediates take the
        # output's container extension so the final move cannot rename an mp4
        # to .mkv, which is how "choose .mkv for stereo metadata" used to
        # produce an mp4 with an .mkv name and no metadata.
        items = ["ff_temp_left", "ff_temp_right", "ff_temp_normal",
                 "left.mp4", "right.mp4", "left_swapped.mp4", "right_swapped.mp4",
                 "left_interp.mp4", "right_interp.mp4",
                 "smoothed_result.mp4", "ff_result.mp4", "trimmed_target.mp4",
                 "upscale_frames", RUN_STATE_NAME]
        for name in os.listdir(temp_dir) if os.path.isdir(temp_dir) else []:
            if name.startswith(("merged_vr.", "smoothed_vr.")):
                items.append(name)
            # An overlapped chunked run gives each chunk its own directory
            # under output/temp so that the next chunk's split can be written
            # while this chunk's eyes are still waiting to be merged. They are
            # only ever swept at the top level, on success - a chunk directory
            # is exactly what a resume needs.
            elif name.startswith("chunk_") and os.path.isdir(
                    os.path.join(temp_dir, name)):
                items.append(name)
    for item in items:
        path = os.path.join(temp_dir, item)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)


# ---------------------------------------------------------------------------
# Run checkpoints
#
# A VR run is a chain of stages that each cost minutes to hours, and until now
# any failure - most often the guard rails aborting on low memory during the
# second eye - threw the whole chain away: the `finally` below wiped temp, so
# the retry re-split the source and re-swapped an eye that had already
# finished perfectly.
#
# Every stage that produces a file now records it in output/temp/run_state.json
# along with the file's measured duration and a key covering everything that
# decides what the file contains. A later run recomputes those keys; any stage
# whose key still matches, and whose file is still on disk at the length it was
# written, is skipped. Change the face, the clip, the trim or a setting and the
# key moves, so nothing stale is ever reused.
# ---------------------------------------------------------------------------

RUN_STATE_NAME = "run_state.json"

# Settings that only reach stages after the swap. Excluded from the swap key so
# that turning upscaling on does not invalidate two finished eyes.
POST_SWAP_KEYS = frozenset({
    "upscale_method", "upscale_scale", "upscale_target_width", "temporal_smoothing",
    # Interpolation runs after the swap too, so changing it must not throw away
    # two finished eyes - it gets its own stage key instead.
    "interpolate_multi", "interpolate_scale",
})


def video_frame_count(path, count_packets=False):
    """
    Frames in the video stream of `path`, or 0 when it cannot be known cheaply.

    Reads `nb_frames` out of the container header, which NVENC and libx26x both
    write and which costs one ffprobe and no decoding. `count_packets=True`
    walks the file instead - exact for any container, but seconds per gigabyte,
    so it is opt-in and used by the tests rather than by the pipeline.

    0 means "no answer", never "empty file": every caller treats it as a check
    that could not be made rather than as a failure.
    """
    if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
        return 0
    if count_packets:
        entries = "stream=nb_read_packets"
        extra = ["-count_packets"]
    else:
        entries = "stream=nb_frames"
        extra = []
    cmd = (["ffprobe", "-v", "error", "-select_streams", "v:0"] + extra
           + ["-show_entries", entries, "-of", "default=nw=1:nk=1", path])
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return int((result.stdout or "").strip().splitlines()[0])
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        return 0


def expected_eye_geometry(target_video, layout, trim_start=None, trim_end=None):
    """
    (width, height, seconds) that one eye of `target_video` should come out at.

    From the source's own header: half its width for side-by-side, half its
    height for top-bottom, and the length of the trimmed range. Seconds is
    0.0 when the source cannot be measured, and callers treat 0.0 as "do not
    check the length".
    """
    probe = probe_video(target_video)
    if layout == "sbs":
        width, height = probe["width"] // 2, probe["height"]
    else:
        width, height = probe["width"], probe["height"] // 2
    duration = float(probe.get("duration") or 0.0)
    start = max(0.0, float(trim_start or 0.0))
    if trim_end is not None:
        end = float(trim_end) if duration <= 0 else min(float(trim_end), duration)
    else:
        end = duration
    seconds = max(0.0, end - start) if end > 0 else 0.0
    return width, height, seconds


def describe_wrong_eye(path, expected_size=None, expected_seconds=None, allow_scaled=False):
    """
    Why the file at `path` is not the eye it is supposed to be, in a phrase -
    or None if it is.

    A checkpoint proves a stage finished and that its file is still the
    length it was then. It does not prove the file is this clip's eye. On
    Sep 10 2026 an older scripts/tests/test_overlap.py, which still shared
    output/temp with the app, dropped its 15 s 320x240 stand-in onto chunk
    1's right eye while a real run was mid-split; the split checkpoint then
    recorded that file's length as the truth, the restart reused it, and
    FaceFusion dutifully swapped a 320x240 clip. That is what this catches.

    `allow_scaled` is for a swapped eye with the frame enhancer on, which
    comes out at an integer multiple of the input.
    """
    if not os.path.isfile(path):
        return None
    if os.path.getsize(path) == 0:
        return "it is an empty file"
    probe = probe_video(path)
    width, height = probe["width"], probe["height"]
    seconds = float(probe.get("duration") or 0.0)
    if seconds <= 0:
        return "it cannot be read as a video"
    if expected_size and (width, height) != tuple(expected_size):
        ew, eh = expected_size
        scaled = (allow_scaled and ew and eh and width % ew == 0 and height % eh == 0
                  and width // ew == height // eh)
        if not scaled:
            return f"it is {width}x{height} where this clip's eye is {ew}x{eh}"
    if expected_seconds and expected_seconds > 0:
        tolerance = max(1.0, expected_seconds * 0.05)
        if abs(seconds - expected_seconds) > tolerance:
            return (f"it holds {seconds:.1f}s where this range is "
                    f"{expected_seconds:.1f}s")
    return None


def check_eyes_match(left_path, right_path, expected_seconds=None, log_cb=None):
    """
    The two eyes about to be merged must be the same shape and the same length.

    hstack cannot put a 320x240 file beside a 4096x4096 one, and the way
    ffmpeg reports that - "Could not open encoder before EOF" - reads as an
    NVENC failure. Worse, with the merge running in the background under the
    next chunk's swap, that error only surfaces at the following chunk
    boundary, an hour and a half after the eye went wrong (Sep 10 2026:
    right eye wrong at 13:36, reported at 15:47). Two ffprobe calls here
    turn that into an immediate, named failure.
    """
    for label, path in (("left", left_path), ("right", right_path)):
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise PipelineError(
                f"The {label} eye is missing or empty, so there is nothing to "
                f"merge: {path}")
    left, right = probe_video(left_path), probe_video(right_path)
    lw, lh, ld = left["width"], left["height"], float(left.get("duration") or 0.0)
    rw, rh, rd = right["width"], right["height"], float(right.get("duration") or 0.0)
    if (lw, lh) != (rw, rh):
        raise PipelineError(
            f"The two eyes are not the same shape - left {lw}x{lh}, right {rw}x{rh} - "
            f"so they cannot be merged side by side. An eye this different was not "
            f"made from this clip: something else wrote it (a test or a second run "
            f"sharing the same temp directory, a file copied from another clip).\n"
            f"    Left:  {left_path}\n"
            f"    Right: {right_path}\n"
            f"Delete the wrong one and start the run again with the same settings; "
            f"resume redoes only that eye.")
    reference = expected_seconds if expected_seconds and expected_seconds > 0 else ld
    tolerance = max(1.0, reference * 0.05)
    if expected_seconds and expected_seconds > 0:
        for label, path, seconds in (("left", left_path, ld), ("right", right_path, rd)):
            if abs(seconds - expected_seconds) > tolerance:
                raise PipelineError(
                    f"The {label} eye holds {seconds:.1f}s where this range is "
                    f"{expected_seconds:.1f}s - a partial encode or the wrong file, and "
                    f"merging it would put a hole in the finished video. The file is at "
                    f"{path}. Delete it and start the run again; resume redoes only "
                    f"that eye.")
    if abs(ld - rd) > tolerance:
        raise PipelineError(
            f"The two eyes are not the same length - left {ld:.1f}s, right {rd:.1f}s - "
            f"and hstack would silently stop at the shorter one.\n"
            f"    Left:  {left_path}\n"
            f"    Right: {right_path}\n"
            f"Delete the short one and start the run again; resume redoes only that eye.")
    if log_cb:
        log_cb(f"  Eyes checked: both {lw}x{lh}, {ld:.1f}s and {rd:.1f}s.")
    return True


def _wrong_reused_eyes(have, paths, target_video, layout, trim_start, trim_end, config):
    """
    {stage name: reason} for every checkpointed eye that is not this clip's eye.

    `paths` maps the four eye stages to their files. One ffprobe of the
    source plus one per reused eye - nothing against the hours a wrong eye
    costs when it is only found at the merge.
    """
    reasons = {}
    if not any(have.get(name) for name in paths):
        return reasons
    width, height, seconds = expected_eye_geometry(target_video, layout, trim_start, trim_end)
    size = (width, height) if width and height else None
    scaled_ok = bool(config.get("frame_enhancer"))
    for name, path in paths.items():
        if not have.get(name):
            continue
        reason = describe_wrong_eye(path, size, seconds,
                                    allow_scaled=scaled_ok and name.startswith("swap_"))
        if reason:
            reasons[name] = reason
    return reasons


def check_merge_frame_counts(left_path, right_path, merged_path, log_cb=None):
    """
    The merged stereo file must hold exactly as many frames as each eye did.

    Cheap (container headers, no decoding) and worth doing on every merge, but
    it exists for the overlapped pipeline in particular. Once a chunk's merge
    runs underneath the *next* chunk's swap, the two are touching the same
    output/temp tree at the same time, and the failure mode of getting that
    wrong is not a crash - it is a merge that reads an eye while something else
    is still writing it, and silently produces a short file. Duration alone
    does not catch it: hstack stops at the shorter input, so a merge that lost
    a hundred frames still reports a plausible length.

    Raises PipelineError on a mismatch. Says nothing and returns False when the
    container did not record a frame count, which is a check that could not be
    made rather than a check that passed.
    """
    left_frames = video_frame_count(left_path)
    right_frames = video_frame_count(right_path)
    merged_frames = video_frame_count(merged_path)
    if not (left_frames and right_frames and merged_frames):
        return False
    if left_frames != right_frames or merged_frames != left_frames:
        raise PipelineError(
            f"The merged file does not hold the frames the eyes did: left "
            f"{left_frames}, right {right_frames}, merged {merged_frames}. "
            f"hstack stops at the shorter input, so this would have gone out as a "
            f"video that plays at the right length and is missing frames.\n"
            f"    Left: {left_path}\n    Right: {right_path}\n"
            f"    Merged: {merged_path}"
        )
    if log_cb:
        log_cb(f"  Frame count checked: {merged_frames} frames in each eye and in "
               f"the merged file.")
    return True


def media_duration(path):
    """Duration of `path` in seconds, or 0.0 if missing, empty or unreadable."""
    if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
        return 0.0
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=nw=1:nk=1", path]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return float((result.stdout or "").strip())
    except (subprocess.SubprocessError, OSError, ValueError):
        return 0.0


def _path_stamp(path):
    """Identify a file by content, not just by name: size + mtime, not the path."""
    try:
        stat = os.stat(path)
        return f"{os.path.abspath(path)}|{stat.st_size}|{int(stat.st_mtime)}"
    except OSError:
        return str(path)


def _stage_key(**parts):
    return hashlib.sha1(
        json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


# The pieces that go into a stage key, in the user's words. A resume that
# decides to redo an eye has to be able to say what changed - "the key moved"
# is exactly the answer that makes resume look broken.
PART_LABELS = {
    "target": "the target video",
    "layout": "the stereo layout",
    "fps": "the output FPS",
    "trim": "the clip range",
    "encoder": "the encoder",
    "sources": "the source face",
    "config": "the quality settings",
    "split": "the split",
    "swap": "the swap",
    "merge": "the merge",
    "interp": "the interpolation settings",
    "multi": "the interpolation multiplier",
    "infer_scale": "the interpolation scale",
    "quality": "the quality",
    "crf": "the merge quality",
    "preset": "the encoder preset",
}

STAGE_LABELS = {
    "split_left": "left eye split", "split_right": "right eye split",
    "swap_left": "left eye swap", "swap_right": "right eye swap",
    "interp_left": "left eye interpolation", "interp_right": "right eye interpolation",
    "merge": "merged stereo video", "smooth": "smoothed video",
}


def _short_stamp(value):
    """`B:/clips/raya.mp4|225685390|1757...` -> `raya.mp4`."""
    text = str(value)
    return os.path.basename(text.split("|", 1)[0]) or text


def _fmt_part(name, value):
    if name == "target":
        return _short_stamp(value)
    if name == "sources":
        items = value if isinstance(value, (list, tuple)) else [value]
        return ", ".join(_short_stamp(v) for v in items) or "none"
    if name == "trim" and isinstance(value, (list, tuple)):
        start, end = (list(value) + [None, None])[:2]
        if start is None and end is None:
            return "whole video"
        return (("start" if start is None else "%.1fs" % float(start)) + " to "
                + ("end" if end is None else "%.1fs" % float(end)))
    return "default" if value is None else str(value)


def _describe_parts_diff(old, new, depth=0):
    """What moved between two stage-key part sets, named the way a user would."""
    if not isinstance(old, dict) or not isinstance(new, dict):
        return ""
    changes = []
    for name in sorted(set(old) | set(new)):
        before, after = old.get(name), new.get(name)
        if before == after:
            continue
        label = PART_LABELS.get(name, name)
        if isinstance(before, dict) and isinstance(after, dict) and depth < 3:
            inner = _describe_parts_diff(before, after, depth + 1)
            changes.append(f"{label} ({inner})" if inner else label)
        else:
            changes.append(f"{label} ({_fmt_part(name, before)} -> "
                           f"{_fmt_part(name, after)})")
    if len(changes) > 5:
        changes = changes[:5] + [f"and {len(changes) - 5} more"]
    return ", ".join(changes)


class RunState:
    """Which stages of a pipeline run have already produced a usable file."""

    def __init__(self, temp_dir, log_cb=None):
        self.temp_dir = temp_dir
        self.path = os.path.join(temp_dir, RUN_STATE_NAME)
        self.log_cb = log_cb
        # Both eyes can finish and record themselves at the same moment now
        # (B10), and two threads writing this file at once would leave half of
        # one JSON document on top of the other - which reads back as "no
        # finished stages" and throws away an hour of GPU time on the retry.
        self._lock = threading.Lock()
        self.stages = {}
        try:
            with open(self.path) as handle:
                data = json.load(handle)
            self.stages = data.get("stages") or {}
        except (OSError, ValueError):
            self.stages = {}

    def reusable(self, name, artifact, key, tolerance=0.5):
        """True if `name` finished in an earlier run and its output still stands."""
        return (name in self.stages
                and self.why_not(name, artifact, key, tolerance=tolerance) is None)

    def why_not(self, name, artifact, key, parts=None, tolerance=0.5):
        """
        Why `name` cannot be reused, in a sentence - or None if it can be.

        Also None when the stage simply never finished: that is not a reuse
        failure and there is nothing to explain.

        The duration check is the point of this: an abort part-way through an
        encode leaves a file that opens perfectly well and holds half the
        frames, and reusing that would put a silently truncated eye into the
        finished video.
        """
        stage = self.stages.get(name)
        if not stage:
            return None
        if stage.get("key") != key:
            diff = _describe_parts_diff(stage.get("parts"), parts)
            if diff:
                return f"{diff} changed since it ran"
            # Recorded by a build that stored only the key. It can still say
            # that something moved, just not what.
            return ("the source, clip range or settings changed since it ran "
                    "(that run did not record which)")
        if stage.get("artifact") != os.path.basename(artifact) or not os.path.isfile(artifact):
            return f"output/temp/{stage.get('artifact')} is no longer on disk"
        recorded = float(stage.get("duration") or 0.0)
        if recorded <= 0:
            return "its length was never recorded, so it cannot be trusted"
        found = media_duration(artifact)
        if abs(found - recorded) > tolerance:
            return (f"the file in output/temp holds {found:.1f}s where the finished "
                    f"stage held {recorded:.1f}s - it is a partial encode")
        return None

    def complete(self, name, artifact, key, parts=None):
        # Measured outside the lock: ffprobe on a 4K eye is not instant, and
        # holding the lock across it would stall the other lane's checkpoint.
        record = {
            "artifact": os.path.basename(artifact),
            "key": key,
            "duration": media_duration(artifact),
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            # Kept beside the key so a later run can say what changed, rather
            # than only that something did.
            "parts": parts,
        }
        with self._lock:
            self.stages[name] = record
            self._save()

    def forget(self, *names):
        with self._lock:
            for name in names:
                self.stages.pop(name, None)
            self._save()

    def clear(self):
        with self._lock:
            self.stages = {}
            try:
                os.remove(self.path)
            except OSError:
                pass

    def _save(self):
        """Write the checkpoint file. Call with the lock held."""
        try:
            with open(self.path, "w") as handle:
                json.dump(
                    {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": self.stages},
                    handle, indent=2,
                )
        except OSError:
            pass


def vr_stage_keys(source_paths, target_video, config, layout, fps, trim_start, trim_end,
                  merge_encoder, merge_crf, merge_preset):
    """
    One key per checkpointed stage, each covering exactly its own inputs.

    Keyed narrowly on purpose: re-extracting the source face has to invalidate
    both swapped eyes, but it must NOT invalidate the split, which does not
    depend on the face at all and costs several minutes at 8K.

    Returns (keys, parts): the hash each stage is checked against, and the
    named inputs behind it. The parts travel into run_state.json with the
    stage, so a later run that cannot reuse it can name what moved. The split
    parts are deliberately unchanged from the first version of this function -
    an eye split at 8K is several minutes, and rearranging its key would throw
    away every split already on disk.
    """
    split_parts = dict(
        target=_path_stamp(target_video), layout=layout, fps=fps,
        trim=[trim_start, trim_end], encoder=config.get("encoder"),
    )
    swap_parts = dict(
        split=split_parts,
        sources=sorted(_path_stamp(p) for p in source_paths),
        config={k: v for k, v in config.items() if k not in POST_SWAP_KEYS},
    )
    # Interpolation sits between the swap and the merge, so it has its own key
    # and the merge depends on its settings.
    interp_settings = dict(
        multi=config.get("interpolate_multi"),
        infer_scale=config.get("interpolate_scale"),
        encoder=config.get("encoder"), quality=config.get("quality"),
    )
    interp_parts = dict(swap=swap_parts, **interp_settings)
    merge_parts = dict(
        swap=swap_parts, layout=layout, fps=fps,
        encoder=merge_encoder, crf=merge_crf, preset=merge_preset,
        interp=interp_settings,
    )
    smooth_parts = dict(merge=merge_parts, quality=config.get("quality"),
                        encoder=config.get("encoder"))
    parts = {
        "split_left": split_parts, "split_right": split_parts,
        "swap_left": swap_parts, "swap_right": swap_parts,
        "interp_left": interp_parts, "interp_right": interp_parts,
        "merge": merge_parts, "smooth": smooth_parts,
    }
    return {name: _stage_key(**value) for name, value in parts.items()}, parts


def _log_resume_miss(state, name, artifact, key, parts, log_cb):
    """Say why a finished-looking stage is being redone, or stay quiet."""
    if not log_cb:
        return
    reason = state.why_not(name, artifact, key, parts)
    if reason:
        log_cb(f"[Resume] Redoing the {STAGE_LABELS.get(name, name)}: {reason}.")


def describe_run_state(temp_dir=None, keys=None, parts=None):
    """
    Lines describing whatever unfinished run is sitting in output/temp.

    Printed at launch: a user who stopped a job overnight should not have to
    guess whether the eight hours already spent are still there - and, just as
    important, should not assume they are when the fields on the window no
    longer describe the job that produced them. Pass `keys` and `parts` from
    vr_stage_keys and each stage is checked against them; without them this
    only says what is on disk.
    """
    temp_dir = temp_dir or os.path.join(PROJECT_DIR, "output", "temp")
    state = RunState(temp_dir)
    if not state.stages:
        return []
    lines = ["[Resume] output/temp still holds finished stages from an earlier attempt:"]
    stale = 0
    for name, stage in state.stages.items():
        label = STAGE_LABELS.get(name, name)
        path = os.path.join(temp_dir, stage.get("artifact") or "")
        when = stage.get("at", "at an unknown time")
        if not os.path.isfile(path):
            lines.append(f"  - {label} - recorded, but its file is GONE from output/temp")
            stale += 1
            continue
        size = os.path.getsize(path) / (1024 ** 3)
        head = f"  - {label} - finished {when}, {size:.1f} GB on disk"
        if not keys or name not in keys:
            lines.append(head)
            continue
        reason = state.why_not(name, path, keys[name], (parts or {}).get(name))
        if reason:
            lines.append(head + f"\n      WILL BE REDONE: {reason}.")
            stale += 1
        else:
            lines.append(head + " - will be reused")
    if keys and stale:
        lines.append("[Resume] Point the fields back at the job those files belong to "
                     "and they are reused as they are; otherwise they are dead weight "
                     "and untick Resume to have them cleared.")
    else:
        lines.append("[Resume] Leave 'Resume from where the last attempt stopped' ticked "
                     "and these are reused; the run log names anything it has to redo, "
                     "and why.")
    return lines


def _ffmpeg_error_excerpt(err, head=400, tail=400):
    """
    The part of ffmpeg's stderr worth showing in an error.

    The last few hundred characters used to be all that was reported, and
    for a filter failure that is the wrong end: ffmpeg says what went wrong
    once, first ("Input 1 height 240 does not match input 0 height 4096"),
    then spends the rest of the log shutting threads down ("Could not open
    encoder before EOF", "Task finished with error code: -22"). Reported on
    Sep 10 2026 as an NVENC failure that was nothing of the sort. Keep both
    ends.
    """
    err = (err or "").strip()
    if len(err) <= head + tail:
        return err
    return err[:head].rstrip() + "\n    [...]\n" + err[-tail:].lstrip()


def _run_ffmpeg_watched(cmd, guard, on_frame=None, total_frames=0, lane=None):
    """
    Run an ffmpeg command and report frames as they are written.

    The split and the merge are single blocking ffmpeg calls, and on a
    full-length VR job they are three and seven hours respectively. Without
    this the progress bar sat still for both of them and the remaining-time
    figure had nothing to go on until the next stage started.

    `-progress pipe:1` makes ffmpeg write `key=value` lines to stdout,
    independently of `-loglevel`, so nothing else has to be parsed. Returns
    (returncode, stderr_tail).
    """
    if not (on_frame and total_frames):
        result = guard.run(cmd, capture_output=True, lane=lane)
        return result.returncode, (result.stderr or b"").decode(errors="replace")

    cmd = cmd[:1] + ["-progress", "pipe:1", "-nostats"] + cmd[1:]
    proc = guard.spawn(cmd, lane=lane, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, bufsize=1, errors="replace")

    # stderr has to be drained or a chatty ffmpeg fills the pipe and blocks
    # forever behind a stdout read that will never come.
    tail = []

    def drain():
        for line in proc.stderr:
            tail.append(line)
            del tail[:-40]

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()

    try:
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("frame=") and on_frame:
                try:
                    on_frame(int(line.split("=", 1)[1]), total_frames)
                except ValueError:
                    pass
            if guard.abort_reason:
                break
    finally:
        proc.wait()
        reader.join(timeout=2)
        getattr(guard, "_forget", lambda _p: None)(proc)

    return proc.returncode, "".join(tail)


class RunClock:
    """
    Wall-clock accounting for a run: what it should cost, what it has cost so
    far, and what is left.

    Two problems, both of which only a multi-hour job makes obvious:

      - The progress bar was laid out on hardcoded percentages that had no
        relationship to time. The face swap owned 75% of the bar and about 85%
        of the clock; the merge owned 9% of the bar and could be seven hours;
        temporal smoothing set the bar BACKWARDS, from 91 to 87. Here each
        stage's slice of the bar is proportional to its estimated seconds, so
        a percentage is a percentage of the wait.

      - Nothing said how long the wait was. The estimate is now logged before
        anything spawns, every 10% reports what that tenth actually cost and
        what is left, and the remaining figure is rescaled by the run's own
        measured pace rather than trusting the model for eight hours.

    The estimate comes from guardrails.estimate_runtime(); what this class
    adds is the live correction and the record that makes the next run's
    estimate better than this one's.
    """

    # The first slice of the bar belongs to work that happens before any
    # costed stage: the CUDA probe, and building source faces from a clip.
    PRELUDE_PCT = 3

    def __init__(self, plan, progress_cb=None, log_cb=None, paused_seconds=None):
        self.plan = plan or {"stages": [], "total_seconds": 0.0, "rough": []}
        self.progress_cb = progress_cb
        self.log_cb = log_cb
        # Cumulative seconds this run has spent frozen by the Pause button.
        # Every elapsed figure here subtracts it, so a run paused overnight
        # does not report that the merge took nine hours, does not halve its
        # own pace estimate, and above all does not teach output/timing.json
        # that a merge on this machine costs nine hours - a poisoned
        # calibration would go on lying to every future run's estimate.
        self._paused_clock = paused_seconds or (lambda: 0.0)

        self.stages = {stage["key"]: dict(stage) for stage in self.plan["stages"]}
        self.order = [stage["key"] for stage in self.plan["stages"]]
        # Stages that run underneath another stage rather than before or after
        # it (B9: a chunk's split and the previous chunk's merge, both hidden
        # under this chunk's swap). They still cost what they cost and are
        # still timed for the calibration, but they are not time anybody waits
        # for - so they get no share of the bar and no place in the budget. A
        # merge that owns 9% of the bar and finishes while the swap is only
        # half done would make the bar lurch and then stall.
        self.overlapped = {stage["key"] for stage in self.plan["stages"]
                           if stage.get("overlapped")}
        self.budget = sum(stage["seconds"] for stage in self.plan["stages"]
                          if stage["key"] not in self.overlapped)

        self.ranges = {}
        cursor = float(self.PRELUDE_PCT)
        span = 100.0 - self.PRELUDE_PCT
        for stage in self.plan["stages"]:
            if stage["key"] in self.overlapped:
                # Zero width: begin() and done() still move the bar to the same
                # place, so nothing jumps.
                self.ranges[stage["key"]] = (int(round(cursor)), int(round(cursor)))
                continue
            share = (stage["seconds"] / self.budget * span) if self.budget else 0.0
            self.ranges[stage["key"]] = (int(round(cursor)), int(round(cursor + share)))
            cursor += share
        self._last_end = int(round(cursor))

        self.started = time.time()
        self.working_started = None      # first real stage, so preflight is not counted
        self.current = None
        self.current_started = None
        self.current_fraction = 0.0
        self.spent_estimate = 0.0        # estimated seconds of finished stages
        self.skipped = set()
        self.actual = {}

        self._pct = 0
        self._decile = 0
        self._decile_at = self.started

        # Readings of the pause clock, one per span that gets measured.
        self._paused_at_start = self._paused_clock()
        self._paused_at_working = self._paused_at_start
        self._paused_at_current = self._paused_at_start
        self._paused_at_decile = self._paused_at_start

    # -- pauses ------------------------------------------------------------

    def _held(self, since):
        """Seconds spent frozen since a recorded reading of the pause clock."""
        return max(0.0, self._paused_clock() - since)

    def _running_since(self, start, since_paused):
        """Wall-clock since `start`, minus whatever of it was spent paused."""
        if not start:
            return 0.0
        return max(0.0, time.time() - start - self._held(since_paused))

    # -- layout ------------------------------------------------------------

    def range(self, key):
        """The slice of the progress bar this stage owns."""
        return self.ranges.get(key, (self._last_end, self._last_end))

    def remaining_budget(self):
        return max(0.0, sum(
            stage["seconds"] for key, stage in self.stages.items()
            if key not in self.skipped and key not in self.actual
            and key not in self.overlapped
        ) - self.current_fraction * self._current_seconds())

    def _current_seconds(self):
        if self.current in self.overlapped:
            return 0.0
        stage = self.stages.get(self.current) if self.current else None
        return stage["seconds"] if stage else 0.0

    # -- pace --------------------------------------------------------------

    def _pace(self):
        """
        actual / predicted so far, or 1.0 while there is too little to go on.

        Below a few percent of the run this is noise - a stage's first frames
        include model loading - so the model's own numbers are used until
        there is something worth trusting.
        """
        spent = self.spent_estimate + self.current_fraction * self._current_seconds()
        if not self.working_started or spent < max(20.0, 0.02 * self.budget):
            return 1.0
        elapsed = self._running_since(self.working_started, self._paused_at_working)
        return max(0.25, min(4.0, elapsed / spent))

    def remaining_seconds(self):
        return self.remaining_budget() * self._pace()

    def elapsed_seconds(self):
        return self._running_since(self.started, self._paused_at_start)

    # -- stage transitions -------------------------------------------------

    def begin(self, key):
        self.current = key
        self.current_started = time.time()
        self.current_fraction = 0.0
        self._paused_at_current = self._paused_clock()
        if self.working_started is None:
            self.working_started = self.current_started
            self._paused_at_working = self._paused_at_current
        start, _ = self.range(key)
        stage = self.stages.get(key)
        if stage and self.log_cb:
            self.log_cb(f"[Time] {stage['label']}: estimated "
                        f"{guardrails.fmt_duration(stage['seconds'])}.")
        self._emit(start, stage["label"] if stage else key)

    def skip(self, key):
        """A stage that will not run - reused from an earlier attempt, or off."""
        self.skipped.add(key)
        self.current = None
        self.current_started = None
        self.current_fraction = 0.0
        stage = self.stages.get(key)
        if stage and self.log_cb:
            self.log_cb(f"[Time] {stage['label']}: reused, saving about "
                        f"{guardrails.fmt_duration(stage['seconds'])}.")
        _, end = self.range(key)
        self._emit(end, (stage["label"] + " - reused") if stage else "reused")

    def done(self, key, record=True):
        stage = self.stages.get(key)
        started = self.current_started
        # Cleared before the remaining figure is worked out: while `current`
        # still names this stage, remaining_budget() would subtract the whole
        # of it a second time and report nothing left to do.
        self.current = None
        self.current_started = None
        self.current_fraction = 0.0
        if stage and started:
            taken = self._running_since(started, self._paused_at_current)
            self.actual[key] = taken
            self.spent_estimate += stage["seconds"]
            if self.log_cb:
                self.log_cb(f"[Time] {stage['label']}: took "
                            f"{guardrails.fmt_duration(taken)} "
                            f"(estimated {guardrails.fmt_duration(stage['seconds'])}); "
                            f"{guardrails.fmt_duration(self.remaining_seconds())} left.")
            if record:
                guardrails.record_stage_timing(stage["kind"], stage["seconds"], taken)
        _, end = self.range(key)
        self._emit(end, stage["label"] if stage else key)

    # -- the callback handed to every stage --------------------------------

    def progress(self, pct, text=""):
        """Drop-in for the pipeline's progress_cb, with the ETA folded in."""
        if self.current:
            start, end = self.range(self.current)
            if end > start:
                self.current_fraction = max(0.0, min(1.0, (pct - start) / float(end - start)))
        self._emit(pct, text)

    def _emit(self, pct, text=""):
        # Never let the bar run backwards: a stage that finishes early used to
        # hand the next one a lower number and the bar visibly jumped back.
        pct = max(self._pct, min(100, int(pct)))
        self._pct = pct
        self._report_decile(pct)

        if self.progress_cb:
            left = self.remaining_seconds()
            if left > 0 and pct < 100 and self.working_started:
                text = f"{text}  -  {guardrails.fmt_duration(left)} left"
            self.progress_cb(pct, text)

    def _report_decile(self, pct):
        decile = pct // 10
        if decile <= self._decile or not self.log_cb:
            self._decile = max(self._decile, decile)
            return
        now = time.time()
        step = self._running_since(self._decile_at, self._paused_at_decile)
        crossed = decile - self._decile
        self._decile = decile
        self._decile_at = now
        self._paused_at_decile = self._paused_clock()
        left = self.remaining_seconds()
        span = "10%" if crossed == 1 else f"{crossed * 10}%"
        self.log_cb(
            f"[Time] {decile * 10}% - that {span} took {guardrails.fmt_duration(step)}, "
            f"{guardrails.fmt_duration(self.elapsed_seconds())} elapsed, "
            f"about {guardrails.fmt_duration(left)} left"
            + (f" (done around {guardrails.fmt_clock(left)})" if left > 120 else "")
        )

    def finish(self):
        if not self.log_cb:
            return
        # Against what the stages that actually ran were expected to cost -
        # comparing a resumed run against the whole job would say it beat the
        # estimate by an hour when all it did was skip an hour of work.
        expected = sum(stage["seconds"] for key, stage in self.stages.items()
                       if key not in self.skipped and key not in self.overlapped)
        self.log_cb(f"[Time] Finished in {guardrails.fmt_duration(self.elapsed_seconds())} "
                    f"(estimated {guardrails.fmt_duration(expected)}"
                    + (f", {len(self.skipped)} stage(s) reused" if self.skipped else "")
                    + ").")


def log_runtime_estimate(plan, log_cb, is_vr):
    """Say up front what the run is going to cost, stage by stage."""
    if not log_cb or not plan or not plan.get("stages"):
        return
    log_cb(f"[Time] Estimated total: ~{guardrails.fmt_duration(plan['total_seconds'])} "
           f"(finishing around {guardrails.fmt_clock(plan['total_seconds'])})")
    for stage in plan["stages"]:
        share = stage["seconds"] / plan["total_seconds"] * 100 if plan["total_seconds"] else 0
        log_cb(f"  - {stage['label']}: {guardrails.fmt_duration(stage['seconds'])} "
               f"({share:.0f}%)")
    if plan.get("calibrated"):
        log_cb("  Adjusted by what previous runs on this machine actually took.")
    else:
        log_cb("  First run: this is the model's guess and it will be corrected "
               "from the real pace within the first few percent.")
    for note in plan.get("rough", []):
        log_cb(f"  Note: {note}.")


def prepare_run(target_video, config, is_vr, trim_start=None, trim_end=None, log_cb=None,
                temp_dir=None, force=False):
    """
    Run the guard-rail preflight and return (safe_config, guard_config, estimate).

    Raises PipelineError if the run cannot safely proceed. Call this before
    starting a pipeline so the GUI can report the problem without having
    already spawned anything.
    """
    temp_dir = temp_dir or os.path.join(PROJECT_DIR, "output", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    guard_config = guardrails.load_guard_config()

    # facefusion/ is an untracked clone, so our memory patches there can be
    # undone by a pull without anything in this repo noticing. Say so rather
    # than quietly using several GB more RAM than the estimates assume.
    if log_cb:
        try:
            import patch_facefusion
            _, _missing, _broken = patch_facefusion.status()
        except Exception:
            _missing, _broken = [], []
        if _missing or _broken:
            log_cb("[Guard] FaceFusion is missing VrDeep's memory patches, so this run "
                   "will hold several GB more RAM than the estimates below assume.")
            for _item in _missing + _broken:
                log_cb(f"  - {_item}")
            log_cb("  Fix: python scripts/patch_facefusion.py")

    # A CPU fallback is not a slow run, it is a broken one: the same job goes
    # from minutes to most of a day. Catch it here rather than three hours in.
    if "cuda" in str(config.get("provider", "")).split():
        gpu = check_gpu(log_cb=log_cb)
        if not gpu["ok"] and not force:
            raise PipelineError(
                "CUDA is not usable, so this run would execute on the CPU and "
                "take roughly 10x longer.\n\n"
                f"onnxruntime: {gpu.get('ort')}\n"
                f"detail: {gpu.get('error') or 'CUDA provider loaded but did not activate'}\n\n"
                "Fix: reinstall the GPU runtime into facefusion/venv:\n"
                "  facefusion/venv/Scripts/python.exe -m pip install "
                "onnxruntime-gpu==1.24.4 nvidia-cudnn-cu12 nvidia-cublas-cu12 "
                "nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-runtime-cu12"
            )

    # Checked here rather than when the upscale stage is reached, which is
    # after the swap has already run - a missing 200 MB download is not worth
    # discovering half an hour in.
    if config.get("upscale_method") == "realesrgan":
        # Ask the worker itself rather than repeating its checks here. It knows
        # two separate things that stop this path - whether the weights are
        # present, and whether the installed torch has kernels for this GPU -
        # and a second copy of that logic here would eventually disagree with
        # the copy that actually runs. Measured on this machine: torch
        # 2.5.1+cu121 is built for sm_50-sm_90 and this card is sm_120, so
        # installing the weights alone would not have helped, which is exactly
        # what the old weights-only message told the user to go and do.
        _probe = subprocess.run(
            [PYTHON, os.path.join(SCRIPTS_DIR, "esrgan_worker.py"), "--preflight",
             "--model-path", os.path.join(FACEFUSION_DIR, "weights",
                                          "realesr-general-x4v3.pth")],
            cwd=FACEFUSION_DIR, capture_output=True, text=True, errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if "ESRGAN_PREFLIGHT_OK" not in (_probe.stdout or ""):
            _detail = ""
            for _line in (_probe.stdout or "").splitlines():
                if _line.startswith("ESRGAN_ERROR "):
                    _detail = _line[len("ESRGAN_ERROR "):]
            _detail = _detail or (_probe.stderr or "").strip()[-400:] or (
                "the Real-ESRGAN worker did not start")
            raise PipelineError(
                "Output Upscaling is set to Real-ESRGAN, which cannot run on "
                "this machine.\n\n" + _detail
            )
    # Same reasoning as the Real-ESRGAN check above, for the other way the upscale
    # stage can fail: an output size the encoder cannot accept. Checking it
    # here means a bad Target Width is caught in the first second rather than
    # after the swap has already run.
    if config.get("upscale_method") and not is_image_path(target_video):
        _probe = probe_video(target_video)
        # In VR the upscale runs on the merged frame, which is back at the
        # source's full size, so this is the right thing to measure in both
        # modes.
        _out_w, _out_h = upscale_output_size(
            _probe, config.get("upscale_scale", 2), config.get("upscale_target_width"))
        check_upscale_encoder(_out_w, _out_h, config.get("encoder", "hevc_nvenc"))
        if log_cb:
            log_cb(f"[Guard] Upscale will produce {_out_w}x{_out_h} "
                   f"from {_probe['width']}x{_probe['height']}")

    # One frame of the pass that is about to run: for VR that is one eye, since
    # the two are swapped one after the other and never both in memory.
    _pass = probe_video(target_video)
    pass_pixels = _pass["width"] * _pass["height"] / (2.0 if is_vr else 1.0)

    safe_config = guardrails.clamp_config(
        config, guard_config, log_cb=log_cb, frame_pixels=pass_pixels
    )

    # A chunked run never holds more than one chunk's temp files, so the disk
    # check is made against one chunk - otherwise a full-length 8K clip is
    # refused for the frame dumps of a range it will never extract at once.
    # The time estimate is still for the whole range, because that is what
    # the wait is.
    plan = [(trim_start, trim_end)]
    if safe_config.get("chunk_seconds") and not is_image_path(target_video):
        plan = chunk_plan(_pass.get("duration") or 0.0, trim_start, trim_end,
                          safe_config["chunk_seconds"])
    chunked = len(plan) > 1
    # An overlapped chunked run holds one more chunk of split intermediates
    # than a sequential one, because the next chunk is being split while this
    # one swaps. The disk check has to know that before it approves the run.
    overlap = bool(chunked and is_vr and guard_config.get("overlap_stages", True))
    blocking, warnings, estimate = guardrails.preflight(
        target_video, temp_dir, safe_config, guard_config, is_vr=is_vr,
        trim_start=plan[0][0] if chunked else trim_start,
        trim_end=plan[0][1] if chunked else trim_end,
        overlap=overlap,
    )
    if chunked:
        whole = guardrails.estimate_disk_need(
            target_video, is_vr, guard_config, trim_start, trim_end,
            pipeline_config=safe_config, overlap=overlap)
        estimate["runtime"] = guardrails.estimate_runtime(
            whole, safe_config, is_vr,
            parallel_eyes=bool((estimate.get("parallel_eyes") or {}).get("ok")))
        estimate["chunks"] = plan
        estimate["frames_total"] = whole.get("frames")
        if log_cb:
            log_cb(f"[Chunks] The range will be processed in {len(plan)} chunks of "
                   f"~{guardrails.fmt_duration(safe_config['chunk_seconds'])}; the temp "
                   f"figures below are per chunk, the time is for all of them.")

    # probe_media has no duration for a still, so it falls back to sixty
    # seconds of video and the time estimate would offer to spend ten minutes
    # on one photograph.
    if is_image_path(target_video):
        # Rebuilt, so it has to be told about the lanes again or the plan would
        # disagree with the run about how many swap stages there are.
        estimate["runtime"] = guardrails.estimate_runtime(
            dict(estimate, frames=1), safe_config, is_vr,
            parallel_eyes=bool((estimate.get("parallel_eyes") or {}).get("ok")))

    if log_cb:
        log_cb(
            f"[Guard] Estimated temp usage: {guardrails.fmt_gb(estimate['bytes'])} "
            f"({estimate['frames']} frames, {estimate['frame_format']}), "
            f"free: {guardrails.fmt_gb(estimate.get('free_disk', 0))}"
        )
        # Resolution per pass is the number that decides whether a run fits in
        # RAM, and it is not something the GUI shows anywhere else - an 8K VR
        # source reads as "1 minute" and processes as 4096x4096 frames.
        log_cb(
            f"[Guard] Source {_pass['width']}x{_pass['height']} @ {_pass['fps']:.0f} fps"
            + (" (2 passes, one per eye)" if is_vr else "")
            + f" - {pass_pixels / 1e6:.1f} MP per pass, {pass_pixels * 3 / 1e6:.0f} MB per raw frame"
        )
        log_cb(
            f"[Guard] Estimated memory: "
            f"{estimate.get('resident_need_mb', 0) / 1024:.1f} GB RAM "
            f"(of {guardrails.fmt_gb(estimate.get('avail_ram', 0))} free), "
            f"{estimate.get('commit_need_mb', 0) / 1024:.1f} GB committed "
            f"(cap {guard_config.get('job_memory_cap_gb', 0)} GB)"
        )
        verdict = estimate.get("parallel_eyes") or {}
        if is_vr and verdict.get("line"):
            # Logged either way. "No" is the answer on this card at 4K with an
            # enhancer, and a user who has just read that the swap is two hours
            # is owed the reason it is not one.
            log_cb(verdict["line"])
        log_runtime_estimate(estimate.get("runtime"), log_cb, is_vr)
        for warning in warnings:
            log_cb(f"[Guard] {warning}")

    if blocking and not force:
        error = PipelineError(
            "Refusing to start - this run would very likely lock up the machine:\n\n"
            + "\n\n".join(text for _kind, text in blocking)
        )
        # Only when EVERY problem is one that can clear by itself: a run that
        # is short of commit *and* short of disk is not going to fit later
        # either, and deferring it would just retry the same refusal all night.
        error.transient = all(kind in ("commit", "ram") for kind, _text in blocking)
        raise error
    if blocking and force and log_cb:
        log_cb("[Guard] Preflight problems OVERRIDDEN by user; watchdog is still armed.")

    return safe_config, guard_config, estimate


def _log_resume_hint(state, log_cb):
    """Tell the user what survived, so a retry is obviously worth starting."""
    kept = [STAGE_LABELS[name] for name in STAGE_LABELS if name in state.stages]
    if not kept:
        return
    log_cb("[Resume] These stages finished and were kept in output/temp:")
    for item in kept:
        log_cb(f"  - {item}")
    log_cb("[Resume] Start the run again with the same settings and it will "
           "pick up from here instead of redoing them.")


def run_vr_pipeline(source_face, target_video, output_path, config, layout="sbs", fps=None,
                    trim_start=None, trim_end=None, merge_encoder=None, merge_crf=None,
                    merge_preset=None, keep_temp=False, progress_cb=None, log_cb=None,
                    guard=None, force=False, prepared=None, refresh_source_faces=False,
                    resume=True, temp_dir=None, presplit=None, defer_merge=False,
                    on_split_done=None):
    """
    The whole VR run: split, swap each eye, optionally interpolate, merge, tag.

    Three parameters exist for the chunked pipeline and are None/False for
    every other caller:

      * `temp_dir` - where the intermediates and the resume checkpoint live.
        A chunked run gives each chunk its own directory (output/temp/chunk_003)
        so that chunk 4's split can be written while chunk 3's eyes are still
        on disk waiting to be merged. Default is output/temp, exactly as before.
      * `presplit` - (left, right) for a split that has already been done, by
        the chunked pipeline, underneath the previous chunk's swap. The files
        are where this function would have written them; all that is left is
        the checkpoint bookkeeping.
      * `defer_merge` - stop after the swaps and hand the two eyes back to the
        caller instead of merging them, so the merge can run underneath the
        next chunk's swap. Returns the handoff dict that
        finish_deferred_merge() consumes, not a path.
      * `on_split_done` - called once, with no arguments, the moment this
        chunk's split is finished and the swap is about to start. That is the
        earliest moment at which starting the NEXT chunk's split is free: both
        a split and a swap want the GPU, but a split wants NVENC and a swap
        wants the CUDA cores, whereas two splits want the same single encoder
        and simply queue behind each other.
    """
    temp_dir = temp_dir or os.path.join(PROJECT_DIR, "output", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_DIR, "output", "final"), exist_ok=True)

    config, guard_config, estimate = prepared or prepare_run(
        target_video, config, is_vr=True, trim_start=trim_start, trim_end=trim_end,
        log_cb=log_cb, temp_dir=temp_dir, force=force,
    )
    # Reuse the caller's guard if it owns one, otherwise own it for this run.
    owns_guard = guard is None
    if defer_merge and owns_guard:
        # The merge happens after this function returns, so a guard created and
        # closed in here would take the merge's ffmpeg down with it.
        raise PipelineError("defer_merge needs the caller to own the ResourceGuard; "
                            "run_chunked_pipeline is the only caller that does.")
    guard = guard or ResourceGuard(temp_dir, guard_config, log_cb=log_cb).start()

    # Owns the progress bar from here on. Every stage asks it where its slice
    # of the bar is rather than carrying a hardcoded percentage, so the bar
    # tracks time and the remaining figure means something.
    clock = RunClock(estimate.get("runtime"), progress_cb=progress_cb, log_cb=log_cb,
                     paused_seconds=guard.paused_seconds)

    # The merge step used to hardcode libx265/preset medium, which pinned every
    # core for hours regardless of what the user had configured.
    merge_encoder = merge_encoder or config.get("encoder", "hevc_nvenc")
    # The merge is the last encode the output goes through, and it used to be
    # pinned at cq 20 regardless of the Quality setting - so asking for 95 in
    # the GUI bought a better FaceFusion intermediate and then threw the
    # difference away here. Track the same setting instead.
    if merge_crf is None:
        merge_crf = quality_to_cq(config.get("quality", 80))

    state = RunState(temp_dir, log_cb=log_cb)
    # `presplit` means the caller has just written left.mp4 and right.mp4 into
    # this directory on purpose. Wiping them here because resume is off would
    # delete work that was done for this attempt, not work left over from a
    # previous one - the chunked pipeline has already cleared what resume=False
    # is meant to clear.
    if not resume and not presplit:
        # Unticking resume throws away finished eyes, and a finished eye is the
        # better part of an hour of GPU time. It should never happen silently.
        if state.stages and log_cb:
            log_cb("[Resume] OFF - discarding finished work from the last attempt: "
                   + ", ".join(STAGE_LABELS.get(n, n) for n in state.stages))
        state.clear()
        cleanup_temp(temp_dir)
    succeeded = False

    clock.progress(0, "Initializing VR Deepfake Pipeline")
    if log_cb:
        log_cb("=" * 60)
        log_cb("VR DEEPFAKE PIPELINE STARTED")
        log_cb(f"  Source Face: {source_face}")
        log_cb(f"  Target Video: {target_video}")
        log_cb(f"  Layout: {layout.upper()} | Output: {output_path}")
        log_cb("=" * 60)

    try:
        # Resolved once, not once per eye: both eyes have to be swapped against
        # exactly the same identity, or the two swaps disagree and the face
        # shimmers in the headset.
        source_face = resolve_source_faces(
            source_face, config, log_cb=log_cb, guard=guard,
            progress_cb=lambda p, t: clock.progress(int(p * clock.PRELUDE_PCT / 100.0), t),
            refresh=refresh_source_faces,
        )

        left_video = os.path.join(temp_dir, "left.mp4")
        right_video = os.path.join(temp_dir, "right.mp4")
        if presplit:
            left_video, right_video = presplit
        left_out = os.path.join(temp_dir, "left_swapped.mp4")
        right_out = os.path.join(temp_dir, "right_swapped.mp4")
        left_temp = os.path.join(temp_dir, "ff_temp_left")
        right_temp = os.path.join(temp_dir, "ff_temp_right")

        keys, parts = vr_stage_keys(
            split_sources(source_face), target_video, config, layout, fps,
            trim_start, trim_end, merge_encoder, merge_crf, merge_preset,
        )
        # Worked out before anything runs, so the log can say up front what this
        # attempt is actually going to do.
        have = {
            name: resume and state.reusable(name, path, keys[name])
            for name, path in (
                ("split_left", left_video), ("split_right", right_video),
                ("swap_left", left_out), ("swap_right", right_out),
            )
        }
        # The checkpoint says the stage finished and the file is still the
        # length it was then. This says the file is the right eye of the
        # right clip - see describe_wrong_eye() for the afternoon it cost.
        for name, reason in _wrong_reused_eyes(
                have, {"split_left": left_video, "split_right": right_video,
                       "swap_left": left_out, "swap_right": right_out},
                target_video, layout, trim_start, trim_end, config).items():
            have[name] = False
            state.forget(name)
            if log_cb:
                log_cb(f"[Resume] Redoing the {STAGE_LABELS.get(name, name)}: {reason}.")
        if any(have.values()) and log_cb:
            log_cb("[Resume] Reusing finished work from the last attempt:")
            for name in ("split_left", "split_right", "swap_left", "swap_right"):
                if have[name]:
                    log_cb(f"  - {STAGE_LABELS[name]}")
        # A stage that was finished and is being redone anyway is the case that
        # makes resume look like it never worked. Name it and name the cause.
        if resume:
            for name, path in (("split_left", left_video), ("split_right", right_video),
                               ("swap_left", left_out), ("swap_right", right_out)):
                if not have[name]:
                    _log_resume_miss(state, name, path, keys[name], parts[name], log_cb)
            # A record that cannot be reused is not merely useless, it is a
            # trap. It names a file this attempt is about to overwrite, so an
            # abort part-way through that rewrite left the record standing
            # beside a half-written file - and the failure hint then told the
            # user that a stage they had just watched being interrupted had
            # "finished and was kept". Drop them before anything is written.
            stale = [
                name for name in list(state.stages)
                if state.why_not(
                    name,
                    os.path.join(temp_dir, state.stages[name].get("artifact") or ""),
                    keys.get(name), parts.get(name)) is not None
            ]
            if stale:
                state.forget(*stale)

        if presplit:
            # The split for this chunk ran under the previous chunk's face
            # swap. The files are already here, so all that is left is the
            # bookkeeping the split itself would have done - and doing it means
            # a resume after a crash reuses this split rather than redoing it.
            for eye, path in (("left", left_video), ("right", right_video)):
                if not have["swap_" + eye] and os.path.isfile(path):
                    state.complete("split_" + eye, path, keys["split_" + eye],
                                   parts["split_" + eye])
                    have["split_" + eye] = True
            if log_cb:
                log_cb("[Stage 1/5] The split for this chunk already ran underneath the "
                       "previous chunk's face swap - nothing to wait for here.")

        # An eye whose swap already finished needs no split at all - the split
        # is deleted the moment the swap that consumes it is done.
        split_skip = {eye for eye in ("left", "right")
                      if have["split_" + eye] or have["swap_" + eye]}
        if len(split_skip) == 2:
            clock.skip("split")
        else:
            clock.begin("split")
            if log_cb:
                log_cb("[Stage 1/5] Splitting stereoscopic 3D video into Left and "
                       "Right eye streams...")

        split_start, split_end = clock.range("split")
        split_video(
            target_video, temp_dir, layout, fps, trim_start, trim_end, log_cb,
            guard=guard, encoder=config.get("encoder", "h264_nvenc"), guard_config=guard_config,
            skip=split_skip, progress_cb=clock.progress,
            start_pct=split_start, end_pct=split_end,
        )
        if len(split_skip) < 2:
            # Half a split is not a sample of what a split costs, so it only
            # teaches the estimator anything when both eyes really ran.
            clock.done("split", record=not split_skip)
        for eye, path in (("left", left_video), ("right", right_video)):
            if not have["split_" + eye] and not have["swap_" + eye]:
                state.complete("split_" + eye, path, keys["split_" + eye], parts["split_" + eye])
                have["split_" + eye] = True

        # This chunk has its eyes and is about to start swapping, which leaves
        # the encoder idle - so this, and not a moment earlier, is when the
        # next chunk's split can start for free.
        if on_split_done:
            on_split_done()

        # -- the face swap ------------------------------------------------
        #
        # Either one eye after the other, which is what this has always done,
        # or both at once in two guard lanes when parallel_eyes_verdict() has
        # found room on the card for two FaceFusion processes. The verdict is
        # taken at preflight (so the time estimate describes the right run) and
        # again here, because VRAM belongs to the whole machine and a split can
        # take minutes - something else may have taken the card in between.
        eyes = {
            "left": {"target": left_video, "out": left_out, "temp": left_temp,
                     "label": "Left Eye", "stage": "[Stage 2/5]"},
            "right": {"target": right_video, "out": right_out, "temp": right_temp,
                      "label": "Right Eye", "stage": "[Stage 3/5]"},
        }
        todo = [eye for eye in ("left", "right") if not have["swap_" + eye]]

        def retire_eye(eye):
            """An eye's frame dump and its split are dead the moment it renders."""
            if keep_temp:
                return
            cleanup_temp(temp_dir, ["ff_temp_" + eye, os.path.basename(eyes[eye]["target"])])
            state.forget("split_" + eye)

        def swap_eye(eye, eye_config, start_pct, end_pct, lane=None, progress=None):
            info = eyes[eye]
            run_facefusion(
                source_face, info["target"], info["out"], info["temp"], eye_config,
                log_cb=log_cb, progress_cb=progress or clock.progress,
                start_pct=start_pct, end_pct=end_pct, stage_name=info["label"],
                guard=guard, guard_config=guard_config, lane=lane,
            )
            state.complete("swap_" + eye, info["out"], keys["swap_" + eye],
                           parts["swap_" + eye])
            retire_eye(eye)

        # The estimate decided which shape the progress bar has: one combined
        # "swap_both" stage, or one per eye. Follow it rather than fighting it -
        # a stage the clock has never heard of owns no part of the bar.
        both_stage = "swap_both" if "swap_both" in clock.stages else None
        parallel = bool((estimate.get("parallel_eyes") or {}).get("ok")) and len(todo) == 2

        if parallel:
            pass_pixels = (estimate.get("width", 0) * estimate.get("height", 0)) / 2.0
            ok_now, reason_now = guardrails.parallel_eyes_verdict(
                config, guard_config, pass_pixels)
            if not ok_now:
                parallel = False
                if log_cb:
                    log_cb(f"[Guard] Both eyes at once was going to be used, but the "
                           f"machine has changed since the preflight: {reason_now}. "
                           f"Running the eyes one after the other.")

        for eye in ("left", "right"):
            if have["swap_" + eye]:
                if log_cb:
                    log_cb(f"{eyes[eye]['stage']} {eyes[eye]['label']} already swapped "
                           f"by an earlier attempt - skipping.")
                retire_eye(eye)
        if not todo:
            clock.skip(both_stage or "swap_left")
            if not both_stage:
                clock.skip("swap_right")
        elif not both_stage:
            for eye in ("left", "right"):
                if have["swap_" + eye]:
                    clock.skip("swap_" + eye)

        if parallel:
            lane_threads = guardrails.parallel_eyes_lane_threads(config)
            lane_config = dict(config, threads=lane_threads)
            stage = both_stage or "swap_left"
            clock.begin(stage)
            start_pct, end_pct = clock.range(stage)
            span = max(1, end_pct - start_pct)
            if log_cb:
                log_cb(f"[Stage 2/5] Running BOTH eyes at once, one FaceFusion process "
                       f"each at {lane_threads} execution threads. "
                       f"{(estimate.get('parallel_eyes') or {}).get('reason', '')}")
                log_cb("  Either lane can be frozen on its own if the machine runs "
                       "short; the other one keeps the GPU.")

            fractions = {"left": 0.0, "right": 0.0}
            progress_lock = threading.Lock()

            def lane_progress(eye):
                def report(pct, text=""):
                    # The whole thing under one lock: RunClock keeps mutable
                    # state (the bar's high-water mark, the current stage's
                    # fraction) that was written for one caller at a time, and
                    # two lanes reporting a frame at the same instant is the
                    # one place in this pipeline where that is not true.
                    with progress_lock:
                        fractions[eye] = max(0.0, min(1.0, (pct - start_pct) / float(span)))
                        left_pct = fractions["left"] * 100
                        right_pct = fractions["right"] * 100
                        mean = (fractions["left"] + fractions["right"]) / 2.0
                        clock.progress(
                            int(start_pct + mean * span),
                            f"Both eyes: left {left_pct:.0f}%, right {right_pct:.0f}%")
                return report

            failures = {}

            def lane_worker(eye):
                try:
                    swap_eye(eye, lane_config, start_pct, end_pct, lane=eye,
                             progress=lane_progress(eye))
                except BaseException as exc:      # re-raised on the main thread
                    failures[eye] = exc

            workers = [threading.Thread(target=lane_worker, args=(eye,),
                                        name="vrdeep-swap-" + eye, daemon=True)
                       for eye in ("left", "right")]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()

            guard.check()
            if len(failures) == 2:
                # Both lanes died. That is not a lane problem, it is the run's
                # problem, and it goes up the normal path.
                raise list(failures.values())[0]
            if failures:
                eye, exc = next(iter(failures.items()))
                if log_cb:
                    log_cb(f"[Guard] The {eye} lane did not survive running alongside "
                           f"the other eye: {exc}")
                    log_cb(f"[Guard] The {'left' if eye == 'right' else 'right'} eye "
                           f"finished. Re-running the {eye} eye on its own with the full "
                           f"{int(config.get('threads') or 3)} threads - this machine "
                           f"could not hold two lanes at these settings.")
                # Whatever the lane left behind is a partial encode; the retry
                # overwrites it, and the checkpoint for it was never written.
                swap_eye(eye, config, start_pct, end_pct)
            clock.done(stage)
        else:
            stage = both_stage
            if stage and todo:
                clock.begin(stage)
                low, high = clock.range(stage)
            for index, eye in enumerate(todo):
                if log_cb:
                    log_cb(f"{eyes[eye]['stage']} Running FaceFusion AI processing on "
                           f"{eyes[eye]['label']} stream...")
                if stage:
                    # The plan expected one combined stage and only one eye is
                    # left (or the verdict changed under us). Give each eye its
                    # share of the combined stage's slice rather than leaving
                    # the bar still.
                    width = (high - low) / float(len(todo))
                    start_pct = int(low + index * width)
                    end_pct = int(low + (index + 1) * width)
                else:
                    clock.begin("swap_" + eye)
                    start_pct, end_pct = clock.range("swap_" + eye)
                swap_eye(eye, config, start_pct, end_pct)
                if not stage:
                    clock.done("swap_" + eye)
            if stage and todo:
                # Only a stage that really ran both eyes is a sample of what
                # both eyes together cost.
                clock.done(stage, record=len(todo) == 2)

        # Frame interpolation, per eye, before the merge. It has to be here
        # rather than after: RIFE needs ~8.7 GB of VRAM on a 4096x4096 eye and
        # the merged 8192x4096 frame is four times the pixels, which does not
        # fit on this card at all.
        interp_multi = int(config.get("interpolate_multi") or 1)
        if interp_multi > 1:
            for eye, swapped in (("left", left_out), ("right", right_out)):
                target = os.path.join(temp_dir, f"{eye}_interp.mp4")
                stage = "interp_" + eye
                if resume and state.reusable(stage, target, keys[stage]):
                    if log_cb:
                        log_cb(f"[Stage 3.5/5] {eye.title()} eye interpolation from an "
                               f"earlier attempt is intact - skipping.")
                    clock.skip(stage)
                else:
                    _log_resume_miss(state, stage, target, keys[stage],
                                     parts[stage], log_cb)
                    clock.begin(stage)
                    start_pct, end_pct = clock.range(stage)
                    interpolate_video(
                        swapped, target, multi=interp_multi,
                        infer_scale=float(config.get("interpolate_scale") or 0.75),
                        encoder=config.get("encoder", "hevc_nvenc"),
                        crf=quality_to_cq(config.get("quality", 80)),
                        log_cb=log_cb, progress_cb=clock.progress, guard=guard,
                        guard_config=guard_config,
                        stage_name=f"{eye.title()} Eye {interp_multi}x",
                        start_pct=start_pct, end_pct=end_pct,
                    )
                    clock.done(stage)
                    state.complete(stage, target, keys[stage], parts[stage])
                # The swapped eye is consumed by this and nothing else.
                if not keep_temp and os.path.exists(swapped):
                    os.remove(swapped)
                    state.forget("swap_" + eye)
                if eye == "left":
                    left_out = target
                else:
                    right_out = target

        # Same shape, same length, or nothing downstream can use them. Done
        # here, before the handoff, so a wrong eye fails now and is named -
        # not ninety minutes later as an NVENC error from a background merge.
        check_eyes_match(
            left_out, right_out,
            expected_seconds=expected_eye_geometry(target_video, layout, trim_start, trim_end)[2],
            log_cb=log_cb)

        if defer_merge:
            # B9: stop here and let the caller merge these two eyes underneath
            # the NEXT chunk's swap. Nothing is cleaned up and nothing is
            # marked finished - the run is genuinely not finished, and the
            # `finally` below only drops the frame dumps, which are of no use
            # to the merge or to a resume.
            clock.progress(100, "Both eyes swapped - merging in the background")
            if log_cb:
                log_cb("[Stage 4/5] Merge handed to the background so the next chunk "
                       "can start its face swap now.")
            return {
                "left": left_out, "right": right_out,
                "temp_dir": temp_dir, "output_path": output_path,
                "layout": layout, "fps": fps,
                "merge_encoder": merge_encoder, "merge_crf": merge_crf,
                "merge_preset": merge_preset,
                "keep_temp": keep_temp, "state": state,
                "merge_key": keys["merge"], "merge_parts": parts["merge"],
                "guard_config": guard_config,
            }

        # Match the output container, so the final move is a move and not a
        # rename into a format the file is not actually in.
        out_ext = os.path.splitext(output_path)[1].lower() or ".mp4"
        merged_path = os.path.join(temp_dir, "merged_vr" + out_ext)
        if resume and state.reusable("merge", merged_path, keys["merge"]):
            if log_cb:
                log_cb("[Stage 4/5] Merged file from an earlier attempt is intact - skipping.")
            clock.skip("merge")
        else:
            _log_resume_miss(state, "merge", merged_path, keys["merge"],
                             parts["merge"], log_cb)
            clock.begin("merge")
            if log_cb:
                log_cb("[Stage 4/5] Merging processed Left and Right eye videos with "
                       "VR spatial metadata...")
            start_pct, end_pct = clock.range("merge")
            merge_videos(
                left_out, right_out, merged_path, layout, merge_encoder, merge_crf,
                merge_preset, True, fps, log_cb, guard=guard, guard_config=guard_config,
                progress_cb=clock.progress, start_pct=start_pct, end_pct=end_pct,
            )
            clock.done("merge")
            check_merge_frame_counts(left_out, right_out, merged_path, log_cb)
            state.complete("merge", merged_path, keys["merge"], parts["merge"])

        current_output = merged_path

        if config.get("temporal_smoothing"):
            smoothed_output = os.path.join(temp_dir, "smoothed_vr" + out_ext)
            if resume and state.reusable("smooth", smoothed_output, keys["smooth"]):
                if log_cb:
                    log_cb("[Stage 5/6] Smoothed file from an earlier attempt is intact - skipping.")
                clock.skip("smooth")
            else:
                _log_resume_miss(state, "smooth", smoothed_output, keys["smooth"],
                                 parts["smooth"], log_cb)
                clock.begin("smooth")
                if log_cb:
                    log_cb("[Stage 5/6] Applying temporal smoothing to reduce flickering...")
                apply_temporal_smoothing(
                    current_output, smoothed_output,
                    encoder=config.get("encoder", "hevc_nvenc"),
                    crf=quality_to_cq(config.get("quality", 80)),
                    log_cb=log_cb, guard=guard, guard_config=guard_config,
                )
                clock.done("smooth")
                state.complete("smooth", smoothed_output, keys["smooth"], parts["smooth"])
            if not keep_temp:
                os.remove(current_output)
                state.forget("merge")
            current_output = smoothed_output

        if config.get("upscale_method"):
            clock.begin("upscale")
            if log_cb:
                log_cb(f"[Stage 6/6] Upscaling merged video via {config['upscale_method']}...")
            upscale_out = output_path
            start_pct, end_pct = clock.range("upscale")
            upscale_video(
                current_output, upscale_out,
                method=config["upscale_method"],
                scale=config.get("upscale_scale", 2),
                target_width=config.get("upscale_target_width"),
                encoder=config.get("encoder", "hevc_nvenc"),
                crf=quality_to_cq(config.get("quality", 80)),
                log_cb=log_cb, guard=guard,
                progress_cb=lambda pct, text="": clock.progress(
                    int(start_pct + (pct / 100.0) * (end_pct - start_pct)), text),
                tile=guard_config.get("esrgan_tile", 512),
                guard_config=guard_config,
            )
            clock.done("upscale")
            if not keep_temp:
                os.remove(current_output)
        else:
            shutil.move(current_output, output_path)

        # Last, because smoothing and upscaling each re-encode into a new file
        # and would drop the tag.
        apply_stereo_metadata(output_path, layout, log_cb=log_cb, guard=guard)

        succeeded = True

        if not keep_temp:
            if log_cb:
                log_cb("Cleaning up temporary processing files...")
            cleanup_temp(temp_dir)

        clock.progress(100, "VR Deepfake Pipeline completed successfully!")
        clock.finish()
        if log_cb:
            log_cb("[SUCCESS] VR Deepfake Pipeline finished!")

        return output_path

    except GuardAbort as e:
        if log_cb:
            log_cb(f"[STOPPED] {e}")
            _log_resume_hint(state, log_cb)
        raise PipelineError(str(e))

    except Exception as e:
        if log_cb:
            log_cb(f"[ERROR] VR Pipeline Failed: {str(e)}")
            _log_resume_hint(state, log_cb)
        raise

    finally:
        # Only a finished run clears temp. A failed one leaves its completed
        # stages on disk on purpose: that is what the next attempt resumes
        # from, and re-swapping an eye that already rendered is an hour of GPU
        # time thrown away. The frame dumps, which are the part that actually
        # fills the drive, are dropped either way - they belong to the stage
        # that was interrupted and are of no use to anyone.
        if not keep_temp:
            if succeeded:
                cleanup_temp(temp_dir)
            else:
                cleanup_temp(temp_dir, ["ff_temp_left", "ff_temp_right", "ff_temp_normal",
                                        "upscale_frames"])
        if owns_guard:
            guard.close()


def finish_deferred_merge(handoff, guard=None, log_cb=None, progress_cb=None):
    """
    The tail of a VR chunk whose merge was deferred: merge, tag, verify, clean.

    Runs on a worker thread while the *next* chunk's face swap has the GPU.
    That is the whole point of it - the merge is NVENC and a little CPU, the
    swap is CUDA inference, and at 8192x4096 the merge is ~8 minutes of every
    84-minute chunk that nobody has to wait for any more.

    Everything it touches lives in this chunk's own temp directory
    (output/temp/chunk_NNN), which is why those directories exist: the next
    chunk is writing left.mp4 and right.mp4 of its own at the same moment, and
    before B9 that was the same file.
    """
    guard = guard or NULL_GUARD
    guard.check()
    state = handoff["state"]
    temp_dir = handoff["temp_dir"]
    output_path = handoff["output_path"]
    left_path, right_path = handoff["left"], handoff["right"]

    out_ext = os.path.splitext(output_path)[1].lower() or ".mp4"
    merged_path = os.path.join(temp_dir, "merged_vr" + out_ext)

    # Cheap, and the last chance to say which eye is wrong before an
    # eight-minute encode fails with an error about the encoder.
    check_eyes_match(left_path, right_path)

    merge_videos(
        left_path, right_path, merged_path, handoff["layout"],
        handoff["merge_encoder"], handoff["merge_crf"], handoff["merge_preset"],
        True, handoff["fps"], log_cb, guard=guard,
        guard_config=handoff["guard_config"], progress_cb=progress_cb, lane="merge",
        # The merge owns no part of the bar when it is overlapped; the chunk
        # bar belongs to whatever is running in front of it.
        start_pct=0, end_pct=0,
    )
    # The eyes and the merged file must hold the same number of frames. This is
    # the check that would catch an overlap bug - a merge reading an eye that
    # something else is still writing - and it is cheap enough to always do.
    check_merge_frame_counts(left_path, right_path, merged_path, log_cb)
    state.complete("merge", merged_path, handoff["merge_key"], handoff["merge_parts"])

    guard.check()
    shutil.move(merged_path, output_path)
    # Last, because it lives in the container header and the move is the last
    # thing to rewrite one.
    apply_stereo_metadata(output_path, handoff["layout"], log_cb=log_cb, guard=guard)
    if not handoff["keep_temp"]:
        cleanup_temp(temp_dir)
    return output_path


def run_normal_pipeline(source_face, target_video, output_path, config, fps=None,
                        trim_start=None, trim_end=None, keep_temp=False,
                        progress_cb=None, log_cb=None, guard=None, force=False, prepared=None,
                        refresh_source_faces=False):
    temp_dir = os.path.join(PROJECT_DIR, "output", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_DIR, "output", "final"), exist_ok=True)

    config, guard_config, estimate = prepared or prepare_run(
        target_video, config, is_vr=False, trim_start=trim_start, trim_end=trim_end,
        log_cb=log_cb, temp_dir=temp_dir, force=force,
    )
    owns_guard = guard is None
    guard = guard or ResourceGuard(temp_dir, guard_config, log_cb=log_cb).start()

    # See run_vr_pipeline: the clock owns the bar and the remaining-time figure.
    clock = RunClock(estimate.get("runtime"), progress_cb=progress_cb, log_cb=log_cb,
                     paused_seconds=guard.paused_seconds)

    # The VR path hands fps to split/merge, but this path has neither, so the
    # override has to reach FaceFusion itself. Without this the Output FPS box
    # in the Normal tab was read from the GUI and then silently dropped.
    if fps:
        config = dict(config)
        config["fps"] = fps

    clock.progress(0, "Initializing Normal Deepfake Pipeline")
    if log_cb:
        log_cb("=" * 60)
        log_cb("NORMAL DEEPFAKE PIPELINE STARTED")
        log_cb(f"  Source Face: {source_face}")
        log_cb(f"  Target Video: {target_video}")
        log_cb(f"  Output: {output_path}")
        log_cb("=" * 60)

    try:
        # A source clip becomes stills here, before anything else runs, so a
        # clip with no usable face fails in seconds rather than after the trim.
        source_face = resolve_source_faces(
            source_face, config, log_cb=log_cb, guard=guard,
            progress_cb=lambda p, t: clock.progress(int(p * clock.PRELUDE_PCT / 100.0), t),
            refresh=refresh_source_faces,
        )

        # A still photo is a legitimate 2D swap target, but trimming, fps,
        # temporal smoothing and the video encoder settings are all meaningless
        # for one - and FaceFusion decides image vs video from the *output*
        # extension, so ff_result.mp4 would silently turn a jpg into a 1-frame
        # video.
        is_image = is_image_path(target_video)
        if is_image:
            if log_cb:
                log_cb("Target is a still image; trim, fps and smoothing do not apply.")
            trim_start = trim_end = None

        trimmed_target = None
        if trim_start is not None or trim_end is not None:
            trimmed_target = os.path.join(temp_dir, "trimmed_target.mp4")
            # See split_video: one end on its own is a valid range, and used to
            # be ignored entirely.
            span = []
            if trim_start is not None:
                span += ["-ss", str(trim_start)]
            if trim_end is not None:
                span += ["-to", str(trim_end)]
            if log_cb:
                log_cb(f"[Stage 1/4] Trimming video clip: "
                       f"{trim_start if trim_start is not None else 0}s - "
                       f"{str(trim_end) + 's' if trim_end is not None else 'end'}...")
            cmd = (
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
                + span + _decode_args(guard_config) + ["-i", target_video]
                # Same role as the VR split: this is the *input* to the swap,
                # so match its quality rather than the old blanket 18.
                + _encoder_args(config.get("encoder", "h264_nvenc"), 16,
                                guard_config, preset="p6")
                + ["-c:a", "aac", "-b:a", "192k", trimmed_target]
            )
            result = guard.run(cmd, capture_output=True)
            if result.returncode != 0:
                guard.check()
                raise PipelineError(f"Trim failed: {(result.stderr or b'').decode(errors='replace')[:400]}")
            target_for_ff = trimmed_target
        else:
            target_for_ff = target_video

        if log_cb:
            log_cb("[Stage 2/4] Executing FaceFusion AI (Face Swapping + Enhancement)...")

        ff_temp = os.path.join(temp_dir, "ff_temp_normal")
        if is_image:
            suffix = os.path.splitext(output_path)[1].lower()
            if suffix not in IMAGE_EXTS:
                suffix = os.path.splitext(target_video)[1].lower()
                output_path = os.path.splitext(output_path)[0] + suffix
                if log_cb:
                    log_cb(f"Output renamed to {os.path.basename(output_path)} to match the image target.")
            ff_output = os.path.join(temp_dir, "ff_result" + suffix)
        else:
            ff_output = os.path.join(temp_dir, "ff_result.mp4")

        clock.begin("swap")
        start_pct, end_pct = clock.range("swap")
        run_facefusion(
            source_face, target_for_ff, ff_output, ff_temp, config,
            log_cb=log_cb, progress_cb=clock.progress,
            start_pct=start_pct, end_pct=end_pct, stage_name="FaceFusion AI", guard=guard,
            guard_config=guard_config
        )
        clock.done("swap")

        current_output = ff_output

        # Frame dumps are the bulk of the temp usage; drop them as soon as
        # FaceFusion has rendered its result.
        if not keep_temp:
            cleanup_temp(temp_dir, ["ff_temp_normal"])

        if config.get("temporal_smoothing") and not is_image:
            clock.begin("smooth")
            if log_cb:
                log_cb("[Stage 3/5] Applying temporal smoothing to reduce flickering...")
            smoothed_output = os.path.join(temp_dir, "smoothed_result.mp4")
            apply_temporal_smoothing(
                current_output, smoothed_output,
                encoder=config.get("encoder", "hevc_nvenc"),
                crf=quality_to_cq(config.get("quality", 80)),
                log_cb=log_cb, guard=guard, guard_config=guard_config,
            )
            # Smoothing reads current_output and writes a new file, so the
            # input is dead either way - including when it is FaceFusion's own
            # render, which is the usual case and which the old `!= ff_output`
            # test excluded, leaving a full-size encode behind in temp.
            if not keep_temp:
                try:
                    os.remove(current_output)
                except OSError:
                    pass
            clock.done("smooth")
            current_output = smoothed_output

        if config.get("upscale_method") and is_image and log_cb:
            log_cb("Upscaling is skipped for image targets; use the frame enhancer instead.")

        if config.get("upscale_method") and not is_image:
            clock.begin("upscale")
            if log_cb:
                log_cb(f"[Stage 4/5] Upscaling output video via {config['upscale_method']}...")
            start_pct, end_pct = clock.range("upscale")
            upscale_video(
                current_output, output_path,
                method=config["upscale_method"],
                scale=config.get("upscale_scale", 2),
                target_width=config.get("upscale_target_width"),
                encoder=config.get("encoder", "hevc_nvenc"),
                crf=quality_to_cq(config.get("quality", 80)),
                log_cb=log_cb, guard=guard,
                progress_cb=lambda pct, text="": clock.progress(
                    int(start_pct + (pct / 100.0) * (end_pct - start_pct)), text),
                tile=guard_config.get("esrgan_tile", 512),
                guard_config=guard_config,
            )
            clock.done("upscale")
            if not keep_temp:
                os.remove(current_output)
        else:
            shutil.move(current_output, output_path)

        if not keep_temp:
            if log_cb:
                log_cb("[Stage 5/5] Cleaning up temporary working files...")
            cleanup_temp(temp_dir, ["ff_temp_normal", "trimmed_target.mp4",
                                    "smoothed_result.mp4", os.path.basename(ff_output)])

        clock.progress(100, "Normal Deepfake Pipeline completed successfully!")
        clock.finish()
        if log_cb:
            log_cb("[SUCCESS] Normal Deepfake Pipeline finished!")

        return output_path

    except GuardAbort as e:
        if log_cb:
            log_cb(f"[STOPPED] {e}")
        raise PipelineError(str(e))

    except Exception as e:
        if log_cb:
            log_cb(f"[ERROR] Pipeline Failed: {str(e)}")
        raise

    finally:
        if not keep_temp:
            cleanup_temp(temp_dir, ["ff_temp_normal", "trimmed_target.mp4"])
        if owns_guard:
            guard.close()


# ---------------------------------------------------------------------------
# Chunked runs
#
# A long clip used to be one run: one split, one swap per eye, one merge. A
# failure two hours in - the guard aborting on memory, FaceFusion dying, the
# machine rebooting - kept the finished *stages*, but a stage is an eye of the
# whole clip, so nothing that had been made was watchable until everything was.
#
# Here the selected range is cut into chunks of about a minute and each chunk
# goes through the whole pipeline on its own, landing as a finished, playable
# file in <output>_chunks/ beside the final output. Only when every chunk is
# done are they joined into the requested file and the chunk folder removed.
# A run that dies at chunk 7 of 30 leaves six finished minutes on disk, and
# the next run with the same settings starts at chunk 7 - with the inner
# checkpoints in output/temp still resuming chunk 7 from whichever eye it
# had finished.
#
# Chunking also caps temp usage at one chunk's worth, which is what lets a
# full-length 8K clip start at all on a drive that could never hold the frame
# dumps for the whole thing.
# ---------------------------------------------------------------------------

CHUNK_MANIFEST = "chunks.json"

# A tail shorter than this is folded into the chunk before it: FaceFusion pays
# roughly half a minute of model loading per chunk, and spending it on a 3 s
# remainder is silly.
CHUNK_MIN_TAIL = 10.0


def chunk_plan(duration, trim_start=None, trim_end=None, chunk_seconds=60.0):
    """
    Cut the selected range into (start, end) pairs of about chunk_seconds.

    Returns a single pair covering the whole range when chunking is off or
    the range is not meaningfully longer than one chunk. Boundaries are on the
    source timeline, so they mean the same thing to the split, the trim and
    the final audio mux.
    """
    start = max(0.0, float(trim_start or 0.0))
    end = float(duration) if trim_end is None else min(float(duration), float(trim_end))
    if duration and end <= start:
        end = float(duration)
    span = max(0.0, end - start)
    chunk_seconds = float(chunk_seconds or 0)
    if chunk_seconds <= 0 or span <= chunk_seconds + CHUNK_MIN_TAIL:
        return [(start, end)]

    plan = []
    cursor = start
    while cursor < end:
        nxt = min(end, cursor + chunk_seconds)
        # Fold a short tail into this chunk rather than making it a chunk.
        if end - nxt < CHUNK_MIN_TAIL:
            nxt = end
        plan.append((cursor, nxt))
        cursor = nxt
    return plan


def chunk_dir_for(output_path):
    """Chunks live beside the final file, named after it, so they are findable."""
    stem, _ext = os.path.splitext(output_path)
    return stem + "_chunks"


def chunk_file(chunk_dir, index, ext):
    return os.path.join(chunk_dir, f"part_{index + 1:03d}{ext}")


def _chunk_key(is_vr, source_face, target_video, config, plan, **extra):
    """Everything that decides what a chunk contains. If it moves, the chunks are stale."""
    return _stage_key(
        is_vr=is_vr,
        sources=[_path_stamp(p) for p in split_sources(source_face)],
        target=_path_stamp(target_video),
        config={k: v for k, v in sorted(config.items()) if k != "chunk_seconds"},
        plan=[[round(a, 3), round(b, 3)] for a, b in plan],
        **extra,
    )


def _read_chunk_manifest(chunk_dir):
    try:
        with open(os.path.join(chunk_dir, CHUNK_MANIFEST), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_chunk_manifest(chunk_dir, manifest):
    os.makedirs(chunk_dir, exist_ok=True)
    tmp = os.path.join(chunk_dir, CHUNK_MANIFEST + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(tmp, os.path.join(chunk_dir, CHUNK_MANIFEST))


def chunk_is_complete(path, expected_seconds, tolerance=1.0):
    """
    A chunk counts only if it is on disk at the length it was meant to be.

    The inner pipelines move their result into place as the last step, so a
    file that exists is nearly always whole - but the upscale stage writes to
    the final name directly, and an abort in the middle of that leaves a
    headerless mp4 that ffprobe reads as zero seconds. The length check
    catches it.
    """
    if not os.path.isfile(path):
        return False
    got = media_duration(path)
    return got > 0 and abs(got - expected_seconds) <= tolerance


def _has_audio(path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
           "stream=codec_type", "-of", "csv=p=0", path]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return "audio" in (result.stdout or "")
    except (subprocess.SubprocessError, OSError):
        return False


def _concat_list_entry(path):
    # The concat demuxer wants single-quoted paths with the quote itself
    # escaped; forward slashes keep Windows backslashes out of the escaping.
    quoted = os.path.abspath(path).replace("\\", "/").replace("'", "'\\''")
    return "file '" + quoted + "'\n"


def join_chunks(chunk_paths, output_path, target_video, plan, encoder="hevc_nvenc",
                crf=20, guard=None, guard_config=None, log_cb=None, progress_cb=None,
                start_pct=0, end_pct=0):
    """
    Join finished chunks into the requested output.

    Video is stream-copied: every chunk came out of the same encoder with the
    same settings, so the concat demuxer can splice them without touching a
    pixel, and the join takes seconds rather than another hour of encoding.
    Audio is NOT taken from the chunks. Each chunk's AAC track carries its
    own encoder priming, and splicing thirty of those puts an audible tick at
    every boundary - so the audio is cut from the source in one piece across
    the whole range and muxed in alongside the joined video.

    If the copy splice is refused - which it would be if two chunks somehow
    disagree on codec parameters - it falls back to a single re-encode, and
    says so, because that is slow and the user should know why.
    """
    guard = guard or NULL_GUARD
    guard.check()
    chunk_dir = os.path.dirname(chunk_paths[0])
    list_path = os.path.join(chunk_dir, "join.txt")
    with open(list_path, "w", encoding="utf-8") as fh:
        for path in chunk_paths:
            fh.write(_concat_list_entry(path))

    range_start, range_end = plan[0][0], plan[-1][1]
    out_ext = os.path.splitext(output_path)[1] or ".mp4"
    joining = os.path.splitext(output_path)[0] + ".joining" + out_ext
    dec = _decode_args(guard_config)
    with_audio = _has_audio(target_video)

    def build(copy_video):
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "concat", "-safe", "0", "-i", list_path]
        if with_audio:
            cmd += ["-ss", str(range_start), "-to", str(range_end)] + dec + ["-i", target_video]
        cmd += ["-map", "0:v:0"]
        if with_audio:
            cmd += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"]
        if copy_video:
            cmd += ["-c:v", "copy"]
        else:
            cmd += _encoder_args(encoder, crf, guard_config, preset="p7")
        cmd += ["-movflags", "+faststart", joining]
        return cmd

    if log_cb:
        log_cb(f"[Chunks] Joining {len(chunk_paths)} chunks into "
               f"{os.path.basename(output_path)}...")
    if progress_cb and end_pct > start_pct:
        progress_cb(start_pct, f"Joining {len(chunk_paths)} chunks")

    code, err = _run_ffmpeg_watched(build(copy_video=True), guard)
    if code != 0:
        guard.check()
        if log_cb:
            log_cb("[Chunks] The chunks could not be spliced without re-encoding "
                   f"({err.strip()[-200:] or 'ffmpeg gave no reason'}); "
                   "re-encoding the joined video once instead. This is slow.")
        total_frames = 0
        on_frame = None
        if progress_cb and end_pct > start_pct:
            first = probe_video(chunk_paths[0])
            total_frames = int((range_end - range_start) * first["fps"])

            def on_frame(done, total):
                progress_cb(int(start_pct + min(1.0, done / float(total)) * (end_pct - start_pct)),
                            f"Joining chunks: frame {done}/{total}")

        code, err = _run_ffmpeg_watched(build(copy_video=False), guard,
                                        on_frame=on_frame, total_frames=total_frames)
        if code != 0:
            guard.check()
            if os.path.exists(joining):
                os.remove(joining)
            raise PipelineError(f"Joining the chunks failed: {_ffmpeg_error_excerpt(err)}")

    os.replace(joining, output_path)
    try:
        os.remove(list_path)
    except OSError:
        pass
    if progress_cb and end_pct > start_pct:
        progress_cb(end_pct, "Chunks joined")
    return output_path


def run_chunked_pipeline(is_vr, source_face, target_video, output_path, config,
                         trim_start=None, trim_end=None, keep_temp=False, resume=True,
                         progress_cb=None, log_cb=None, guard=None, force=False,
                         prepared=None, refresh_source_faces=False, **inner_kwargs):
    """
    Run the VR or Normal pipeline over the selected range one chunk at a time.

    Every keyword the inner pipeline takes (layout, fps, merge_* for VR) is
    passed straight through. With chunking off - chunk_seconds 0, a still
    image, or a range no longer than one chunk - this is exactly one inner
    run, so callers never need to choose.
    """
    temp_dir = os.path.join(PROJECT_DIR, "output", "temp")
    os.makedirs(temp_dir, exist_ok=True)

    inner = run_vr_pipeline if is_vr else run_normal_pipeline
    if not is_vr:
        for key in ("layout", "merge_encoder", "merge_crf", "merge_preset"):
            inner_kwargs.pop(key, None)

    chunk_seconds = float(config.get("chunk_seconds") or 0)
    plan = [(trim_start, trim_end)]
    if chunk_seconds > 0 and not is_image_path(target_video):
        plan = chunk_plan(probe_video(target_video).get("duration") or 0.0,
                          trim_start, trim_end, chunk_seconds)

    if len(plan) < 2:
        kwargs = dict(inner_kwargs, trim_start=trim_start, trim_end=trim_end,
                      keep_temp=keep_temp, progress_cb=progress_cb, log_cb=log_cb,
                      guard=guard, force=force, prepared=prepared,
                      refresh_source_faces=refresh_source_faces)
        if is_vr:
            kwargs["resume"] = resume
        return inner(source_face, target_video, output_path, config, **kwargs)

    # The full-range preflight has to run before anything is spawned, whoever
    # the caller is. The GUI has already done it and hands the result over;
    # a bare call gets it here.
    config, guard_config, estimate = prepared or prepare_run(
        target_video, config, is_vr=is_vr, trim_start=trim_start, trim_end=trim_end,
        log_cb=log_cb, temp_dir=temp_dir, force=force,
    )
    owns_guard = guard is None
    guard = guard or ResourceGuard(temp_dir, guard_config, log_cb=log_cb).start()
    paused_clock = getattr(guard, "paused_seconds", None) or (lambda: 0.0)

    out_ext = os.path.splitext(output_path)[1].lower() or ".mp4"
    chunk_dir = chunk_dir_for(output_path)
    n = len(plan)
    total_span = sum(b - a for a, b in plan)
    key = _chunk_key(is_vr, source_face, target_video, config, plan,
                     **{k: inner_kwargs.get(k) for k in ("layout", "fps", "merge_encoder",
                                                        "merge_crf", "merge_preset")})

    # Chunks from an earlier attempt are only worth anything if they were made
    # from the same inputs with the same settings. Check that before trusting
    # a single file, and say what is being thrown away when it is not.
    manifest = _read_chunk_manifest(chunk_dir)
    if os.path.isdir(chunk_dir):
        if not resume:
            if log_cb:
                log_cb("[Chunks] Resume is OFF - discarding the chunks from the last attempt.")
            shutil.rmtree(chunk_dir, ignore_errors=True)
            manifest = None
        elif not manifest or manifest.get("key") != key:
            if log_cb:
                log_cb(f"[Chunks] The chunks in {chunk_dir} were made with different "
                       "inputs or settings (or no record of which) - discarding them.")
            shutil.rmtree(chunk_dir, ignore_errors=True)
            manifest = None
    manifest = manifest or {}
    manifest.update({
        "key": key, "target": target_video, "output": output_path,
        "range": [plan[0][0], plan[-1][1]], "chunk_seconds": chunk_seconds,
        "chunks": [[a, b] for a, b in plan], "done": sorted(set(manifest.get("done", []))),
    })
    _write_chunk_manifest(chunk_dir, manifest)

    paths = [chunk_file(chunk_dir, i, out_ext) for i in range(n)]
    finished = [i for i, (a, b) in enumerate(plan)
                if chunk_is_complete(paths[i], b - a, tolerance=max(1.0, (b - a) * 0.05))]
    # A file that is missing or short is not done whatever the manifest says.
    manifest["done"] = finished
    _write_chunk_manifest(chunk_dir, manifest)

    if log_cb:
        log_cb("=" * 60)
        log_cb(f"CHUNKED RUN: {n} chunks of ~{guardrails.fmt_duration(chunk_seconds)} "
               f"covering {guardrails.fmt_duration(total_span)} "
               f"({guardrails.fmt_duration(plan[0][0])} to {guardrails.fmt_duration(plan[-1][1])})")
        log_cb(f"  Chunks are written to: {chunk_dir}")
        log_cb("  Each finished chunk is a complete, playable video of its own range. "
               "If the run stops, the finished chunks stay and the next run continues "
               "from the first unfinished one.")
        if finished:
            log_cb(f"[Chunks] {len(finished)} of {n} chunks already finished by an "
                   f"earlier attempt: {', '.join(str(i + 1) for i in finished)}")
        log_cb("=" * 60)

    # Overall pace, measured on this run. The inner clock only knows its own
    # chunk, and what the user wants to know is when the whole thing is done.
    run_started = time.time()
    paused_at_start = paused_clock()
    done_span = 0.0
    plan_total = float(((estimate or {}).get("runtime") or {}).get("total_seconds") or 0)

    def overall_left():
        elapsed = time.time() - run_started - (paused_clock() - paused_at_start)
        if done_span > 0:
            return max(0.0, (total_span - done_span) * elapsed / done_span)
        if plan_total > 0:
            return max(0.0, plan_total - elapsed)
        return 0.0

    # Reserve the last slice of the bar for the join; it is seconds of work,
    # but the bar should not say 100% while a file is still being written.
    JOIN_PCT = 2
    work_pct = 100 - JOIN_PCT
    reused_span = sum(plan[i][1] - plan[i][0] for i in finished)

    # -- overlapping the stages (B9) --------------------------------------
    #
    # Per 60 s chunk at 8192x4096 90 fps the measured shares are: split ~3.6
    # min, both swaps ~72 min, merge ~8 min. The split and the merge are NVENC
    # and NVDEC with very little CPU; the swap is CUDA inference that leaves
    # the encoder at 12-16% and, since the onnxruntime thread pools were
    # bounded, the machine at ~20% CPU. So running the next chunk's split and
    # the previous chunk's merge underneath this chunk's swap hides ~12 minutes
    # of every 84 and costs one extra chunk of split files on disk.
    #
    # It is only possible because each chunk now has its own scratch directory:
    # before this, chunk 4's split and chunk 3's merge would both have been
    # reading and writing output/temp/left.mp4.
    overlap = bool(is_vr and guard_config.get("overlap_stages", True))
    # Deferring the merge needs nothing inside the chunk to run after it.
    # Temporal smoothing and the upscale both consume the merged file, so a
    # chunk with either of them merges in line, exactly as it always did.
    defer_merge = bool(overlap and not config.get("temporal_smoothing")
                       and not config.get("upscale_method"))
    if is_vr and log_cb:
        if overlap:
            log_cb("[Chunks] Overlap is ON: each chunk's split runs underneath the "
                   "previous chunk's face swap"
                   + (", and each chunk's merge underneath the next chunk's"
                      if defer_merge else
                      " (the merge stays in line - smoothing or upscaling needs it)")
                   + ".")
        else:
            log_cb("[Chunks] Overlap is OFF (overlap_stages in scripts/guardrails.json): "
                   "split, swap and merge run one after another.")

    def chunk_temp_dir(index):
        """
        Where one chunk's intermediates live.

        Per chunk for VR, because with the stages overlapped two chunks are
        alive at once and output/temp/left.mp4 cannot mean two different files.
        The 2D pipeline does not overlap anything, so it keeps the single
        directory and the resume behaviour that goes with it.
        """
        return os.path.join(temp_dir, "chunk_%03d" % index) if is_vr else temp_dir

    if is_vr and not resume:
        # Resume off means the chunk scratch from a previous attempt is not
        # wanted either; leaving it would let a stale split be adopted.
        for name in (os.listdir(temp_dir) if os.path.isdir(temp_dir) else []):
            if name.startswith("chunk_"):
                shutil.rmtree(os.path.join(temp_dir, name), ignore_errors=True)

    split_pool = merge_pool = None
    if overlap:
        # One worker each, deliberately. Two splits at once would race for the
        # same decoder and the same disk, and the point is to fill the gaps in
        # one swap, not to start a third pipeline.
        split_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vrdeep-split")
        merge_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vrdeep-merge")
    split_futures = {}
    merge_futures = {}

    def prefetch_split(index, under):
        """Split chunk `index` while chunk `under` has the GPU."""
        guard.check()
        target_dir = chunk_temp_dir(index)
        os.makedirs(target_dir, exist_ok=True)
        a, b = plan[index]
        chunk_config = dict(config, chunk_seconds=0)
        left, right = split_video(
            target_video, target_dir, inner_kwargs.get("layout") or "sbs",
            inner_kwargs.get("fps"), a, b, log_cb=None, guard=guard,
            encoder=config.get("encoder", "h264_nvenc"), guard_config=guard_config,
            lane="split")
        # Record it the way the inner pipeline would have, so that a run which
        # dies between here and the start of that chunk reuses this split
        # instead of doing it again. The split key covers the target, the
        # layout, the fps, the range and the encoder - not the source face -
        # so it can be computed here without resolving anything.
        try:
            keys, parts = vr_stage_keys(
                split_sources(source_face), target_video, chunk_config,
                inner_kwargs.get("layout") or "sbs", inner_kwargs.get("fps"), a, b,
                inner_kwargs.get("merge_encoder"), inner_kwargs.get("merge_crf"),
                inner_kwargs.get("merge_preset"))
            chunk_state = RunState(target_dir)
            for eye, path in (("left", left), ("right", right)):
                chunk_state.complete("split_" + eye, path, keys["split_" + eye],
                                     parts["split_" + eye])
        except Exception:
            # The checkpoint is an optimisation for the next attempt; the split
            # itself is what this chunk needs and it is already on disk.
            pass
        if log_cb:
            log_cb(f"[Chunks] split of chunk {index + 1} finished under chunk "
                   f"{under + 1}'s swap")
        return left, right

    def finish_merge(index, handoff, under):
        """Merge chunk `index`'s two eyes while chunk `under` has the GPU."""
        finish_deferred_merge(handoff, guard=guard, log_cb=log_cb)
        a, b = plan[index]
        span = b - a
        if not chunk_is_complete(paths[index], span, tolerance=max(1.0, span * 0.05)):
            got = media_duration(paths[index])
            raise PipelineError(
                f"Chunk {index + 1} came out {got:.1f}s long where {span:.1f}s was "
                f"expected; not joining a video with a hole in it. The file is at "
                f"{paths[index]}.")
        if log_cb:
            log_cb(f"[Chunks] merge of chunk {index + 1} finished under chunk "
                   f"{under + 1}'s swap")
        return index

    completed = list(finished)
    current = None
    ran_any = False
    chunk_started_at = {}

    def mark_chunk_done(index):
        """Bookkeeping for a chunk whose output file is now final."""
        nonlocal done_span
        a, b = plan[index]
        completed.append(index)
        done_span += b - a
        manifest["done"] = sorted(completed)
        _write_chunk_manifest(chunk_dir, manifest)
        scratch = chunk_temp_dir(index)
        # Only ever a per-chunk directory. The 2D pipeline shares output/temp
        # with everything else and removing that here would take the run's own
        # resume state with it.
        if not keep_temp and os.path.basename(scratch).startswith("chunk_"):
            shutil.rmtree(scratch, ignore_errors=True)
        if log_cb:
            started, paused_then = chunk_started_at.get(index, (time.time(), 0.0))
            took = time.time() - started - (paused_clock() - paused_then)
            log_cb(f"[Chunks] Chunk {index + 1}/{n} finished in "
                   f"{guardrails.fmt_duration(took)} and kept at {paths[index]}")

    def harvest_merges(wait=False):
        """
        Collect the background merges that have landed, in chunk order.

        A merge that failed raises here, at the next chunk boundary, with the
        same message it would have raised in line - which is what keeps a
        failure attributable to the chunk it belongs to rather than to whatever
        was running when it happened.
        """
        for index in sorted(merge_futures):
            future = merge_futures[index]
            if not wait and not future.done():
                continue
            merge_futures.pop(index)
            try:
                future.result()
            except PipelineError as exc:
                # It surfaces at the next chunk boundary, so without this the
                # log said "Stopped during chunk 3" about chunk 1's merge.
                raise PipelineError(
                    f"Chunk {index + 1}'s merge, which ran in the background under "
                    f"the next chunk's swap, failed: {exc}") from exc
            mark_chunk_done(index)

    try:
        for i, (a, b) in enumerate(plan):
            span = b - a
            if i in finished:
                continue
            current = i
            guard.check()
            harvest_merges()
            # Progress across the run is proportional to the seconds of video
            # done, and chunks already on disk count as done from the start. A
            # chunk whose merge is still running in the background counts too:
            # its swap is finished, which is 90% of it, and leaving it out would
            # send the bar back to where the previous chunk started.
            accounted = set(completed) | set(merge_futures)
            base = work_pct * (sum(plan[j][1] - plan[j][0] for j in accounted)) / total_span
            width = work_pct * span / total_span

            def chunk_progress(pct, text="", _base=base, _width=width, _i=i):
                # The inner clock appends its own "left" for this chunk;
                # replace it with the figure for the whole run.
                text = (text or "").split("  -  ")[0]
                left = overall_left()
                if left > 0:
                    text = f"{text}  -  {guardrails.fmt_duration(left)} left overall"
                if progress_cb:
                    progress_cb(int(_base + (max(0, min(100, pct)) / 100.0) * _width),
                                f"Chunk {_i + 1}/{n}: {text}")

            if log_cb:
                log_cb("-" * 60)
                log_cb(f"[Chunks] Starting chunk {i + 1}/{n}: "
                       f"{guardrails.fmt_duration(a)} to {guardrails.fmt_duration(b)}")
                left = overall_left()
                if left > 0:
                    log_cb(f"[Chunks] About {guardrails.fmt_duration(left)} left overall "
                           f"(done around {guardrails.fmt_clock(left)})")

            # A preflight sized to this chunk, so the inner clock's estimate
            # and its slice of the bar are for the minute it is about to do,
            # not the hours the whole range would take. Quiet, because the
            # full-range version was already logged; forced, because a chunk
            # is a subset of a range that has already been accepted.
            chunk_scratch = chunk_temp_dir(i)
            os.makedirs(chunk_scratch, exist_ok=True)

            # The split for this chunk may already have run under the previous
            # chunk's swap. Wait for it if it is still going - that wait is the
            # only part of the split anybody pays for.
            presplit = None
            pending = split_futures.pop(i, None)
            if pending is not None:
                if log_cb and not pending.done():
                    log_cb(f"[Chunks] Waiting for chunk {i + 1}'s split, which started "
                           f"under the previous chunk's swap...")
                presplit = pending.result()
            guard.check()

            # The NEXT chunk's split is started by the inner pipeline, through
            # the callback below, at the moment THIS chunk's split finishes and
            # its swap begins. Submitting it here instead - before the inner
            # pipeline has run - was the obvious thing to do and was wrong for
            # the first chunk of a run: chunk 1 has no swap yet to hide the
            # prefetch under, so the two splits ran at the same time and queued
            # behind the card's single NVENC engine, roughly doubling the time
            # before the first swap could start. Measured on the 13:21 run of
            # Sep 10: two 4096x4096 crops encoding at once, chunk 1's own split
            # against a 3m 23s estimate.
            def start_following_split(_index=i):
                following = _index + 1
                if (overlap and following < n and following not in finished
                        and following not in split_futures):
                    split_futures[following] = split_pool.submit(
                        prefetch_split, following, _index)

            chunk_config = dict(config, chunk_seconds=0)
            chunk_prepared = prepare_run(
                target_video, chunk_config, is_vr=is_vr, trim_start=a, trim_end=b,
                log_cb=None, temp_dir=chunk_scratch, force=True,
            )
            # A stage that is running underneath another one is not part of
            # this chunk's wait, so it gets none of the chunk's progress bar
            # and none of its budget. It keeps its predicted seconds, because
            # the calibration still wants to know what a split costs.
            hidden = set()
            if presplit:
                hidden.add("split")
            if defer_merge:
                hidden.add("merge")
            runtime = (chunk_prepared[2] or {}).get("runtime") or {}
            if hidden and runtime.get("stages"):
                for stage in runtime["stages"]:
                    if stage["key"] in hidden:
                        stage["overlapped"] = True
                runtime["total_seconds"] = sum(
                    stage["seconds"] for stage in runtime["stages"]
                    if not stage.get("overlapped"))

            kwargs = dict(inner_kwargs, trim_start=a, trim_end=b, keep_temp=keep_temp,
                          progress_cb=chunk_progress, log_cb=log_cb, guard=guard,
                          force=True, prepared=chunk_prepared,
                          # The source analysis is cached; redo it once, not per chunk.
                          refresh_source_faces=refresh_source_faces and not ran_any)
            if is_vr:
                kwargs["resume"] = resume
                kwargs["temp_dir"] = chunk_scratch
                kwargs["presplit"] = presplit
                kwargs["defer_merge"] = defer_merge
                kwargs["on_split_done"] = start_following_split
            chunk_started_at[i] = (time.time(), paused_clock())
            result = inner(source_face, target_video, paths[i], chunk_config, **kwargs)
            ran_any = True
            guard.check()

            if defer_merge:
                # `result` is the handoff: two swapped eyes and everything the
                # merge needs. It goes to the background and this loop moves on
                # to the next chunk's swap; the merge is collected, and the
                # chunk marked done, at the next boundary.
                merge_futures[i] = merge_pool.submit(
                    finish_merge, i, result, min(i + 1, n - 1))
                if log_cb:
                    log_cb(f"[Chunks] Chunk {i + 1}/{n}: both eyes swapped. The merge is "
                           f"running in the background while the next chunk starts.")
                current = None
            else:
                if not chunk_is_complete(paths[i], span, tolerance=max(1.0, span * 0.05)):
                    got = media_duration(paths[i])
                    raise PipelineError(
                        f"Chunk {i + 1} came out {got:.1f}s long where {span:.1f}s was expected; "
                        f"not joining a video with a hole in it. The file is at {paths[i]}.")
                current = None
                mark_chunk_done(i)

        # Everything still in the background has to land before the join: the
        # last chunk's merge is very often still running here, and joining a
        # file that is being written is the one way this could produce a broken
        # output rather than a slow one.
        if merge_futures and log_cb:
            log_cb(f"[Chunks] Waiting for {len(merge_futures)} background merge(s) to "
                   f"finish before joining.")
        harvest_merges(wait=True)

        if log_cb and reused_span:
            log_cb(f"[Chunks] {guardrails.fmt_duration(reused_span)} of video was reused "
                   "from earlier attempts.")

        # Every chunk is on disk. Join them, tag the result, and only then
        # take the chunks away.
        join_chunks(
            paths, output_path, target_video, plan,
            encoder=config.get("encoder", "hevc_nvenc"),
            crf=quality_to_cq(config.get("quality", 80)),
            guard=guard, guard_config=guard_config, log_cb=log_cb,
            progress_cb=progress_cb, start_pct=work_pct, end_pct=100,
        )
        if is_vr:
            # The tag lives in the container header, which the join rebuilt.
            apply_stereo_metadata(output_path, inner_kwargs.get("layout") or "sbs",
                                  log_cb=log_cb, guard=guard)

        if keep_temp:
            if log_cb:
                log_cb(f"[Chunks] Keep temp is on - the chunks are left in {chunk_dir}")
        else:
            shutil.rmtree(chunk_dir, ignore_errors=True)
            if log_cb:
                log_cb("[Chunks] Joined; the individual chunks have been deleted.")

        elapsed = time.time() - run_started - (paused_clock() - paused_at_start)
        if progress_cb:
            progress_cb(100, "All chunks joined - pipeline completed successfully!")
        if log_cb:
            log_cb(f"[Chunks] {n} chunks done in {guardrails.fmt_duration(elapsed)}.")
            log_cb(f"[SUCCESS] Output: {output_path}")
        return output_path

    except BaseException as exc:
        # Whatever went wrong, what is finished stays finished. Say so, and
        # say where, because that is the whole point of chunking.
        if log_cb:
            kept = sorted(completed)
            where = f"chunk {current + 1}/{n}" if current is not None else "the join"
            log_cb(f"[Chunks] Stopped during {where}: {exc}")
            if kept:
                log_cb(f"[Chunks] {len(kept)} finished chunk(s) are kept in {chunk_dir} - "
                       "each is a playable video of its own range:")
                for i in kept:
                    log_cb(f"  - {os.path.basename(paths[i])}: "
                           f"{guardrails.fmt_duration(plan[i][0])} to "
                           f"{guardrails.fmt_duration(plan[i][1])}")
            else:
                log_cb("[Chunks] No chunk had finished yet. Anything the inner stages "
                       "finished is kept in output/temp for resume.")
            log_cb("[Chunks] Start the run again with the same settings and it continues "
                   f"from chunk {(current if current is not None else n) + 1}.")
        raise

    finally:
        # Cancel whatever has not started and wait for whatever has. A split or
        # a merge left running here would be writing into a directory the
        # cleanup below is about to remove, and its ffmpeg would outlive the
        # run that owns it.
        for pool in (split_pool, merge_pool):
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        if owns_guard:
            guard.close()
