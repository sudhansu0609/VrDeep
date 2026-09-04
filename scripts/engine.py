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
    "threads": 3,
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
    pass


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
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
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
            return {"width": w, "height": h, "fps": fps}
        except Exception:
            pass
    return {"width": 1920, "height": 1080, "fps": 30.0}


def quality_to_cq(quality):
    """
    Map the GUI's 0-100 Quality onto a -cq / -crf value the same way FaceFusion
    maps --output-video-quality internally (numpy.interp over [51, 0]), so the
    merge does not silently re-encode at a different quality than the swap.
    """
    quality = max(0, min(100, int(quality if quality is not None else 80)))
    return int(round(51 - (quality / 100.0) * 51))


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
                log_cb=None, guard=None, encoder="h264_nvenc", guard_config=None):
    guard = guard or NULL_GUARD
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

    for label, filt, out in [("Left eye", fl, left_out), ("Right eye", fr, right_out)]:
        guard.check()
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
        cmd += ["-i", input_path, "-vf", filt]
        if fps:
            cmd += ["-r", str(fps)]
        cmd += enc_args + ["-c:a", "aac", "-b:a", "256k", out]

        result = guard.run(cmd, capture_output=True)
        if result.returncode != 0:
            guard.check()
            err = (result.stderr or b"").decode(errors="replace")[:400]
            raise PipelineError(f"Split failed for {label}: {err}")

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
                   guard=None):
    guard = guard or NULL_GUARD
    guard.check()

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
        cmd, cwd=FACEFUSION_DIR,
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
                "  - raise job_memory_cap_gb in scripts/guardrails.json (this machine's "
                f"commit limit is {guardrails.fmt_gb(guardrails.commit_limit_bytes())})\n\n"
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
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_path,
         "-vf", f"scale={new_w}:{new_h}:flags=lanczos"]
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

    proc = guard.spawn(cmd, cwd=FACEFUSION_DIR, stdout=subprocess.PIPE,
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


def apply_temporal_smoothing(input_path, output_path, encoder="hevc_nvenc", crf=18,
                             log_cb=None, guard=None):
    """Apply motion-compensated interpolation for temporal smoothing to reduce flickering."""
    guard = guard or NULL_GUARD
    guard.check()
    if log_cb:
        log_cb("Applying temporal smoothing to reduce flickering...")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    probe = probe_video(input_path)
    fps = probe.get("fps", 30)

    # minterpolate is single-threaded, extremely slow, and holds several full
    # frames in memory. Cap ffmpeg's thread use so it cannot spread out.
    cmd = (
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", input_path,
         "-vf", f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
         "-threads", str(max(2, min(6, (os.cpu_count() or 8) // 4)))]
        + _encoder_args(encoder, crf, preset="p7")
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
                 guard_config=None):
    guard = guard or NULL_GUARD
    guard.check()
    if log_cb:
        maxrate = int((guard_config or {}).get("nvenc_maxrate_mbps") or 120)
        log_cb(f"Merging into {layout.upper()} video (cq {crf}, cap {maxrate} Mbit/s)")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    filt = "[0:v][1:v]hstack=2[vout]" if layout == "sbs" else "[0:v][1:v]vstack=2[vout]"

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", left_path, "-i", right_path,
           "-filter_complex", filt]
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

    result = guard.run(cmd, capture_output=True)
    if result.returncode != 0:
        guard.check()
        raise PipelineError(f"Merge failed: {(result.stderr or b'').decode(errors='replace')[:400]}")

    return output_path


# stereo_mode is a Matroska element. ffmpeg accepts -metadata:s:v:0
# stereo_mode=... for any container and silently drops it for the ones that
# have nowhere to put it, so an .mp4 "VR" file carries no stereo tag at all and
# a headset has to be told the layout by hand.
STEREO_METADATA_FORMATS = {".mkv", ".webm"}


def apply_stereo_metadata(path, layout, log_cb=None, guard=None):
    """
    Stamp the stereo layout onto a finished file, in place, without re-encoding.

    This runs last rather than as part of the merge because every optional step
    after the merge - temporal smoothing, upscaling - re-encodes into a fresh
    file and would drop the tag again.
    """
    guard = guard or NULL_GUARD
    ext = os.path.splitext(path)[1].lower()
    mode = "left_right" if layout == "sbs" else "top_bottom"

    if ext not in STEREO_METADATA_FORMATS:
        if log_cb:
            log_cb(f"Note: {ext or 'this container'} cannot store the stereo_mode tag, "
                   f"so the output is not self-describing - the headset or player has to "
                   f"be told it is {layout.upper()}. Use .mkv to have it tagged.")
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
    return path


def cleanup_temp(temp_dir, items=None):
    if items is None:
        # Prefixes, not names: the merged and smoothed intermediates take the
        # output's container extension so the final move cannot rename an mp4
        # to .mkv, which is how "choose .mkv for stereo metadata" used to
        # produce an mp4 with an .mkv name and no metadata.
        items = ["ff_temp_left", "ff_temp_right", "ff_temp_normal",
                 "left.mp4", "right.mp4", "left_swapped.mp4", "right_swapped.mp4",
                 "smoothed_result.mp4", "ff_result.mp4", "trimmed_target.mp4",
                 "upscale_frames"]
        for name in os.listdir(temp_dir) if os.path.isdir(temp_dir) else []:
            if name.startswith(("merged_vr.", "smoothed_vr.")):
                items.append(name)
    for item in items:
        path = os.path.join(temp_dir, item)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)


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

    safe_config = guardrails.clamp_config(config, guard_config, log_cb=log_cb)
    blocking, warnings, estimate = guardrails.preflight(
        target_video, temp_dir, safe_config, guard_config,
        is_vr=is_vr, trim_start=trim_start, trim_end=trim_end,
    )

    if log_cb:
        log_cb(
            f"[Guard] Estimated temp usage: {guardrails.fmt_gb(estimate['bytes'])} "
            f"({estimate['frames']} frames, {estimate['frame_format']}), "
            f"free: {guardrails.fmt_gb(estimate.get('free_disk', 0))}"
        )
        for warning in warnings:
            log_cb(f"[Guard] {warning}")

    if blocking and not force:
        raise PipelineError(
            "Refusing to start - this run would very likely lock up the machine:\n\n"
            + "\n\n".join(blocking)
        )
    if blocking and force and log_cb:
        log_cb("[Guard] Preflight problems OVERRIDDEN by user; watchdog is still armed.")

    return safe_config, guard_config, estimate


def run_vr_pipeline(source_face, target_video, output_path, config, layout="sbs", fps=None,
                    trim_start=None, trim_end=None, merge_encoder=None, merge_crf=None,
                    merge_preset=None, keep_temp=False, progress_cb=None, log_cb=None,
                    guard=None, force=False, prepared=None, refresh_source_faces=False):
    temp_dir = os.path.join(PROJECT_DIR, "output", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_DIR, "output", "final"), exist_ok=True)

    config, guard_config, _ = prepared or prepare_run(
        target_video, config, is_vr=True, trim_start=trim_start, trim_end=trim_end,
        log_cb=log_cb, temp_dir=temp_dir, force=force,
    )
    # Reuse the caller's guard if it owns one, otherwise own it for this run.
    owns_guard = guard is None
    guard = guard or ResourceGuard(temp_dir, guard_config, log_cb=log_cb).start()

    # The merge step used to hardcode libx265/preset medium, which pinned every
    # core for hours regardless of what the user had configured.
    merge_encoder = merge_encoder or config.get("encoder", "hevc_nvenc")
    # The merge is the last encode the output goes through, and it used to be
    # pinned at cq 20 regardless of the Quality setting - so asking for 95 in
    # the GUI bought a better FaceFusion intermediate and then threw the
    # difference away here. Track the same setting instead.
    if merge_crf is None:
        merge_crf = quality_to_cq(config.get("quality", 80))

    if progress_cb:
        progress_cb(0, "Initializing VR Deepfake Pipeline")
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
            progress_cb=(lambda p, t: progress_cb(int(p * 0.04), t)) if progress_cb else None,
            refresh=refresh_source_faces,
        )

        if progress_cb:
            progress_cb(5, "Stage 1/5: Splitting stereoscopic video streams")
        if log_cb:
            log_cb("[Stage 1/5] Splitting stereoscopic 3D video into Left and Right eye streams...")

        left_video, right_video = split_video(
            target_video, temp_dir, layout, fps, trim_start, trim_end, log_cb,
            guard=guard, encoder=config.get("encoder", "h264_nvenc"), guard_config=guard_config,
        )

        left_out = os.path.join(temp_dir, "left_swapped.mp4")
        left_temp = os.path.join(temp_dir, "ff_temp_left")
        if log_cb:
            log_cb("[Stage 2/5] Running FaceFusion AI processing on Left Eye stream...")
        run_facefusion(
            source_face, left_video, left_out, left_temp, config,
            log_cb=log_cb, progress_cb=progress_cb,
            start_pct=10, end_pct=48, stage_name="Left Eye", guard=guard
        )

        # The left eye's extracted frames are dead weight once it has rendered,
        # and holding both eyes' frame dumps at once is what filled the disk.
        if not keep_temp:
            cleanup_temp(temp_dir, ["ff_temp_left", "left.mp4"])

        right_out = os.path.join(temp_dir, "right_swapped.mp4")
        right_temp = os.path.join(temp_dir, "ff_temp_right")
        if log_cb:
            log_cb("[Stage 3/5] Running FaceFusion AI processing on Right Eye stream...")
        run_facefusion(
            source_face, right_video, right_out, right_temp, config,
            log_cb=log_cb, progress_cb=progress_cb,
            start_pct=48, end_pct=85, stage_name="Right Eye", guard=guard
        )

        if not keep_temp:
            cleanup_temp(temp_dir, ["ff_temp_right", "right.mp4"])

        if progress_cb:
            progress_cb(85, "Stage 4/5: Merging streams into 3D Stereoscopic VR video")
        if log_cb:
            log_cb("[Stage 4/5] Merging processed Left and Right eye videos with VR spatial metadata...")

        # Match the output container, so the final move is a move and not a
        # rename into a format the file is not actually in.
        out_ext = os.path.splitext(output_path)[1].lower() or ".mp4"
        merged_path = os.path.join(temp_dir, "merged_vr" + out_ext)
        merge_videos(
            left_out, right_out, merged_path, layout, merge_encoder, merge_crf,
            merge_preset, True, fps, log_cb, guard=guard, guard_config=guard_config,
        )

        current_output = merged_path

        if config.get("temporal_smoothing"):
            if progress_cb:
                progress_cb(87, "Stage 5/6: Applying temporal smoothing to reduce flickering...")
            if log_cb:
                log_cb("[Stage 5/6] Applying temporal smoothing to reduce flickering...")
            smoothed_output = os.path.join(temp_dir, "smoothed_vr" + out_ext)
            apply_temporal_smoothing(
                current_output, smoothed_output,
                encoder=config.get("encoder", "hevc_nvenc"),
                crf=quality_to_cq(config.get("quality", 80)),
                log_cb=log_cb, guard=guard
            )
            if not keep_temp:
                os.remove(current_output)
            current_output = smoothed_output

        if config.get("upscale_method"):
            if progress_cb:
                progress_cb(92, f"Stage 6/6: Upscaling video via {config['upscale_method']}")
            if log_cb:
                log_cb(f"[Stage 6/6] Upscaling merged video via {config['upscale_method']}...")
            upscale_out = output_path
            upscale_video(
                current_output, upscale_out,
                method=config["upscale_method"],
                scale=config.get("upscale_scale", 2),
                target_width=config.get("upscale_target_width"),
                encoder=config.get("encoder", "hevc_nvenc"),
                crf=quality_to_cq(config.get("quality", 80)),
                log_cb=log_cb, guard=guard, progress_cb=progress_cb,
                tile=guard_config.get("esrgan_tile", 512),
                guard_config=guard_config,
            )
            if not keep_temp:
                os.remove(current_output)
        else:
            shutil.move(current_output, output_path)

        # Last, because smoothing and upscaling each re-encode into a new file
        # and would drop the tag.
        apply_stereo_metadata(output_path, layout, log_cb=log_cb, guard=guard)

        if not keep_temp:
            if log_cb:
                log_cb("Cleaning up temporary processing files...")
            cleanup_temp(temp_dir)

        if progress_cb:
            progress_cb(100, "VR Deepfake Pipeline completed successfully!")
        if log_cb:
            log_cb("[SUCCESS] VR Deepfake Pipeline finished!")

        return output_path

    except GuardAbort as e:
        if log_cb:
            log_cb(f"[STOPPED] {e}")
        raise PipelineError(str(e))

    except Exception as e:
        if log_cb:
            log_cb(f"[ERROR] VR Pipeline Failed: {str(e)}")
        raise

    finally:
        # Always drop the temp frame dumps; leaving them behind is how the
        # drive ended up full before the next run even started.
        if not keep_temp:
            cleanup_temp(temp_dir)
        if owns_guard:
            guard.close()


def run_normal_pipeline(source_face, target_video, output_path, config, fps=None,
                        trim_start=None, trim_end=None, keep_temp=False,
                        progress_cb=None, log_cb=None, guard=None, force=False, prepared=None,
                        refresh_source_faces=False):
    temp_dir = os.path.join(PROJECT_DIR, "output", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_DIR, "output", "final"), exist_ok=True)

    config, guard_config, _ = prepared or prepare_run(
        target_video, config, is_vr=False, trim_start=trim_start, trim_end=trim_end,
        log_cb=log_cb, temp_dir=temp_dir, force=force,
    )
    owns_guard = guard is None
    guard = guard or ResourceGuard(temp_dir, guard_config, log_cb=log_cb).start()

    # The VR path hands fps to split/merge, but this path has neither, so the
    # override has to reach FaceFusion itself. Without this the Output FPS box
    # in the Normal tab was read from the GUI and then silently dropped.
    if fps:
        config = dict(config)
        config["fps"] = fps

    if progress_cb:
        progress_cb(0, "Initializing Normal Deepfake Pipeline")
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
            progress_cb=(lambda p, t: progress_cb(int(p * 0.04), t)) if progress_cb else None,
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
                + span + ["-i", target_video]
                # Same role as the VR split: this is the *input* to the swap,
                # so match its quality rather than the old blanket 18.
                + _encoder_args(config.get("encoder", "h264_nvenc"), 16, preset="p6")
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

        run_facefusion(
            source_face, target_for_ff, ff_output, ff_temp, config,
            log_cb=log_cb, progress_cb=progress_cb,
            start_pct=10, end_pct=80, stage_name="FaceFusion AI", guard=guard
        )

        current_output = ff_output

        # Frame dumps are the bulk of the temp usage; drop them as soon as
        # FaceFusion has rendered its result.
        if not keep_temp:
            cleanup_temp(temp_dir, ["ff_temp_normal"])

        if config.get("temporal_smoothing") and not is_image:
            if progress_cb:
                progress_cb(82, "Applying temporal smoothing to reduce flickering...")
            if log_cb:
                log_cb("[Stage 3/5] Applying temporal smoothing to reduce flickering...")
            smoothed_output = os.path.join(temp_dir, "smoothed_result.mp4")
            apply_temporal_smoothing(
                current_output, smoothed_output,
                encoder=config.get("encoder", "hevc_nvenc"),
                crf=quality_to_cq(config.get("quality", 80)),
                log_cb=log_cb, guard=guard
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
            current_output = smoothed_output

        if config.get("upscale_method") and is_image and log_cb:
            log_cb("Upscaling is skipped for image targets; use the frame enhancer instead.")

        if config.get("upscale_method") and not is_image:
            if progress_cb:
                progress_cb(90, f"Stage 4/5: Upscaling video via {config['upscale_method']}")
            if log_cb:
                log_cb(f"[Stage 4/5] Upscaling output video via {config['upscale_method']}...")
            upscale_video(
                current_output, output_path,
                method=config["upscale_method"],
                scale=config.get("upscale_scale", 2),
                target_width=config.get("upscale_target_width"),
                encoder=config.get("encoder", "hevc_nvenc"),
                crf=quality_to_cq(config.get("quality", 80)),
                log_cb=log_cb, guard=guard, progress_cb=progress_cb,
                tile=guard_config.get("esrgan_tile", 512),
                guard_config=guard_config,
            )
            if not keep_temp:
                os.remove(current_output)
        else:
            shutil.move(current_output, output_path)

        if not keep_temp:
            if log_cb:
                log_cb("[Stage 5/5] Cleaning up temporary working files...")
            cleanup_temp(temp_dir, ["ff_temp_normal", "trimmed_target.mp4",
                                    "smoothed_result.mp4", os.path.basename(ff_output)])

        if progress_cb:
            progress_cb(100, "Normal Deepfake Pipeline completed successfully!")
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
