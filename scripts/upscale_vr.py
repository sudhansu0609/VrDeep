"""
Standalone VR upscaler: take any stereoscopic (or flat) video and enlarge it.

Nothing here touches the deepfake pipeline. Point it at a file and it produces a
bigger one.

Two things make this more than `ffmpeg -vf scale`:

  * **The eyes are scaled separately.** A resize filter reads a neighbourhood
    around every output pixel - lanczos reaches three source pixels either side
    - so scaling a side-by-side frame as one image drags left-eye pixels into
    the right eye along the seam, and vice versa. It is a narrow band, but it is
    the one part of the frame where the two views are guaranteed to disagree,
    which is exactly what the visual system uses to judge depth. Each eye is
    cropped out, scaled on its own, and restacked.

  * **It is one encode, not three.** The obvious way to do this is split to
    disk, upscale each eye, merge - which is three lossy generations for a job
    that needs one. For the resize methods the whole thing is a single ffmpeg
    filtergraph, so the source is decoded once and encoded once.

The encoder ceiling is the thing that decides what is actually possible, and it
is checked before any work starts rather than discovered forty minutes in. On
this machine (RTX 5060 Ti):

    hevc_nvenc / h264_nvenc / av1_nvenc   max 8192 x 8192
    NVDEC (hardware decode)               max 8192 wide

So 8K VR SBS (7680x3840) is comfortably inside hardware, and 16K VR SBS
(15360x7680) is outside it in both directions - it can only be produced by a
software encoder, and nothing with a hardware decoder will play it back. See
UPSCALE_8K_16K_PLAN.md for the measurements behind that.

There are two ceilings, and the lower one is usually not this PC. NVENC stops
at 8192; a headset's own decoder stops well before that. --headset applies the
second one, which is the difference between a file that encodes and a file that
plays.

Flat 2D video works the same way and needs no extra flags - an ordinary
1920x1080 clip goes to 7680x4320, which is true 8K UHD. The layout is detected
from the file rather than assumed; see detect_layout for why it errs toward
flat when nothing declares otherwise.

Examples:

    # An ordinary 2D video to 8K
    python upscale_vr.py holiday.mp4 -o holiday_8k.mp4 --target 8k

    # VR, for a Quest 3
    python upscale_vr.py vr180.mp4 -o out.mp4 --target 8k --headset quest3

    # Override the detection when a stereo file declares nothing
    python upscale_vr.py in.mp4 -o out.mkv --target 8k --layout sbs

    # neural super-resolution instead of a resize. Slow - see --method.
    python upscale_vr.py in.mp4 -o out.mkv --target 8k --method esrgan

    # Past 8K there is no merged file that hardware-decodes anywhere. One file
    # per eye is the only route that stays in hardware.
    python upscale_vr.py in.mp4 -o out.mkv --target 12k --split-eyes
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import guardrails
from guardrails import GuardAbort, ResourceGuard
from engine import (
    FACEFUSION_DIR, PROJECT_DIR, PYTHON, PipelineError,
    _encoder_args, apply_stereo_metadata, merge_videos, probe_video,
    quality_to_cq, split_video,
)


# Measured on this machine, encoder by encoder, by actually trying them:
#
#   hevc_nvenc  8192x8192 OK, 10240x5120 "No capable devices found"
#   av1_nvenc   8192x4096 OK, 15360x7680 "No capable devices found"
#   h264_nvenc  4096x4096 OK, 5120x2560  "No capable devices found"
#
# H.264 is the odd one out at half the limit, which is easy to miss because
# every NVENC encoder fails the same unhelpful way. NVDEC will not *decode*
# above 8192 either, so crossing that line costs playback, not just encoding.
NVENC_MAX_DIMENSION = 8192
ENCODER_MAX_DIMENSION = {
    "hevc_nvenc": 8192,
    "av1_nvenc": 8192,
    "h264_nvenc": 4096,
}

# HEVC level 6.2 - the highest the standard defines - tops out at 35,651,584
# luma samples per frame. Past this a software encoder is not just slow, it is
# producing an out-of-spec stream, and its default thread pool becomes a
# memory problem in its own right. Used as the trigger for both.
HEVC_MAX_LEVEL_SAMPLES = 35_651_584

# Named targets, as the packed width of a side-by-side frame - i.e. two eyes
# wide. resolve_geometry() derives the per-eye size from this, so the same name
# means the same per-eye resolution in every layout.
TARGETS = {
    "1080p": 1920,
    "4k": 3840,
    "6k": 5760,
    "8k": 7680,
    # The largest side-by-side frame a Quest 3 will decode: 8192x4320 is exactly
    # 138,240 blocks, the decoder's per-frame limit. See HEADSETS.
    "8.2k": 8192,
    "12k": 11520,
    "16k": 15360,
}

# Playback targets. These are constraints of the *headset's* decoder, which is
# a different and much lower ceiling than what this PC can encode - a file that
# came out of NVENC perfectly can still fail to play in a headset.
#
# The Quest 3 figures below were read off the device itself, out of
# /vendor/etc/media_codecs.xml (`c2.qti.hevc.decoder`, and `c2.qti.av1.decoder`
# which is identical), not taken from any published summary. They are markedly
# more generous than the "8K30" that gets quoted everywhere:
#
#     <Limit name="size"              max="8192x8192"/>
#     <Limit name="block-size"        value="16x16"/>
#     <Limit name="block-count"       range="36-138240"/>
#     <Limit name="blocks-per-second" max="7776000"/>
#     <Limit name="bitrate"           range="1-160000000"/>
#
# The important part is that the decoder has a *throughput* budget in 16x16
# blocks per second, not a fixed frame rate. Resolution and frame rate trade
# against each other, and 8K30 is one point on that curve rather than the end
# of it - 7680x3840 is 115,200 blocks, so the budget allows 67.5 fps.
#
# This is capability, not a promise about sustained playback: thermals and
# whatever the player app actually asks for still apply. Test a short clip.
HEADSETS = {
    "quest3": {
        "label": "Meta Quest 3",
        # Per-axis cap. The real area limit is max_blocks below, which binds
        # first on any wide frame.
        "max_dimension": 8192,
        "block_size": 16,
        # 138,240 blocks = 35,389,440 pixels per frame. For a 2:1 side-by-side
        # frame that works out to a maximum of exactly 8192x4320.
        "max_blocks": 138_240,
        "max_blocks_per_sec": 7_776_000,
        # 160 Mbit/s, from the decoder's own bitrate range - well above this
        # project's 120 default, so the headset is not the binding constraint.
        "maxrate_mbps": 160,
        # NVENC defaults to a 250-frame GOP, which is ten seconds at 25 fps.
        # Scrubbing a ten-second GOP on a headset feels broken.
        "gop_seconds": 2,
        # Per-eye panel and the horizontal field of view it covers. `wm size`
        # on the device reports 4128x2208 for both eyes, i.e. 2064x2208 each.
        # Used by report_sampling(), which is usually a more interesting
        # question than the decoder limit.
        "panel_w": 2064,
        "panel_h": 2208,
        "fov_h": 110,
    },
}


def decoder_budget(headset, geometry, fps):
    """
    Work out where a frame size and rate sit against the decoder's real budget.

    Returns (blocks, max_fps_at_this_size, fits) - `fits` covering both the
    per-axis cap and the per-frame block count.
    """
    profile = HEADSETS[headset]
    block = profile["block_size"]
    blocks = (geometry["out_w"] // block) * (geometry["out_h"] // block)
    if blocks <= 0:
        return 0, 0.0, False
    max_fps = profile["max_blocks_per_sec"] / blocks
    fits = (blocks <= profile["max_blocks"]
            and max(geometry["out_w"], geometry["out_h"]) <= profile["max_dimension"])
    return blocks, max_fps, fits


# How much of a frame's width is in front of you at once. A VR180 file spreads
# its width over 180 degrees, so a 110-degree headset sees about 61% of it; a
# VR360 file spreads the same pixels over twice the arc and you see half as
# many of them.
CONTENT_ARC = {"vr180": 180.0, "vr360": 360.0, "flat": None}

# FaceFusion's ONNX frame enhancers come in fixed scales. --output-video-scale
# is what actually decides the output size (see resolve_esrgan_plan).
ESRGAN_MODELS = {
    2: "real_esrgan_x2_fp16",
    4: "real_esrgan_x4_fp16",
    8: "real_esrgan_x8_fp16",
}

# Measured on this machine: real_esrgan_x2_fp16 at 3 execution threads.
#   1920x1080 in -> 0.099 frame/s      3840x2160 in -> 0.027 frame/s
# 4x the pixels cost 3.7x the time, so the cost tracks *input* pixels - the
# model is tiled, so it is the tile count that grows, not the model.
#
# Calibrated on the x2 model, which is the cheapest of the three. x4 and x8 run
# a heavier network over the same tiles, so for those this is a floor rather
# than a prediction, and the printed estimate says "at least" for that reason.
ESRGAN_SEC_PER_INPUT_MP = 4.5


def log(message):
    print(message, flush=True)


def even(value):
    """Encoders want even dimensions; yuv420 chroma is subsampled by two."""
    return max(2, int(round(value)) // 2 * 2)


# Markers VR libraries actually use in file names. Checked longest-first so
# "_360_tb" is not matched by the bare "tb" rule.
FILENAME_STEREO_MARKERS = [
    ("180x180_3dh", "sbs"), ("360x180_3dh", "sbs"), ("180x180_3dv", "tb"),
    ("_3dh", "sbs"), ("_3dv", "tb"),
    ("half-sbs", "sbs"), ("half_sbs", "sbs"), ("full-sbs", "sbs"), ("full_sbs", "sbs"),
    ("over-under", "tb"), ("over_under", "tb"),
    ("_sbs", "sbs"), ("_lr", "sbs"), ("_tb", "tb"), ("_ou", "tb"),
]

# Matroska stereo_mode values, and the packing this tool understands.
STEREO_MODE_LAYOUTS = {
    "left_right": "sbs", "right_left": "sbs",
    "top_bottom": "tb", "bottom_top": "tb",
    "mono": "mono",
}


def detect_layout(path):
    """
    Work out how a file is packed, from signals that are actually reliable.

    Comparing the two halves of the frame is the obvious idea and it does not
    work. Measured across real files, the PSNR between halves was ~13 dB for
    genuinely stereo footage and ~13 dB for flat footage - the parallax between
    two eyes is about as large as the difference between two unrelated halves
    of an ordinary shot, so the distributions sit on top of each other. Only an
    exact duplicate stands out (47 dB), which is not the case worth detecting.

    So this uses declarations rather than pixels, in descending order of how
    much they can be trusted, and **falls back to mono**.

    That fallback direction is deliberate and it is not symmetric. Treating a
    stereo file as mono scales the packed frame as one image: the output is
    still correct stereo, just with a little filter bleed across the seam.
    Treating a flat file as stereo crops half the frame away and duplicates it,
    which destroys the video. Guessing wrong in one direction is a blemish and
    in the other it is a ruined encode, so an unsure answer goes to mono.

    Returns (layout, reason).
    """
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:stream_tags=stereo_mode:"
                          "format_tags=stereo_mode",
         "-of", "json", path],
        capture_output=True, text=True, timeout=60,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    tag = None
    width = height = 0
    try:
        data = json.loads(result.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
        tag = ((stream.get("tags") or {}).get("stereo_mode")
               or (data.get("format", {}).get("tags") or {}).get("stereo_mode"))
    except (ValueError, KeyError, IndexError):
        pass

    # 1. An explicit declaration in the container. Authoritative.
    if tag and tag.lower() in STEREO_MODE_LAYOUTS:
        layout = STEREO_MODE_LAYOUTS[tag.lower()]
        return layout, f"the file declares stereo_mode={tag}"

    # 2. The naming conventions VR libraries use. Not authoritative, but people
    #    rely on them precisely because players do.
    name = os.path.basename(path).lower()
    for marker, layout in FILENAME_STEREO_MARKERS:
        if marker in name:
            return layout, f"the file name contains '{marker}'"

    # 3. Nothing said so. Aspect is only a hint - VR180 SBS is 2:1, but so are
    #    plenty of flat cinematic crops - so it is reported, not acted on.
    if width and height:
        ratio = width / height
        if 1.95 <= ratio <= 2.05:
            return "mono", (f"nothing declares a stereo layout. The {ratio:.2f}:1 aspect "
                            f"is typical of VR180 side-by-side, so if this IS stereo pass "
                            f"--layout sbs; treating it as flat")
    return "mono", "nothing declares a stereo layout, so treating it as flat 2D"


def eye_dimensions(src_w, src_h, layout):
    """The size of one eye inside the packed frame."""
    if layout == "sbs":
        return src_w / 2.0, float(src_h)
    if layout == "tb":
        return float(src_w), src_h / 2.0
    return float(src_w), float(src_h)


def resolve_geometry(probe, layout, target, scale):
    """
    Work out the output size, and each eye's size within it.

    A named target describes the **per-eye** resolution, not the packed frame,
    and this matters more than it looks. The same VR180 content at "8K" is
    7680x3840 side-by-side and 3840x7680 top-bottom - one eye is 3840x3840
    either way. Sizing from the packed frame's width instead would make "8K
    top-bottom" mean a 7680-wide eye, i.e. four times the pixels of "8K
    side-by-side", which is not what anybody means by it.

    Everything is then driven from one uniform scale factor, so the aspect
    ratio is preserved exactly and the two eyes stay identical in shape.
    """
    src_w, src_h = probe["width"], probe["height"]
    eye_src_w, eye_src_h = eye_dimensions(src_w, src_h, layout)

    if scale:
        factor = float(scale)
    elif target in TARGETS:
        # TARGETS holds the SBS packed width, which is two eyes wide.
        want_eye_w = TARGETS[target] / (1 if layout == "mono" else 2)
        factor = want_eye_w / eye_src_w
    else:
        # An explicit WxH names the packed frame. Only its width drives the
        # factor and the height is derived, so an odd request cannot silently
        # stretch the image.
        try:
            want_w, _ = (int(v) for v in str(target).lower().split("x"))
        except ValueError:
            raise PipelineError(
                f"Could not understand --target {target!r}.\n\n"
                f"Use a name ({', '.join(TARGETS)}), an explicit packed size like "
                f"7680x3840, or --scale 2.0."
            )
        factor = want_w / src_w

    if factor <= 0:
        raise PipelineError("The output scale works out to zero or less.")

    eye_w, eye_h = even(eye_src_w * factor), even(eye_src_h * factor)

    # Build the packed frame back up from the eyes rather than scaling it
    # directly: hstack/vstack refuse mismatched inputs, and rounding the total
    # and the eyes independently can leave them a pixel apart.
    if layout == "sbs":
        out_w, out_h = eye_w * 2, eye_h
    elif layout == "tb":
        out_w, out_h = eye_w, eye_h * 2
    else:
        out_w, out_h = eye_w, eye_h

    return {
        "factor": factor,
        "src_w": src_w, "src_h": src_h,
        "out_w": out_w, "out_h": out_h,
        "eye_w": eye_w, "eye_h": eye_h,
    }


def check_encoder_ceiling(geometry, encoder, allow_software, method):
    """
    Decide whether the requested size can be encoded, before anything runs.

    Without this the run reaches ffmpeg and dies with "No capable devices
    found", which names neither the cause nor the fix.
    """
    out_w, out_h = geometry["out_w"], geometry["out_h"]
    limit = ENCODER_MAX_DIMENSION.get(encoder or "")

    if limit is None or max(out_w, out_h) <= limit:
        # A software encoder has no practical frame limit, but the *decoder*
        # ceiling still applies to whoever has to play the result.
        if max(out_w, out_h) > NVENC_MAX_DIMENSION:
            log(f"  NOTE: {out_w}x{out_h} is beyond the {NVENC_MAX_DIMENSION}px hardware "
                f"decode limit as well. This file will not play back on hardware - "
                f"treat it as a master to downscale from.")
        return encoder

    if not allow_software:
        # H.264 stops at half what HEVC and AV1 manage, so name the encoder's
        # own limit rather than a single number that is wrong for one of them.
        swap_hint = ""
        if limit < NVENC_MAX_DIMENSION:
            swap_hint = (f"  0. Switch to hevc_nvenc or av1_nvenc, which reach "
                         f"{NVENC_MAX_DIMENSION}px on this card - {encoder} is the "
                         f"only one limited to {limit}px.\n")
        raise PipelineError(
            f"{out_w}x{out_h} cannot be encoded by {encoder}: it stops at "
            f"{limit}x{limit} on this card.\n\n"
            f"Ways forward:\n" + swap_hint +
            f"  1. Pick a smaller --target. 8k (7680 wide) is the largest that "
            f"stays in hardware, and the only one a headset can hardware-decode.\n"
            f"  2. Pass --allow-software to encode with libx265 instead. It works, "
            f"but it is far slower, and above 8192px the file will be beyond HEVC's "
            f"highest defined level - software decode only, on every machine.\n"
            f"  3. Pass --split-eyes to write one file per eye instead of a merged "
            f"one. At this size each eye is {geometry['eye_w']}x{geometry['eye_h']}, "
            f"which may well be inside the limit, so both files encode and decode "
            f"normally."
        )

    log(f"  {out_w}x{out_h} is past the NVENC limit; falling back to libx265 "
        f"(--allow-software was given).")
    log(f"  This file will be beyond HEVC's highest defined level (6.2) and will "
        f"only decode in software.")
    return "libx265"


def encoder_args_for(encoder, crf, guard_config, geometry, preset="p7",
                     headset=None, probe=None, output_path=""):
    """
    Encoder flags, with x265 re-tuned when the frame is enormous.

    x265's defaults assume a frame you can hold a lot of. At 15360x7680 one
    yuv420p frame is 177 MB, and the default pool - frame-level parallelism
    across 8 threads plus a 15-25 frame lookahead - holds dozens of them.
    Measured on this machine, 50 frames of 16K:

        preset veryfast, 8 threads (the default)   1.68 frame/s   23.6 GB peak
        + frame-threads=1, lookahead=10, 4 threads 1.46 frame/s   14.8 GB peak
        preset ultrafast, frame-threads=1, la=5    4.64 frame/s    9.6 GB peak

    The first of those drove free RAM to 0.6 GB on a 31.8 GB machine and the
    resource watchdog aborted the run - correctly. Serialising the frame
    threads costs about 13% of the speed and halves the memory, which is the
    trade worth making, since without it the run does not finish at all.
    """
    if headset:
        # The headset's sustained-read limit is lower than this project's own
        # default cap, so it has to override rather than sit alongside it.
        guard_config = dict(guard_config or {})
        guard_config["nvenc_maxrate_mbps"] = HEADSETS[headset]["maxrate_mbps"]

    args = _encoder_args(encoder, crf, guard_config, preset=preset)
    args += headset_encoder_args(headset, probe or {}, output_path, encoder)

    samples = geometry["out_w"] * geometry["out_h"]
    if encoder != "libx265" or samples <= HEVC_MAX_LEVEL_SAMPLES:
        return args

    # Drop the thread count _encoder_args chose and re-specify it.
    if "-threads" in args:
        index = args.index("-threads")
        del args[index:index + 2]

    return args + [
        "-threads", "4",
        "-x265-params",
        "frame-threads=1:rc-lookahead=10:bframes=2:log-level=none",
    ]


def headset_encoder_args(headset, probe, output_path, encoder=None):
    """
    The extra encoder flags a headset needs, beyond a correct resolution.

    A file can be the right size, in the right codec, and still not play. These
    are the three things that actually bite:

      * an over-long GOP, which makes seeking feel broken;
      * a bitrate the device cannot sustain reading while also decoding;
      * an HEVC-in-MP4 stream tagged hev1 rather than hvc1, which several
        players will not open at all.
    """
    if not headset:
        return []

    profile = HEADSETS[headset]
    fps = probe.get("fps") or 30.0
    encoder = encoder or ""
    is_hevc = encoder in ("hevc_nvenc", "libx265")

    args = [
        # This is the real guarantee, and it applies to every codec: 8-bit
        # 4:2:0 is what the hardware decoder wants. 10-bit or 4:4:4 drops the
        # device to software decode, if it plays at all.
        "-pix_fmt", "yuv420p",
        "-g", str(max(2, int(round(fps * profile["gop_seconds"])))),
    ]

    # Profile names are per-codec, not universal. `-profile:v main` is valid
    # for HEVC and meaningless to av1_nvenc, which has no -profile option at
    # all and fails to open with "Undefined constant ... in 'main'". Since
    # -pix_fmt already pins the thing that matters, only HEVC gets it.
    if is_hevc:
        args += ["-profile:v", "main"]

    if is_hevc and os.path.splitext(output_path)[1].lower() in (".mp4", ".m4v", ".mov"):
        # hvc1 vs hev1 is a signalling difference, not a coding one, and it is
        # the difference between "plays" and "black screen" in several players.
        # AV1 and H.264 get correct tags from ffmpeg without help.
        args += ["-tag:v", "hvc1"]
    return args


def check_headset_limits(headset, geometry, probe):
    """
    Report the frame against the decoder's real budget, and warn if it is over.

    The decoder trades resolution against frame rate through one blocks-per-
    second budget, so this reports the actual headroom at the chosen size
    rather than checking against a single quoted "8K30" figure.
    """
    if not headset:
        return
    profile = HEADSETS[headset]
    label = profile["label"]
    fps = probe.get("fps") or 30.0
    blocks, max_fps, fits = decoder_budget(headset, geometry, fps)

    log(f"  Decoder    : {blocks:,} blocks/frame of {profile['max_blocks']:,} max; "
        f"headroom is {max_fps:.1f} fps at this size (source is {fps:.4g})")

    if not fits:
        over = []
        if max(geometry["out_w"], geometry["out_h"]) > profile["max_dimension"]:
            over.append(f"past the {profile['max_dimension']}px per-axis limit")
        if blocks > profile["max_blocks"]:
            over.append(f"{blocks:,} blocks against a {profile['max_blocks']:,} limit")
        log(f"  WARNING: this frame is {' and '.join(over)} for the {label}. "
            f"It will encode here and will not play there.")
        return

    if fps > max_fps + 0.5:
        log(f"  WARNING: {fps:.4g} fps at {geometry['out_w']}x{geometry['out_h']} needs "
            f"{blocks * fps / 1e6:.1f}M blocks/s, over the "
            f"{profile['max_blocks_per_sec'] / 1e6:.1f}M the {label} decoder has.")
        log(f"           Drop to {max_fps:.0f} fps at this size, or pick a smaller "
            f"--target and keep the frame rate.")


def report_sampling(headset, content, geometry):
    """
    Say whether the headset can actually see the resolution being produced.

    The decoder ceiling gets all the attention, but for VR180 it is rarely the
    thing that limits what you see - the panel is. Only the part of the frame
    inside the field of view is on screen at any moment, so what matters is how
    many source pixels land on each panel pixel there:

        visible source pixels = eye width * (headset FOV / content arc)
        sampling ratio        = visible source pixels / panel width

    Below 1.0 the file is softer than the panel and more resolution genuinely
    helps. Around 1.0-1.4 they are matched, with a little headroom for
    resampling, and more resolution buys nothing you can see. Well above that
    you are spending bitrate and decode budget on detail the display cannot
    show - which for VR180 on this headset is what happens past 8K.

    VR360 is the opposite case: the same pixels are spread over twice the arc,
    so even 8K is short of the panel and the decoder ceiling really does bite.
    """
    if not headset or content not in CONTENT_ARC or CONTENT_ARC[content] is None:
        return
    profile = HEADSETS[headset]
    arc = CONTENT_ARC[content]

    visible = geometry["eye_w"] * (profile["fov_h"] / arc)
    ratio = visible / profile["panel_w"]

    log(f"  Sampling   : {visible:.0f} of {geometry['eye_w']} source px per eye are "
        f"in view at once, onto a {profile['panel_w']}px panel = {ratio:.2f}x")
    if ratio < 0.9:
        log(f"               Below the panel - this will look soft, and more "
            f"resolution would genuinely help.")
    elif ratio <= 1.4:
        log(f"               Well matched to the panel. More resolution than this "
            f"would not be visible.")
    else:
        log(f"               Past what the panel can show. A smaller --target "
            f"would look the same and decode more easily.")


def build_scale_filtergraph(geometry, layout, resize_flags):
    """
    Crop each eye, scale it on its own, restack. One decode, one encode.

    Scaling the stacked frame as a single image would let the resize kernel
    read across the seam - lanczos reaches three source pixels either side - so
    left-eye content bleeds into the right eye exactly where the two views
    disagree most.
    """
    eye_w, eye_h = geometry["eye_w"], geometry["eye_h"]
    scale = f"scale={eye_w}:{eye_h}:flags={resize_flags}"

    if layout == "sbs":
        return (f"[0:v]crop=iw/2:ih:0:0,{scale}[l];"
                f"[0:v]crop=iw/2:ih:iw/2:0,{scale}[r];"
                f"[l][r]hstack=inputs=2[v]")
    if layout == "tb":
        return (f"[0:v]crop=iw:ih/2:0:0,{scale}[t];"
                f"[0:v]crop=iw:ih/2:0:ih/2,{scale}[b];"
                f"[t][b]vstack=inputs=2[v]")
    return f"[0:v]{scale}[v]"


def run_ffmpeg_with_progress(cmd, guard, total_frames, progress_cb, label):
    """
    Run an ffmpeg encode, reporting progress as it goes.

    `-progress pipe:1` emits machine-readable `key=value` lines, which is far
    more reliable to parse than the human `-stats` line that rewrites itself
    with carriage returns. An 8K encode of any real length runs for minutes, so
    a progress bar that never moves is its own kind of bug.

    **stderr is merged into stdout, and that is not a stylistic choice.** An
    earlier version opened stderr as its own pipe and never read it. ffmpeg
    keeps writing to stderr for the life of the encode, so once the OS pipe
    buffer filled - about 64 KB, a few minutes in - ffmpeg blocked forever on
    the write and the encode froze with the progress bar stopped partway. It
    survived testing because a fifty-frame clip never produces enough output to
    fill the buffer; only a real encode does. One reader draining one pipe
    cannot deadlock, so that is what this does.
    """
    if not progress_cb or not total_frames:
        result = guard.run(cmd)
        return result.returncode

    # Drop -stats rather than trying to countermand it: with ffmpeg the last
    # flag wins, and inserting -nostats ahead of it left stats switched on,
    # which was half of the deadlock described above.
    cmd = [a for a in cmd if a != "-stats"]
    cmd[1:1] = ["-nostats", "-progress", "pipe:1"]

    proc = guard.spawn(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, bufsize=1, errors="replace")
    tail = []
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        if line.startswith("frame="):
            try:
                done = int(line.split("=", 1)[1])
            except ValueError:
                continue
            progress_cb(min(99, int(done / total_frames * 100)),
                        f"{label} frame {done}/{int(total_frames)}")
            continue
        # Everything that is not progress is kept, briefly, so a failure can
        # say what ffmpeg actually complained about.
        if "=" not in line:
            tail.append(line)
            del tail[:-15]

    proc.wait()
    if proc.returncode != 0 and tail:
        log("  ffmpeg said: " + " | ".join(tail[-3:]))
    return proc.returncode


def upscale_resize(input_path, output_path, geometry, layout, encoder, crf,
                   resize_flags, guard, guard_config, trim_start, trim_end,
                   headset=None, probe=None, progress_cb=None, total_frames=0):
    """The fast path: a single ffmpeg pass, no intermediate files."""
    filtergraph = build_scale_filtergraph(geometry, layout, resize_flags)

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats"]
    if trim_start is not None:
        cmd += ["-ss", str(trim_start)]
    if trim_end is not None:
        cmd += ["-to", str(trim_end)]
    cmd += [
        "-i", input_path,
        "-filter_complex", filtergraph,
        "-map", "[v]",
        # Optional: a VR file without an audio track is normal.
        "-map", "0:a?",
    ]
    cmd += encoder_args_for(encoder, crf, guard_config, geometry,
                            headset=headset, probe=probe, output_path=output_path)
    # Audio is untouched - re-encoding it would be a second lossy generation
    # for no reason, since this tool changes nothing about it.
    cmd += ["-c:a", "copy", output_path]

    code = run_ffmpeg_with_progress(cmd, guard, total_frames, progress_cb, "Upscaling")
    if code != 0:
        guard.check()
        raise PipelineError(
            f"ffmpeg failed. Command was:\n  {' '.join(cmd[:14])} ...")
    return output_path


def resolve_esrgan_plan(geometry, layout):
    """
    Pick the frame-enhancer model and the output scale FaceFusion needs, and
    snap the geometry to the size that will actually come out.

    The critical part is --output-video-scale. FaceFusion sizes the output
    encoder from the *target* video's resolution multiplied by that flag
    (workflows/to_video.py), so with it left at its default of 1.0 the enhancer
    computes a 2x/4x/8x frame and the encoder then scales it straight back down
    to the source size. Measured: 426x226 in, 426x226 out, with the entire
    super-resolution cost paid for nothing. It has to be passed.

    It is also quantised to 0.25 steps, so this path cannot hit an arbitrary
    target exactly. Rather than print a size and deliver a different one, the
    geometry is recomputed from the scale that will really be used - so a 1920
    request that can only be met as 1916 says 1916 up front.
    """
    eye_src_w, eye_src_h = eye_dimensions(geometry["src_w"], geometry["src_h"], layout)
    needed = geometry["eye_w"] / eye_src_w

    if needed > 8:
        raise PipelineError(
            f"--method esrgan tops out at 8x per eye; this target needs {needed:.2f}x. "
            f"Use --method lanczos, or pick a smaller --target."
        )

    # FaceFusion accepts 0.25 steps between 0.25 and 8.0.
    scale = min(8.0, max(0.25, round(needed * 4) / 4))
    # The model only has to be at least as large as the scale asked for; the
    # encoder resizes down from it, which is supersampling and costs nothing
    # extra beyond the model itself.
    model_scale = next((s for s in sorted(ESRGAN_MODELS) if s >= scale - 1e-6), 8)

    eye_w, eye_h = even(eye_src_w * scale), even(eye_src_h * scale)
    if layout == "sbs":
        out_w, out_h = eye_w * 2, eye_h
    elif layout == "tb":
        out_w, out_h = eye_w, eye_h * 2
    else:
        out_w, out_h = eye_w, eye_h

    snapped = dict(geometry, factor=scale, eye_w=eye_w, eye_h=eye_h,
                   out_w=out_w, out_h=out_h)
    return ESRGAN_MODELS[model_scale], scale, snapped


def upscale_esrgan_eye(input_path, output_path, temp_path, model, output_scale,
                       encoder, quality, guard, log_prefix):
    """Run FaceFusion with only the frame enhancer - no swapping, no source face."""
    # FaceFusion refuses any run whose output extension differs from its
    # target's (frame_enhancer.pre_process -> same_file_extension). It refuses
    # it after downloading the model and before writing anything, with only
    # "match the target and output extension!" to say why - so `-o out.mkv` on
    # an .mp4 input, which is this script's own documented example, could never
    # have worked. FaceFusion therefore always writes a file carrying the
    # input's extension, and a container change becomes the remux below.
    src_ext = os.path.splitext(input_path)[1]
    ff_output = output_path
    if src_ext.lower() != os.path.splitext(output_path)[1].lower():
        ff_output = os.path.splitext(output_path)[0] + ".ffwork" + src_ext

    cmd = [
        PYTHON, "facefusion.py", "headless-run",
        "--target-path", input_path,
        "--output-path", ff_output,
        "--temp-path", temp_path,
        "--processors", "frame_enhancer",
        "--frame-enhancer-model", model,
        "--frame-enhancer-blend", "100",
        # Without this the enhanced frame is scaled back down to the input size
        # and the entire super-resolution is discarded. See resolve_esrgan_plan.
        "--output-video-scale", str(output_scale),
        "--execution-providers", "cuda",
        "--execution-thread-count", "3",
        "--output-video-encoder", encoder,
        "--output-video-quality", str(quality),
        "--output-video-preset", "slow",
        # 'disk' writes every frame out as an image first, which at these sizes
        # is hundreds of GB and an extra lossy generation on top.
        "--workflow-strategy", "memory",
        "--log-level", "info",
    ]

    proc = guard.spawn(cmd, cwd=FACEFUSION_DIR, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        tail.append(line)
        del tail[:-25]
        if "%|" in line:
            # tqdm rewrites one line with \r; show the last state only.
            print(f"\r  {log_prefix} {line.split('|')[0].strip()}   ", end="", flush=True)
        elif "PROCESSORS" in line or "error" in line.lower():
            print(f"\n  {log_prefix} {line}", flush=True)
    print(flush=True)

    proc.wait()
    guard.check()
    if proc.returncode != 0 or not os.path.exists(ff_output):
        raise PipelineError(f"FaceFusion frame enhancement failed:\n" + "\n".join(tail[-12:]))

    if ff_output != output_path:
        # Container only. The frames are already encoded and must not be
        # re-encoded here, so this is -c copy or it is a bug.
        log(f"  {log_prefix} remuxing to {os.path.splitext(output_path)[1]}...")
        code = subprocess.call(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", ff_output, "-map", "0", "-c", "copy", output_path])
        if code != 0 or not os.path.exists(output_path):
            raise PipelineError(
                "Frame enhancement succeeded but the remux to "
                f"{os.path.splitext(output_path)[1]} failed. The enhanced video "
                f"is at {ff_output}.")
        os.remove(ff_output)
    return output_path


def upscale_neural(input_path, output_path, geometry, layout, encoder, crf, quality,
                   guard, guard_config, temp_dir, keep_temp):
    """
    The slow path: real super-resolution, per eye, through FaceFusion.

    This one does need intermediate files - FaceFusion works on whole video
    files, so there is no single-filtergraph version of it.
    """
    model, output_scale, _ = resolve_esrgan_plan(geometry, layout)
    log(f"  Model: {model}, --output-video-scale {output_scale} per eye")

    if layout == "mono":
        upscale_esrgan_eye(input_path, output_path, os.path.join(temp_dir, "ff_up"),
                           model, output_scale, encoder, quality, guard, "frame")
        return output_path

    log("  Splitting eyes...")
    left, right = split_video(input_path, temp_dir, layout, log_cb=lambda m: log(f"  {m}"),
                              guard=guard, encoder=encoder, guard_config=guard_config)

    left_up = os.path.join(temp_dir, "left_upscaled.mp4")
    right_up = os.path.join(temp_dir, "right_upscaled.mp4")
    upscale_esrgan_eye(left, left_up, os.path.join(temp_dir, "ff_up_left"),
                       model, output_scale, encoder, quality, guard, "left eye ")
    if not keep_temp:
        os.remove(left)
    upscale_esrgan_eye(right, right_up, os.path.join(temp_dir, "ff_up_right"),
                       model, output_scale, encoder, quality, guard, "right eye")
    if not keep_temp:
        os.remove(right)

    log("  Merging eyes...")
    merge_videos(left_up, right_up, output_path, layout, encoder, crf,
                 None, True, None, lambda m: log(f"  {m}"),
                 guard=guard, guard_config=guard_config)
    return output_path


def estimate_neural_hours(geometry, layout, probe, duration):
    """Wall-clock estimate for the neural path, from the measured rate."""
    eye_w = geometry["src_w"] / 2 if layout == "sbs" else geometry["src_w"]
    eye_h = geometry["src_h"] / 2 if layout == "tb" else geometry["src_h"]
    input_mp = (eye_w * eye_h) / 1e6
    eyes = 1 if layout == "mono" else 2
    frames = duration * probe["fps"]
    return frames * input_mp * ESRGAN_SEC_PER_INPUT_MP * eyes / 3600.0


def trimmed_duration(path, trim_start=None, trim_end=None):
    """
    How much of `path` a run will actually touch, in seconds.

    Either end can be left open - ffmpeg's -ss with no -to runs to the end of
    the file, and -to with no -ss starts at zero - so this must not require
    both to be set before it believes a range exists.
    """
    duration = max(0.0, guardrails.probe_media(path)["duration"])
    start = max(0.0, trim_start or 0.0)
    end = duration if trim_end is None else min(duration, trim_end)
    return max(0.0, end - start)


def describe(args):
    """
    Say what a run would produce, without running it. Returns a list of lines.

    This is the same preflight run_upscale() performs, so the Upscale tab's
    live preview cannot claim one thing and the run then do another. It answers
    the two questions worth asking before spending an encode: can this be
    encoded and played at all, and will the extra resolution be visible.
    """
    lines = []
    probe = probe_video(args.input)
    layout = args.layout
    if layout == "auto":
        layout, why = detect_layout(args.input)
        lines.append(f"Layout: {layout.upper()} - {why}.")

    geometry = resolve_geometry(probe, layout, args.target, args.scale)
    if args.method == "esrgan":
        _, _, geometry = resolve_esrgan_plan(geometry, layout)

    lines.append(f"{geometry['src_w']}x{geometry['src_h']} -> "
                 f"{geometry['out_w']}x{geometry['out_h']}  ({geometry['factor']:.3g}x)"
                 + (f", {geometry['eye_w']}x{geometry['eye_h']} per eye"
                    if layout != "mono" else ""))

    ceiling_geometry = geometry
    if args.split_eyes and layout != "mono":
        ceiling_geometry = dict(geometry, out_w=geometry["eye_w"], out_h=geometry["eye_h"])

    # check_encoder_ceiling reports the "this will not hardware-decode" note by
    # logging it, which in a preview would go to stdout and be invisible.
    # Borrow the log for the length of the call so it lands in the preview
    # instead - that note is one of the more important things it can say.
    global log
    captured, real_log = [], log
    log = captured.append
    try:
        check_encoder_ceiling(ceiling_geometry, args.encoder, args.allow_software, args.method)
        lines.extend(l.strip() for l in captured if l.strip())
    except PipelineError as exc:
        log = real_log
        # First line only: the preview is a few lines, not a wall of advice.
        lines.append("CANNOT ENCODE: " + str(exc).split("\n")[0])
        # Lead with the fix that actually applies. If this encoder is the one
        # holding things back rather than the resolution, swapping it is a free
        # win and the other suggestions are beside the point.
        limit = ENCODER_MAX_DIMENSION.get(args.encoder or "")
        if limit is not None and limit < NVENC_MAX_DIMENSION:
            lines.append(f"Switch to hevc_nvenc or av1_nvenc - they reach "
                         f"{NVENC_MAX_DIMENSION}px, {args.encoder} stops at {limit}px.")
        elif layout == "mono":
            lines.append("Pick a smaller target, or allow a software encoder "
                         "(slow, and the result will not hardware-decode).")
        else:
            lines.append("Tick 'one file per eye', or allow a software encoder, "
                         "or pick a smaller target.")
        return lines
    finally:
        log = real_log

    if args.headset:
        profile = HEADSETS[args.headset]
        blocks, max_fps, fits = decoder_budget(args.headset, geometry, probe["fps"])
        lines.append(f"{profile['label']}: {blocks:,} of {profile['max_blocks']:,} blocks, "
                     f"up to {max_fps:.0f} fps at this size "
                     f"(source is {probe['fps']:.4g})")
        if not fits:
            lines.append("WARNING: past this headset's decoder limit - it will not play.")
        elif probe["fps"] > max_fps + 0.5:
            lines.append(f"WARNING: {probe['fps']:.4g} fps is over the decoder's budget "
                         f"at this size.")

        content = args.content or ("flat" if layout == "mono" else "vr180")
        if CONTENT_ARC.get(content):
            visible = geometry["eye_w"] * (profile["fov_h"] / CONTENT_ARC[content])
            ratio = visible / profile["panel_w"]
            verdict = ("softer than the panel - more resolution would help" if ratio < 0.9
                       else "well matched to the panel" if ratio <= 1.4
                       else "past what the panel can show")
            lines.append(f"Sampling ({content}): {ratio:.2f}x - {verdict}.")

    frames = probe["fps"] * trimmed_duration(args.input, args.trim_start, args.trim_end)
    if args.method == "esrgan":
        hours = estimate_neural_hours(geometry, layout, probe, frames / (probe["fps"] or 30))
        lines.append(f"Estimated time: at least {hours:.1f} hours. A resize would take "
                     f"about {frames / 11 / 60:.1f} minutes.")
    elif frames:
        lines.append(f"Estimated time: about {frames / 10 / 60:.1f} minutes "
                     f"({int(frames)} frames at ~10 frame/s).")
    return lines


def build_parser():
    parser = argparse.ArgumentParser(
        description="Upscale a VR (or flat) video. Independent of the deepfake pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:")[-1],
    )
    parser.add_argument("input", help="Video to upscale")
    parser.add_argument("-o", "--output", required=True, help="Output path")
    parser.add_argument("--target", default="8k",
                        help=f"Output size: {', '.join(TARGETS)}, or an explicit "
                             f"width like 7680x3840. Default 8k.")
    parser.add_argument("--scale", type=float, default=None,
                        help="Uniform scale factor; overrides --target.")
    parser.add_argument("--layout", default="auto", choices=["auto", "sbs", "tb", "mono"],
                        help="How the INPUT is packed. 'mono' is ordinary flat 2D video. "
                             "'auto' (default) reads the container's stereo_mode tag and "
                             "the usual file-name markers, and falls back to mono - "
                             "because mis-reading stereo as flat is a blemish, while "
                             "mis-reading flat as stereo destroys the frame.")
    parser.add_argument("--method", default="lanczos",
                        choices=["lanczos", "bicubic", "spline", "esrgan"],
                        help="lanczos (default) is a high-quality resize, ~11 frame/s "
                             "to 8K. esrgan is real super-resolution and is roughly "
                             "300x slower - see the estimate it prints.")
    parser.add_argument("--content", default=None, choices=sorted(CONTENT_ARC),
                        help="What arc the footage covers, used with --headset to say "
                             "whether the resolution is actually visible. VR360 spreads "
                             "the same pixels over twice the arc as VR180, so it needs "
                             "twice the resolution to look equally sharp. Defaults to "
                             "vr180 for a stereo layout and flat for mono.")
    parser.add_argument("--headset", default=None, choices=sorted(HEADSETS),
                        help="Tune the output for a specific headset's decoder: caps "
                             "the bitrate, shortens the GOP so seeking works, pins "
                             "8-bit 4:2:0 Main, and warns when the size or frame rate "
                             "is past what the device can decode.")
    parser.add_argument("--encoder", default="hevc_nvenc")
    parser.add_argument("--quality", type=int, default=95,
                        help="0-100, mapped onto the encoder's cq the same way the "
                             "rest of this project does. Default 95.")
    parser.add_argument("--allow-software", action="store_true",
                        help="Permit a software encoder when the target is beyond "
                             "what NVENC can do (i.e. wider or taller than 8192).")
    parser.add_argument("--split-eyes", action="store_true",
                        help="Write one file per eye instead of a merged stereo file. "
                             "The only way to get 16K that still hardware-decodes.")
    parser.add_argument("--trim-start", type=float, default=None)
    parser.add_argument("--trim-end", type=float, default=None)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--yes", action="store_true",
                        help="Do not stop for confirmation on a long neural run.")
    return parser

def run_upscale(args, log_fn=None, progress_cb=None, guard=None):
    """
    Do the work, given an options object.

    Split out of main() so the GUI drives exactly the same code path the CLI
    does rather than a parallel copy of it - `args` is whatever carries the
    same attribute names, so argparse's namespace and a plain object from the
    Upscale tab both work. Returns a process exit code.
    """
    global log
    if log_fn is not None:
        log = log_fn


    if not os.path.exists(args.input):
        log(f"ERROR: input not found: {args.input}")
        return 1

    probe = probe_video(args.input)

    if args.layout == "auto":
        args.layout, why = detect_layout(args.input)
        log(f"  Layout detected as {args.layout.upper()}: {why}.")
        log("")

    if args.content is None:
        # A flat video has no arc to spread pixels over, so the sampling
        # comparison against a headset panel does not apply to it.
        args.content = "flat" if args.layout == "mono" else "vr180"

    try:
        geometry = resolve_geometry(probe, args.layout, args.target, args.scale)
        if args.method == "esrgan":
            # --output-video-scale only moves in 0.25 steps, so the reachable
            # size is not always the requested one. Settle it here, before the
            # summary is printed, rather than announcing a size and delivering
            # a different one.
            _, _, geometry = resolve_esrgan_plan(geometry, args.layout)
    except PipelineError as exc:
        log(f"ERROR: {exc}")
        return 1

    duration = trimmed_duration(args.input, args.trim_start, args.trim_end)

    log("=" * 68)
    log("VR UPSCALE")
    log("=" * 68)
    log(f"  Input      : {os.path.basename(args.input)}  "
        f"{geometry['src_w']}x{geometry['src_h']}  {probe['fps']:.3g} fps  {duration:.1f}s")
    log(f"  Layout     : {args.layout.upper()}")
    log(f"  Output     : {geometry['out_w']}x{geometry['out_h']}  "
        f"({geometry['factor']:.3g}x)")
    if args.layout != "mono":
        log(f"  Per eye    : {geometry['eye_w']}x{geometry['eye_h']}")
    log(f"  Method     : {args.method}")

    guard_config = guardrails.load_guard_config()
    crf = quality_to_cq(args.quality)

    # The eyes are what actually get encoded when --split-eyes is used, so the
    # ceiling applies to them rather than to the merged frame.
    ceiling_geometry = geometry
    if args.split_eyes and args.layout != "mono":
        ceiling_geometry = dict(geometry, out_w=geometry["eye_w"], out_h=geometry["eye_h"])
    log("")
    if args.headset:
        log(f"  Headset    : {HEADSETS[args.headset]['label']}  ({args.content})")
        report_sampling(args.headset, args.content, geometry)
        check_headset_limits(args.headset, geometry, probe)
    try:
        encoder = check_encoder_ceiling(ceiling_geometry, args.encoder,
                                        args.allow_software, args.method)
    except PipelineError as exc:
        # A wall, not a crash - it is the whole point of checking before any
        # work starts, so it should read like advice rather than a traceback.
        log(f"Cannot do that:\n\n{exc}")
        return 1

    if args.method == "esrgan":
        hours = estimate_neural_hours(geometry, args.layout, probe, duration)
        log("")
        log(f"  Estimated time: at least {hours:.1f} hours "
            f"({'both eyes' if args.layout != 'mono' else 'whole frame'}), from the "
            f"measured rate of {ESRGAN_SEC_PER_INPUT_MP} s per input megapixel on "
            f"the x2 model. The x4 and x8 models are slower still.")
        log(f"  A resize (--method lanczos) would take about "
            f"{duration * probe['fps'] / 11 / 60:.1f} minutes instead, and for VR "
            f"footage viewed at arm's length it is often indistinguishable.")
        if hours > 1 and not args.yes:
            log("")
            try:
                if input("  Continue? [y/N] ").strip().lower() not in ("y", "yes"):
                    log("  Cancelled.")
                    return 0
            except EOFError:
                log("  No terminal to confirm on; pass --yes to run anyway.")
                return 1

    temp_dir = os.path.join(PROJECT_DIR, "output", "temp", "upscale")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    # Reuse the caller's guard when there is one. The GUI starts its own and
    # its STOP button acts on that instance, so creating a second one here
    # would put the real work in a Job Object the Stop button cannot reach -
    # and leave two guards fighting over the same limits.
    owns_guard = guard is None
    guard = guard or ResourceGuard(
        temp_dir, guard_config, log_cb=lambda m: log(f"  {m}")).start()
    started = time.time()
    try:
        outputs = []
        if args.split_eyes and args.layout != "mono":
            # Each eye becomes its own file. Nothing is stacked, so nothing has
            # to fit under the merged-frame ceiling.
            stem, ext = os.path.splitext(args.output)
            for side, crop in (("left", "0:0"), ("right", "iw/2:0" if args.layout == "sbs" else "0:ih/2")):
                path = f"{stem}_{side}{ext}"
                log(f"\n  Writing {side} eye -> {os.path.basename(path)}")
                geo = dict(geometry, out_w=geometry["eye_w"], out_h=geometry["eye_h"])
                one = dict(geo, eye_w=geometry["eye_w"], eye_h=geometry["eye_h"])
                crop_expr = ("crop=iw/2:ih:" + crop) if args.layout == "sbs" \
                    else ("crop=iw:ih/2:" + crop)
                filtergraph = (f"[0:v]{crop_expr},scale={geometry['eye_w']}:"
                               f"{geometry['eye_h']}:flags={args.method}[v]")
                cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats"]
                if args.trim_start is not None:
                    cmd += ["-ss", str(args.trim_start)]
                if args.trim_end is not None:
                    cmd += ["-to", str(args.trim_end)]
                cmd += ["-i", args.input, "-filter_complex", filtergraph,
                        "-map", "[v]", "-map", "0:a?"]
                cmd += encoder_args_for(encoder, crf, guard_config, geo,
                                        headset=args.headset, probe=probe,
                                        output_path=path)
                cmd += ["-c:a", "copy", path]
                # Each eye gets half the bar, so it spans both files rather
                # than jumping back to zero halfway through.
                eye_index = len(outputs)
                code = run_ffmpeg_with_progress(
                    cmd, guard, duration * (probe.get("fps") or 0),
                    (lambda p, t, i=eye_index: progress_cb(i * 50 + p // 2, t))
                    if progress_cb else None,
                    f"{side.capitalize()} eye")
                if code != 0:
                    guard.check()
                    raise PipelineError(f"ffmpeg failed writing the {side} eye.")
                outputs.append(path)

        elif args.method == "esrgan":
            upscale_neural(args.input, args.output, geometry, args.layout, encoder,
                           crf, args.quality, guard, guard_config, temp_dir,
                           args.keep_temp)
            outputs.append(args.output)

        else:
            log("")
            upscale_resize(args.input, args.output, geometry, args.layout, encoder,
                           crf, args.method, guard, guard_config,
                           args.trim_start, args.trim_end,
                           headset=args.headset, probe=probe,
                           progress_cb=progress_cb,
                           total_frames=duration * (probe.get("fps") or 0))
            outputs.append(args.output)

        # Matroska only; .mp4 cannot carry the tag, and apply_stereo_metadata
        # says so rather than pretending it did.
        if args.layout != "mono" and not args.split_eyes:
            apply_stereo_metadata(args.output, args.layout,
                                  log_cb=lambda m: log(f"  {m}"), guard=guard)

    except GuardAbort as exc:
        log(f"\nSTOPPED: {exc}")
        return 1
    except PipelineError as exc:
        log(f"\nERROR: {exc}")
        return 1
    finally:
        if not args.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
        # Only close what we opened; the GUI closes its own in finish_run().
        if owns_guard:
            guard.close()

    elapsed = time.time() - started
    log("")
    log("=" * 68)
    for path in outputs:
        size_gb = os.path.getsize(path) / 1e9
        log(f"  {path}  ({size_gb:.2f} GB)")
    frames = duration * probe["fps"]
    log(f"  Done in {elapsed:.0f}s" + (f"  ({frames / elapsed:.2f} frame/s)" if elapsed else ""))
    log("=" * 68)
    return 0



def main():
    args = build_parser().parse_args()
    return run_upscale(args)


if __name__ == "__main__":
    sys.exit(main())
