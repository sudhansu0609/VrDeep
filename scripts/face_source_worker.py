"""
Build a set of source face stills from a short video, in its own process.

FaceFusion's --source-paths only accepts images. It reads every one of them,
takes the largest face in each, and averages the ArcFace embeddings
(face_creator.average_face_identity), so several stills of one person give a
better-conditioned identity than any single photo does. What it will not do is
take a video and work that out for itself.

That is what this worker is: decode a short clip, find the face, throw away the
frames that would poison the average, and hand back a handful of stills.

Two things make this more than "grab every 20th frame":

  * Frames that are blurred, badly lit, tiny, or turned away do not carry
    identity - they carry noise, and averaging them in drags the mean embedding
    away from the person. Every sampled frame is scored and the bad ones are
    dropped rather than merely ranked.
  * If a second person appears in the clip - even for a moment - averaging
    across both produces an identity that is neither of them. Candidates are
    clustered by embedding and only the dominant identity survives.

What is kept is then chosen for spread, not just for score: six near-identical
frames from one second of footage average to the same thing one frame does. The
picks are spread over pose, expression and time so the average is taken over
genuinely different views of the same face.

This runs here rather than in the GUI for the same reasons as esrgan_worker.py:
onnxruntime, a CUDA context and four models are all process-lifetime once
loaded, and the parent spawns this through the guard so the memory cap, the CPU
cap, the watchdog and the Stop button all reach it.

Progress goes to stdout as `FACESRC_PROGRESS <done> <total>`, the result as one
`FACESRC_RESULT <json>` line, and fatal problems as a single `FACESRC_ERROR`
line so the parent can show a sentence instead of a traceback.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

# This file lives in scripts/, so Python puts scripts/ on sys.path and not the
# FaceFusion checkout - and the parent runs it from the FaceFusion directory
# only so that facefusion.ini is found, which does nothing for imports. Adding
# it here rather than relying on a PYTHONPATH the caller has to remember keeps
# the worker runnable on its own from anywhere.
FACEFUSION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "facefusion")
if FACEFUSION_DIR not in sys.path:
    sys.path.insert(0, FACEFUSION_DIR)


def emit(line):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def fail(message):
    emit("FACESRC_ERROR " + " ".join(str(message).split()))
    sys.exit(2)


def probe(path):
    """Width/height/fps/duration, or None if ffprobe cannot read the file."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate:format=duration",
             "-of", "json", path],
            capture_output=True, text=True, timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0 or not result.stdout:
            return None
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        num, den = stream.get("r_frame_rate", "30/1").split("/")
        return {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": (float(num) / float(den)) if float(den) else 30.0,
            "duration": float(data.get("format", {}).get("duration") or 0.0),
        }
    except Exception:
        return None


def init_facefusion(args):
    """
    Bring FaceFusion's state up far enough for its face analyser to run.

    Every module in facefusion.face_* reads its settings out of state_manager
    rather than taking them as arguments, so the state has to exist before the
    first detect. Going through the real argument parser rather than setting
    the dozen keys by hand means the defaults for everything we do not care
    about come from FaceFusion itself, and cannot drift out of step with it.
    """
    from facefusion import logger, state_manager
    from facefusion.args import apply_args
    from facefusion.program import create_program

    program = create_program()
    # headless-run wants source/target/output paths; nothing on this path reads
    # them, but the parser will not produce a namespace without them.
    argv = [
        "headless-run",
        "--source-paths", args.video,
        "--target-path", args.video,
        "--output-path", args.video,
        "--execution-providers", *str(args.provider or "cuda").split(),
        "--face-detector-model", args.detector_model,
        "--face-detector-size", args.detector_size,
        "--face-detector-score", str(args.detector_score),
        "--face-detector-angles", *str(args.detector_angles or "0").split(),
        "--face-landmarker-model", args.landmarker_model,
        "--face-landmarker-score", str(args.landmarker_score),
    ]
    apply_args(vars(program.parse_args(argv)), state_manager.init_item)
    logger.init(state_manager.get_item("log_level"))

    import facefusion.face_classifier as face_classifier
    import facefusion.face_detector as face_detector
    import facefusion.face_landmarker as face_landmarker
    import facefusion.face_recognizer as face_recognizer

    for name, module in (("detector", face_detector), ("landmarker", face_landmarker),
                         ("recognizer", face_recognizer), ("classifier", face_classifier)):
        if not module.pre_check():
            fail(f"The face {name} model could not be downloaded or verified. "
                 f"Run a normal swap once so the models are fetched, then retry.")


def decode_frames(video, width, height, sample_fps, trim_start, trim_end,
                  decode_threads=4):
    """
    Yield sampled frames one at a time.

    Sequential decode through ffmpeg's fps filter, not cv2 seeking: seeking a
    long-GOP file lands on the wrong frames and costs a full decode per seek
    anyway. Exactly one frame is held at a time - a 4K frame is 25 MB as bgr24,
    and holding four hundred of them is 10 GB.
    """
    import numpy

    # One decode thread per core is ffmpeg's default, and each holds its own
    # decoded frames - 100 MB apiece on an 8192x4096 source. Capped for the
    # same reason as the pipeline's other decodes (see engine._decode_args).
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-threads", str(max(1, int(decode_threads)))]
    if trim_start:
        cmd += ["-ss", str(trim_start)]
    if trim_end:
        cmd += ["-to", str(trim_end)]
    cmd += [
        "-i", video,
        "-an", "-sn",
        "-vf", f"fps={sample_fps:.6f},scale={width}:{height}:flags=lanczos",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]
    # stderr goes to a temp FILE, not a pipe. Frames are coming back on stdout
    # and are read one at a time over the whole decode, so nothing can drain a
    # second pipe concurrently - and an unread pipe that fills stops the writer
    # dead. ffmpeg is quiet at -loglevel error, but "usually quiet" is not a
    # guarantee, and the failure mode is a permanent hang rather than an error.
    # A file never blocks.
    with tempfile.TemporaryFile() as stderr_file:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr_file,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        frame_bytes = width * height * 3
        try:
            while True:
                buffer = proc.stdout.read(frame_bytes)
                if not buffer or len(buffer) < frame_bytes:
                    break
                yield numpy.frombuffer(buffer, dtype=numpy.uint8).reshape(height, width, 3)
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            proc.wait()
            stderr_file.seek(0)
            # Only ever surfaced when nothing was decoded; a non-zero code after
            # a complete read is usually just the pipe closed from this end.
            decode_frames.last_error = (
                stderr_file.read() or b"").decode(errors="replace").strip()[:300]


decode_frames.last_error = ""


def face_metrics(frame, face):
    """Per-frame measurements that decide whether a face is worth averaging in."""
    import cv2
    import numpy

    x1, y1, x2, y2 = (float(v) for v in face.bounding_box)
    box_w, box_h = max(1.0, x2 - x1), max(1.0, y2 - y1)

    height, width = frame.shape[:2]
    cx1, cy1 = max(0, int(x1)), max(0, int(y1))
    cx2, cy2 = min(width, int(x2)), min(height, int(y2))
    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
        return None
    crop = frame[cy1:cy2, cx1:cx2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Variance of the Laplacian, the standard blur measure. It is scale and
    # content dependent, so it is only ever compared against the other samples
    # from this same clip, never against a fixed threshold.
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    landmark_5 = numpy.asarray(face.landmark_set.get("5/68"), dtype=numpy.float64)
    eye_a, eye_b, nose = landmark_5[0], landmark_5[1], landmark_5[2]
    eye_centre = (eye_a + eye_b) / 2.0
    interocular = float(numpy.linalg.norm(eye_b - eye_a)) or 1.0

    # Yaw proxy: on a face looking straight at the camera the nose sits over
    # the midpoint of the eyes. Turning the head slides it toward the near eye,
    # and the offset scales with the eye separation - so dividing by that makes
    # the number independent of how big the face is in the frame.
    yaw = float(nose[0] - eye_centre[0]) / interocular
    roll = float(numpy.arctan2(eye_b[1] - eye_a[1], eye_b[0] - eye_a[0]))

    # Mouth openness is what separates one moment of a talking clip from
    # another, and that is most of what makes a set of stills different views
    # rather than the same view six times.
    mouth_open = 0.0
    landmark_68 = face.landmark_set.get("68")
    if landmark_68 is not None and len(landmark_68) >= 68:
        landmark_68 = numpy.asarray(landmark_68, dtype=numpy.float64)
        mouth_open = float(abs(landmark_68[66][1] - landmark_68[62][1])) / interocular

    return {
        "box": [x1, y1, x2, y2],
        "size": float(min(box_w, box_h)),
        "detector": float(face.score_set.get("detector", 0.0)),
        "landmarker": float(face.score_set.get("landmarker", 0.0)),
        "sharpness": sharpness,
        "brightness": brightness,
        "yaw": yaw,
        "roll": roll,
        "mouth_open": mouth_open,
        "gender": face.gender,
        "embedding": numpy.asarray(face.embedding, dtype=numpy.float64),
    }


def save_crop(frame, box, path, margin, min_side):
    """
    Write a face-centred crop.

    Cropping rather than saving the whole frame does two things. FaceFusion
    takes the largest face in each source image, so the crop is what guarantees
    it takes the face that was actually chosen here rather than a bystander who
    happens to stand closer to the camera. And its detector resizes whatever it
    is handed to 640x640, so a 400 px face inside a 4K frame arrives 66 px
    across, while the same face cropped arrives at full size.
    """
    import cv2

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(max(x2 - x1, y2 - y1) * margin, min_side)

    left = int(round(max(0, cx - side / 2)))
    top = int(round(max(0, cy - side / 2)))
    right = int(round(min(width, cx + side / 2)))
    bottom = int(round(min(height, cy + side / 2)))
    if right - left < 16 or bottom - top < 16:
        return False

    # PNG rather than JPEG: this is the input to an identity embedding, and it
    # is a few hundred KB either way.
    return bool(cv2.imwrite(path, frame[top:bottom, left:right]))


def normalize(values):
    """Map a list onto 0..1 against its own range; an all-equal list maps to 1."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return [1.0] * len(values)
    return [(v - low) / (high - low) for v in values]


def score_candidates(candidates, min_face_px):
    """
    Combine the per-frame measurements into one 0..1 quality per candidate.

    Sharpness is normalised against this clip rather than against an absolute
    threshold, because the variance of the Laplacian depends on resolution,
    codec and content. The rest are absolute, because they mean the same thing
    in any clip.
    """
    sharp = normalize([c["sharpness"] for c in candidates])

    for candidate, sharp_norm in zip(candidates, sharp):
        # Saturating: twice the minimum face size is already plenty, and more
        # pixels after that buy nothing a 112x112 embedding can use.
        size = min(1.0, candidate["size"] / (2.0 * min_face_px))
        # A face turned past roughly 20 degrees contributes a profile to an
        # average that will be applied to mostly front-facing targets.
        frontal = max(0.0, 1.0 - min(1.0, abs(candidate["yaw"]) / 0.35))
        # Mid-grey is where detail survives; crushed and blown crops have none.
        exposure = max(0.0, 1.0 - abs(candidate["brightness"] - 128.0) / 128.0)

        candidate["quality"] = (
            0.30 * sharp_norm
            + 0.20 * size
            + 0.20 * frontal
            + 0.15 * candidate["detector"]
            + 0.10 * candidate["landmarker"]
            + 0.05 * exposure
        )
    return candidates


def filter_candidates(candidates, max_yaw, min_brightness, max_brightness, blur_ratio):
    """
    Drop the frames that carry no identity, before anything is ranked.

    These are gates on the raw measurements rather than on the composite
    quality, because each one means something on its own: a frame this blurred
    has no detail for the embedding whatever else is right about it, and a
    frame this dark has none either. Left in, they do not merely rank low -
    FaceFusion averages every source it is given, so a bad frame that survives
    selection drags the mean embedding with it.

    The blur gate is relative to this clip's own median rather than absolute,
    because the variance of the Laplacian depends on resolution, codec and
    content; a soft-focus clip should not reject all of itself.
    """
    import statistics

    median_sharpness = statistics.median([c["sharpness"] for c in candidates])
    sharpness_floor = median_sharpness * blur_ratio

    keep = []
    rejected = {"blurred": 0, "exposure": 0, "pose": 0}
    for candidate in candidates:
        if candidate["sharpness"] < sharpness_floor:
            rejected["blurred"] += 1
        elif not min_brightness <= candidate["brightness"] <= max_brightness:
            rejected["exposure"] += 1
        elif abs(candidate["yaw"]) > max_yaw:
            rejected["pose"] += 1
        else:
            keep.append(candidate)
            continue
        # Nothing downstream will look at it, and a pool of PNGs at 4K adds up.
        try:
            os.remove(candidate["pool_path"])
        except OSError:
            pass

    # Never gate the clip down to nothing: if every frame fails, the least-bad
    # frames are still better than refusing to run.
    if not keep:
        return sorted(candidates, key=lambda c: -c["sharpness"])[:8], rejected
    return keep, rejected


def dominant_identity(candidates, threshold):
    """
    Keep only the largest group of candidates that are the same person.

    Averaging two identities does not produce either of them, and one frame of
    somebody walking behind the subject is enough to do it. The medoid - the
    candidate with the most neighbours inside the cosine threshold - is a far
    more stable seed than the highest-scoring one, which may itself be the
    outlier.
    """
    import numpy

    if len(candidates) < 2:
        return candidates, 0

    embeddings = numpy.stack([c["embedding"] for c in candidates])
    norms = numpy.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = embeddings / norms
    similarity = unit @ unit.T

    neighbours = similarity >= threshold
    medoid = int(neighbours.sum(axis=1).argmax())
    keep = [c for c, is_near in zip(candidates, neighbours[medoid]) if is_near]
    return keep, len(candidates) - len(keep)


def select_spread(candidates, count, diversity_weight=0.5):
    """
    Pick `count` candidates that are good *and* different from each other.

    Straight top-N by quality returns the same moment over and over - the
    sharpest second of the clip - and averaging six copies of one view is worth
    no more than that one view. Farthest-point selection over pose, expression
    and time, weighted by quality, spends the picks on views that differ.

    The distance is over pose rather than over the embedding: inside one
    identity every embedding is similar by construction, so it carries almost
    no signal about which frames are different from one another.
    """
    import numpy

    if len(candidates) <= count:
        return sorted(candidates, key=lambda c: -c["quality"])

    descriptors = numpy.array([
        [c["yaw"] / 0.5, c["roll"] / 0.5, c["mouth_open"] / 0.5, c["time"]]
        for c in candidates
    ], dtype=numpy.float64)
    # Time is in seconds, and would otherwise dominate on any clip longer than
    # a couple of seconds; the three pose axes are already ~unit scale.
    span = descriptors[:, 3].max() - descriptors[:, 3].min()
    descriptors[:, 3] /= span if span > 1e-9 else 1.0

    quality = numpy.array([c["quality"] for c in candidates])
    selected = [int(quality.argmax())]

    while len(selected) < count:
        chosen = descriptors[selected]
        # Distance to the nearest already-selected pick: a candidate that sits
        # on top of something already chosen adds little however good it is.
        distances = numpy.linalg.norm(
            descriptors[:, None, :] - chosen[None, :, :], axis=2).min(axis=1)
        spread = distances / (distances.max() or 1.0)
        gain = quality + diversity_weight * spread
        gain[selected] = -numpy.inf
        selected.append(int(gain.argmax()))

    return [candidates[i] for i in selected]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int, default=400)
    parser.add_argument("--min-face-px", type=int, default=160)
    parser.add_argument("--identity-threshold", type=float, default=0.45)
    # Gates on the raw measurements. See filter_candidates.
    parser.add_argument("--max-yaw", type=float, default=0.45,
                        help="Nose offset over interocular distance; ~0.45 is a 30-degree turn.")
    parser.add_argument("--min-brightness", type=float, default=45.0)
    parser.add_argument("--max-brightness", type=float, default=215.0)
    parser.add_argument("--blur-ratio", type=float, default=0.35,
                        help="Reject frames below this fraction of the clip's median sharpness.")
    parser.add_argument("--crop-margin", type=float, default=1.8)
    parser.add_argument("--max-decode-width", type=int, default=3840)
    parser.add_argument("--decode-threads", type=int, default=4)
    parser.add_argument("--gender", default="auto", choices=["auto", "male", "female"])
    parser.add_argument("--trim-start", type=float, default=0.0)
    parser.add_argument("--trim-end", type=float, default=0.0)
    parser.add_argument("--provider", default="cuda")
    parser.add_argument("--detector-model", default="retinaface")
    parser.add_argument("--detector-size", default="640x640")
    parser.add_argument("--detector-score", type=float, default=0.5)
    parser.add_argument("--detector-angles", default="0")
    parser.add_argument("--landmarker-model", default="2dfan4")
    parser.add_argument("--landmarker-score", type=float, default=0.5)
    args = parser.parse_args()

    if not os.path.exists(args.video):
        fail(f"Source video not found: {args.video}")

    info = probe(args.video)
    if not info or not info["width"]:
        fail(f"ffprobe could not read a video stream from {os.path.basename(args.video)}.")

    duration = info["duration"]
    if args.trim_end and args.trim_end > args.trim_start:
        duration = min(duration, args.trim_end - args.trim_start)
    elif args.trim_start:
        duration = max(0.0, duration - args.trim_start)
    if duration <= 0:
        duration = 1.0

    # Sample at the requested rate, but never decode more frames than asked
    # for: the cost is one detect + one landmark + one embedding + one classify
    # per frame, and a five-minute clip at 2 fps is 600 of them.
    sample_fps = args.sample_fps
    if duration * sample_fps > args.max_samples:
        sample_fps = args.max_samples / duration
    sample_fps = max(0.05, sample_fps)
    total_expected = max(1, int(duration * sample_fps))

    decode_w, decode_h = info["width"], info["height"]
    if decode_w > args.max_decode_width:
        scale = args.max_decode_width / decode_w
        decode_w = args.max_decode_width // 2 * 2
        decode_h = max(2, int(round(info["height"] * scale)) // 2 * 2)

    os.makedirs(args.out_dir, exist_ok=True)
    pool_dir = os.path.join(args.out_dir, "pool")
    os.makedirs(pool_dir, exist_ok=True)

    init_facefusion(args)

    from facefusion.face_creator import get_many_faces

    candidates = []
    index = 0
    no_face = 0
    too_small = 0
    wrong_gender = 0

    for frame in decode_frames(args.video, decode_w, decode_h, sample_fps,
                               args.trim_start, args.trim_end, args.decode_threads):
        index += 1
        if index % 5 == 0:
            emit(f"FACESRC_PROGRESS {min(index, total_expected)} {total_expected}")

        faces = get_many_faces([frame])
        if not faces:
            no_face += 1
            continue

        # Largest face in the frame, which is the rule FaceFusion itself
        # applies to a source image - so what is measured here is the face it
        # would actually have used.
        face = max(faces, key=lambda f: (f.bounding_box[2] - f.bounding_box[0])
                   * (f.bounding_box[3] - f.bounding_box[1]))
        metrics = face_metrics(frame, face)
        if metrics is None:
            no_face += 1
            continue
        if metrics["size"] < args.min_face_px:
            too_small += 1
            continue
        if args.gender != "auto" and metrics["gender"] != args.gender:
            wrong_gender += 1
            continue

        metrics["time"] = args.trim_start + (index - 1) / sample_fps
        metrics["pool_path"] = os.path.join(pool_dir, f"cand_{index:05d}.png")
        # Written out now and mostly thrown away later; keeping the frames in
        # memory instead would be 25 MB each.
        if save_crop(frame, metrics["box"], metrics["pool_path"],
                     args.crop_margin, args.min_face_px * args.crop_margin):
            candidates.append(metrics)

    emit(f"FACESRC_PROGRESS {total_expected} {total_expected}")

    if not candidates:
        shutil.rmtree(pool_dir, ignore_errors=True)
        reasons = []
        if no_face:
            reasons.append(f"{no_face} sampled frame(s) had no detectable face")
        if too_small:
            reasons.append(f"{too_small} had a face smaller than {args.min_face_px}px")
        if wrong_gender:
            reasons.append(f"{wrong_gender} were filtered out by gender={args.gender}")
        if not reasons:
            detail = decode_frames.last_error
            reasons.append("no frames were decoded" + (f" ({detail})" if detail else ""))
        fail("No usable face was found in the source video: " + "; ".join(reasons) +
             ". Try a clip where the face is larger and better lit, or lower "
             "Source Video Min Face px in Settings.")

    candidates, gated = filter_candidates(
        candidates, args.max_yaw, args.min_brightness, args.max_brightness, args.blur_ratio)
    candidates = score_candidates(candidates, args.min_face_px)
    candidates, dropped = dominant_identity(candidates, args.identity_threshold)
    picks = select_spread(candidates, max(1, args.count))
    picks.sort(key=lambda c: c["time"])

    saved = []
    for position, candidate in enumerate(picks, start=1):
        final_path = os.path.normpath(os.path.join(args.out_dir, f"face_{position:02d}.png"))
        shutil.move(candidate["pool_path"], final_path)
        saved.append({
            "path": final_path,
            "time": round(candidate["time"], 2),
            "quality": round(candidate["quality"], 3),
            "size_px": int(candidate["size"]),
            "yaw": round(candidate["yaw"], 3),
            "sharpness": round(candidate["sharpness"], 1),
            "detector": round(candidate["detector"], 3),
            "gender": candidate["gender"],
        })

    shutil.rmtree(pool_dir, ignore_errors=True)

    manifest = {
        "video": os.path.abspath(args.video),
        "sampled": index,
        "sample_fps": round(sample_fps, 3),
        "candidates": len(candidates) + dropped,
        "rejected_no_face": no_face,
        "rejected_too_small": too_small,
        "rejected_gender": wrong_gender,
        "rejected_blurred": gated["blurred"],
        "rejected_exposure": gated["exposure"],
        "rejected_pose": gated["pose"],
        "rejected_other_identity": dropped,
        "faces": saved,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)

    emit("FACESRC_RESULT " + json.dumps(manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
