"""
Find the distinct people in a target video, for the app's character picker.

Samples a handful of frames spread across the chosen portion of the target
(the left eye only, for a VR target - the right eye holds the same people),
detects every face, and groups the faces by identity. Each group is one
person: the app shows one tile per person, and whoever the user assigns to
Face 1 or Face 2 is saved as that face's reference crops.

A person is saved as several crops from different moments rather than one,
because a single crop is one pose; matching a turned head against the average
of a few views holds on where a single frontal reference lets go. Every crop is
written with an .npy of its embedding beside it, which is what the swap
actually matches against (scripts/facefusion_addons/vrdeep_slots.py) - the
.png is there so a person can see who was picked.

Runs in FaceFusion's venv, like face_source_worker.py, whose FaceFusion setup
and crop writer it reuses.

Output protocol on stdout:
    CHARS_PROGRESS <done> <total>
    CHARS_RESULT <json>
    CHARS_ERROR <message>        (FACESRC_ERROR from the shared setup code)
"""

import argparse
import json
import os
import subprocess
import sys

import numpy

import face_source_worker as fsw

# 1 - cosine between two faces that are taken to be the same person. Looser
# than the swap's own match distance would be wrong here: merging two people
# into one tile is worse than showing one person as two tiles, since the user
# can simply assign both tiles.
CLUSTER_DISTANCE = 0.5
REFS_PER_PERSON = 4


def emit(line):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def fail(message):
    emit("CHARS_ERROR " + " ".join(str(message).split()))
    sys.exit(2)


def grab_frame(video, seconds, layout, width, height, max_width):
    """One BGR frame at `seconds`, cropped to the left/top eye for VR."""
    filters = []
    eye_w, eye_h = width, height
    if layout == "sbs":
        filters.append("crop=iw/2:ih:0:0")
        eye_w = width // 2
    elif layout == "tb":
        filters.append("crop=iw:ih/2:0:0")
        eye_h = height // 2
    out_w, out_h = eye_w, eye_h
    if out_w > max_width:
        out_h = max(2, int(round(eye_h * max_width / eye_w)) // 2 * 2)
        out_w = max_width
        filters.append(f"scale={out_w}:{out_h}")
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{seconds:.3f}", "-i", video, "-frames:v", "1"]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    result = subprocess.run(cmd, capture_output=True, timeout=120,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    expected = out_w * out_h * 3
    if result.returncode != 0 or len(result.stdout) < expected:
        return None
    return numpy.frombuffer(result.stdout[:expected], dtype=numpy.uint8).reshape(out_h, out_w, 3)


def face_size(face):
    x1, y1, x2, y2 = face.bounding_box
    return float(max(x2 - x1, y2 - y1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layout", default="none", choices=["none", "sbs", "tb"])
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--min-face-px", type=int, default=48)
    parser.add_argument("--max-decode-width", type=int, default=4096)
    parser.add_argument("--provider", default="cuda")
    parser.add_argument("--detector-model", default="retinaface")
    parser.add_argument("--detector-size", default="640x640")
    parser.add_argument("--detector-score", type=float, default=0.5)
    parser.add_argument("--detector-angles", default="0")
    parser.add_argument("--landmarker-model", default="2dfan4")
    parser.add_argument("--landmarker-score", type=float, default=0.5)
    args = parser.parse_args()

    if not os.path.exists(args.video):
        fail(f"Target video not found: {args.video}")
    info = fsw.probe(args.video)
    if not info or not info["width"]:
        fail(f"ffprobe could not read a video stream from {os.path.basename(args.video)}.")

    start = max(0.0, args.start or 0.0)
    end = args.end if args.end and args.end > start else (info["duration"] or start + 1.0)
    count = max(1, args.samples)
    # Midpoints of equal slices, so neither the first nor the last frame - often
    # a fade - is one of the samples.
    times = [start + (end - start) * (i + 0.5) / count for i in range(count)]

    os.makedirs(args.out_dir, exist_ok=True)
    fsw.init_facefusion(argparse.Namespace(
        video=args.video, provider=args.provider, detector_model=args.detector_model,
        detector_size=args.detector_size, detector_score=args.detector_score,
        detector_angles=args.detector_angles, landmarker_model=args.landmarker_model,
        landmarker_score=args.landmarker_score,
    ))
    from facefusion.face_creator import get_many_faces

    people = []   # {"centroid": vec, "members": [...]}
    for index, seconds in enumerate(times, start=1):
        emit(f"CHARS_PROGRESS {index - 1} {count}")
        frame = grab_frame(args.video, seconds, args.layout, info["width"], info["height"],
                           args.max_decode_width)
        if frame is None:
            continue
        for face_index, face in enumerate(get_many_faces([frame])):
            size = face_size(face)
            if size < args.min_face_px:
                continue
            embedding = numpy.asarray(face.embedding_norm, dtype=numpy.float64)
            crop_path = os.path.join(args.out_dir, f"s{index:02d}_f{face_index:02d}.png")
            if not fsw.save_crop(frame, [float(v) for v in face.bounding_box], crop_path,
                                 1.8, 0):
                continue
            member = {"time": round(seconds, 2), "size": size, "embedding": embedding,
                      "crop": crop_path, "gender": face.gender, "age": face.age}
            best, best_distance = None, CLUSTER_DISTANCE
            for person in people:
                distance = 1.0 - float(numpy.dot(embedding, person["centroid"]))
                if distance < best_distance:
                    best, best_distance = person, distance
            if best is None:
                people.append({"centroid": embedding, "members": [member]})
            else:
                best["members"].append(member)
                mean = numpy.mean([m["embedding"] for m in best["members"]], axis=0)
                best["centroid"] = mean / (numpy.linalg.norm(mean) or 1.0)
    emit(f"CHARS_PROGRESS {count} {count}")

    if not people:
        fail(f"No faces were found in {count} frames sampled across the clip. "
             f"Try a different portion of the video.")

    people.sort(key=lambda p: (len(p["members"]), max(m["size"] for m in p["members"])),
                reverse=True)
    result = []
    for number, person in enumerate(people, start=1):
        members = sorted(person["members"], key=lambda m: m["size"], reverse=True)
        refs, seen_times = [], set()
        for member in members:
            if member["time"] in seen_times:
                continue
            seen_times.add(member["time"])
            base = os.path.join(args.out_dir, f"person{number:02d}_{len(refs) + 1}")
            os.replace(member["crop"], base + ".png")
            numpy.save(base + ".npy", member["embedding"].astype(numpy.float32))
            refs.append(base + ".png")
            if len(refs) >= REFS_PER_PERSON:
                break
        genders = [m["gender"] for m in members if m["gender"]]
        ages = [m["age"] for m in members if isinstance(m["age"], range)]
        result.append({
            "person": number,
            "appearances": len({m["time"] for m in members}),
            "gender": max(set(genders), key=genders.count) if genders else "",
            "age": f"{ages[0].start}-{ages[0].stop}" if ages else "",
            "size_px": int(members[0]["size"]),
            "thumb": refs[0],
            "refs": refs,
        })

    # Crops that did not make it into anyone's references.
    for name in os.listdir(args.out_dir):
        if name.startswith("s") and name.endswith(".png") and "_f" in name:
            try:
                os.remove(os.path.join(args.out_dir, name))
            except OSError:
                pass

    emit("CHARS_RESULT " + json.dumps({"samples": count, "people": result}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
