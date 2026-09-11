"""
Answer "why was nothing swapped" without running the whole pipeline.

A swap that produces an untouched frame is almost never a broken swapper. It is
the frame never reaching the swapper at all, and there are only a few places
that can happen:

  1. the detector found no face                  -> detector size / score
  2. it found one, too weak to keep              -> face_detector_score
  3. it kept one, the landmarker rejected it     -> face_landmarker_score
  4. it kept one, the selector filtered it out   -> face_selector_gender/race

FaceFusion reports none of this in a headless run - it just writes a video that
looks like the input. This samples frames, runs the real detector against them
at one or more detector sizes, and prints which of the four it is.

The detector size matters far more on VR footage than anywhere else: FaceFusion
scales the frame down to the detector size before looking at it, so one eye of
an 8K VR clip goes 4096x4096 -> 640x640, a 6.4x reduction. A face 300 px wide
in the source is 47 px by the time RetinaFace sees it.

Run it with FaceFusion's interpreter, from the project root:

    facefusion/venv/Scripts/python.exe scripts/diagnose_faces.py \
        --video output/temp/left.mp4 --source IMG_20210909_175031.jpg

    --sizes 640x640 1024x1024 ...   detector sizes to compare
    --samples 12                    frames to sample across the clip
    --provider cpu                  keep off the GPU while a run is going
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACEFUSION_DIR = os.path.join(PROJECT_DIR, "facefusion")


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:format=duration",
         "-of", "json", path],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        fail(f"ffprobe could not read {path}")
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    return (int(stream["width"]), int(stream["height"]),
            float(data["format"].get("duration") or 0.0))


def read_frame(path, timestamp, width, height):
    """One BGR frame at `timestamp`, as a numpy array."""
    import numpy

    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(timestamp), "-i", path,
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True, timeout=300,
    )
    expected = width * height * 3
    if len(result.stdout) < expected:
        return None
    return numpy.frombuffer(result.stdout[:expected], numpy.uint8).reshape(height, width, 3)


def init_facefusion(args, detector_size):
    """
    Bring FaceFusion's state up far enough for its face analyser to run.

    Same approach as face_source_worker.init_facefusion: go through the real
    argument parser so every default we do not care about comes from FaceFusion
    and cannot drift out of step with it.
    """
    from facefusion import logger, state_manager
    from facefusion.args import apply_args
    from facefusion.program import create_program

    program = create_program()
    argv = [
        "headless-run",
        "--source-paths", args.source or args.video,
        "--target-path", args.video,
        "--output-path", args.video,
        "--execution-providers", *str(args.provider).split(),
        "--face-detector-model", args.detector_model,
        "--face-detector-size", detector_size,
        "--face-detector-score", str(args.detector_score),
        "--face-detector-angles", *str(args.detector_angles).split(),
        "--face-landmarker-model", args.landmarker_model,
        "--face-landmarker-score", str(args.landmarker_score),
    ]
    apply_args(vars(program.parse_args(argv)), state_manager.init_item)
    logger.init("error")

    import facefusion.face_classifier as face_classifier
    import facefusion.face_detector as face_detector
    import facefusion.face_landmarker as face_landmarker
    import facefusion.face_recognizer as face_recognizer

    for name, module in (("detector", face_detector), ("landmarker", face_landmarker),
                         ("recognizer", face_recognizer), ("classifier", face_classifier)):
        if not module.pre_check():
            fail(f"the face {name} model is missing and could not be fetched")
    return state_manager


def describe(faces, frame_width):
    """Largest face's size, score and classification, or None."""
    if not faces:
        return None
    best = max(faces, key=lambda f: (f.bounding_box[2] - f.bounding_box[0]))
    x1, y1, x2, y2 = best.bounding_box
    return {
        "px": int(x2 - x1),
        "pct": (x2 - x1) / frame_width * 100,
        "detector": float(best.score_set.get("detector", 0.0)),
        "landmarker": float(best.score_set.get("landmarker", 0.0)),
        "gender": best.gender,
        "age": best.age,
        "count": len(faces),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="one eye, e.g. output/temp/left.mp4")
    parser.add_argument("--source", help="the source face photo, to check the gender filter")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--sizes", nargs="+", default=["640x640", "1024x1024"])
    parser.add_argument("--provider", default="cpu",
                        help="'cpu' (default) leaves the GPU alone; 'cuda' is faster")
    parser.add_argument("--detector-model", default="retinaface")
    parser.add_argument("--detector-score", type=float, default=0.5)
    parser.add_argument("--detector-angles", default="0")
    parser.add_argument("--landmarker-model", default="2dfan4")
    parser.add_argument("--landmarker-score", type=float, default=0.5)
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        fail(f"{args.video} does not exist")
    sys.path.insert(0, FACEFUSION_DIR)
    os.chdir(FACEFUSION_DIR)

    video = os.path.join(PROJECT_DIR, args.video) if not os.path.isabs(args.video) else args.video
    width, height, duration = probe(video)
    print(f"Target : {os.path.basename(video)}  {width}x{height}, {duration:.0f}s")
    if args.source:
        source = args.source if os.path.isabs(args.source) else os.path.join(PROJECT_DIR, args.source)
        print(f"Source : {os.path.basename(source)}")
    print(f"Filters: detector_score >= {args.detector_score}, "
          f"landmarker_score >= {args.landmarker_score}")
    print()

    # Sample away from the very start and end, where VR clips often have titles.
    stamps = [duration * (0.1 + 0.8 * i / max(1, args.samples - 1))
              for i in range(args.samples)]

    for detector_size in args.sizes:
        state_manager = init_facefusion(args, detector_size)
        from facefusion.face_creator import get_many_faces

        scale = width / int(detector_size.split("x")[0])
        print(f"--- detector size {detector_size} "
              f"(frame is scaled down {scale:.1f}x before detection) ---")

        source_gender = None
        if args.source:
            from facefusion.vision import read_static_image

            source_faces = get_many_faces([read_static_image(source)])
            if not source_faces:
                print("  SOURCE: no face found in the source photo at this detector size.")
            else:
                source_gender = source_faces[0].gender
                print(f"  SOURCE: face found, classified {source_gender}. "
                      f"With face_selector_gender='auto' only {source_gender} "
                      f"target faces will be swapped.")

        hits = []
        for stamp in stamps:
            frame = read_frame(video, stamp, width, height)
            if frame is None:
                continue
            found = describe(get_many_faces([frame]), width)
            hits.append((stamp, found))
            if found:
                flag = ""
                if source_gender and found["gender"] != source_gender:
                    flag = f"  <-- FILTERED OUT: target is {found['gender']}"
                elif found["landmarker"] < args.landmarker_score:
                    flag = "  <-- landmark score below threshold"
                print(f"  {stamp:6.1f}s  {found['count']} face(s), largest {found['px']}px "
                      f"({found['pct']:.1f}% of width), detector {found['detector']:.2f}, "
                      f"landmarker {found['landmarker']:.2f}, {found['gender']}{flag}")
            else:
                print(f"  {stamp:6.1f}s  no face detected")

        detected = [f for _, f in hits if f]
        usable = [f for f in detected
                  if f["landmarker"] >= args.landmarker_score
                  and (not source_gender or f["gender"] == source_gender)]
        print(f"  => detected in {len(detected)}/{len(hits)} frames, "
              f"{len(usable)}/{len(hits)} would actually be swapped")
        if detected:
            print(f"  => median face width {statistics.median(f['px'] for f in detected):.0f}px "
                  f"in the source frame, "
                  f"{statistics.median(f['px'] for f in detected) / scale:.0f}px "
                  f"as the detector sees it")
        print()

        # Each size needs its own state and its own inference pool.
        for module_name in ("face_detector", "face_landmarker",
                            "face_recognizer", "face_classifier"):
            module = sys.modules.get(f"facefusion.{module_name}")
            if module and hasattr(module, "clear_inference_pool"):
                module.clear_inference_pool()

    print("Reading the result:")
    print("  no face detected              -> raise the detector size (the big lever on VR)")
    print("  detected but low detector     -> lower Detector Score")
    print("  detected, landmarker below    -> lower Landmarker Score")
    print("  FILTERED OUT                  -> set Selector Gender explicitly instead of 'auto'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
