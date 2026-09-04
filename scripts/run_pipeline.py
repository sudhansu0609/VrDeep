import subprocess
import sys
import os
import argparse
import shutil
import time


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
FACEFUSION_DIR = os.path.join(PROJECT_DIR, "facefusion")
PYTHON = os.path.join(FACEFUSION_DIR, "venv", "Scripts", "python.exe")


def step_split(input_video, output_dir, layout, fps=None, trim_start=None, trim_end=None):
    print("=" * 60)
    print("STEP 1/4: Splitting stereoscopic video")
    print("=" * 60)

    left_dir = os.path.join(output_dir, "left_eye")
    right_dir = os.path.join(output_dir, "right_eye")

    cmd = [
        PYTHON, os.path.join(SCRIPTS_DIR, "split_vr.py"),
        input_video,
        "-o", output_dir,
        "-l", layout,
    ]

    if fps:
        cmd.extend(["--fps", str(fps)])
    if trim_start:
        cmd.extend(["--trim-start", str(trim_start)])
    if trim_end:
        cmd.extend(["--trim-end", str(trim_end)])

    result = subprocess.run(cmd, shell=False)
    if result.returncode != 0:
        print("ERROR: Split failed")
        sys.exit(1)

    left_video = os.path.join(output_dir, "left.mp4")
    right_video = os.path.join(output_dir, "right.mp4")

    if not os.path.exists(left_video) or not os.path.exists(right_video):
        print("ERROR: Split output files not found")
        sys.exit(1)

    print(f"  Left:  {left_video}")
    print(f"  Right: {right_video}")
    return left_video, right_video


def step_facefusion(source_faces, left_video, right_video, output_dir, config):
    print("\n" + "=" * 60)
    print("STEP 2/4: Running FaceFusion on each eye")
    print("=" * 60)

    left_out = os.path.join(output_dir, "left_swapped.mp4")
    right_out = os.path.join(output_dir, "right_swapped.mp4")
    temp_left = os.path.join(output_dir, "ff_temp_left")
    temp_right = os.path.join(output_dir, "ff_temp_right")

    for label, target, output, temp in [
        ("LEFT EYE", left_video, left_out, temp_left),
        ("RIGHT EYE", right_video, right_out, temp_right),
    ]:
        print(f"\n--- Processing {label} ---")
        cmd = [
            PYTHON, os.path.join(SCRIPTS_DIR, "run_facefusion.py"),
            # Several stills, not one: FaceFusion averages their embeddings
            # into a single identity. A source video resolves to a list of
            # them, and a single photo to a list of one.
            *source_faces, target,
            "-o", output,
            "-t", temp,
            "--swapper-model", config["swapper_model"],
            "--enhancer-model", config["enhancer_model"],
            "--enhancer-blend", str(config["enhancer_blend"]),
            "--detector-model", config["detector_model"],
            "--detector-size", config["detector_size"],
            "--detector-score", str(config["detector_score"]),
            "--encoder", config["encoder"],
            "--quality", str(config["quality"]),
            "--provider", config["provider"],
            "--threads", str(config["threads"]),
            "--mask-types", config["mask_types"],
            "--mask-blur", str(config["mask_blur"]),
            "--mask-padding", config["mask_padding"],
        ]

        if config.get("frame_enhancer"):
            cmd.extend(["--frame-enhancer", config["frame_enhancer"]])
            cmd.extend(["--frame-enhancer-blend", str(config.get("frame_enhancer_blend", 80))])

        if config.get("fps"):
            cmd.extend(["--fps", str(config["fps"])])

        result = subprocess.run(cmd, shell=False)
        if result.returncode != 0:
            print(f"ERROR: FaceFusion failed for {label}")
            sys.exit(1)

    return left_out, right_out


def step_merge(left_video, right_video, output_path, layout, encoder, crf, preset):
    print("\n" + "=" * 60)
    print("STEP 3/4: Merging into stereoscopic VR video")
    print("=" * 60)

    cmd = [
        PYTHON, os.path.join(SCRIPTS_DIR, "merge_vr.py"),
        left_video, right_video,
        "-o", output_path,
        "-l", layout,
        "--encoder", encoder,
        "--crf", str(crf),
        "--preset", preset,
    ]

    result = subprocess.run(cmd, shell=False)
    if result.returncode != 0:
        print("ERROR: Merge failed")
        sys.exit(1)

    return output_path


def step_cleanup(temp_dir, keep_temp=False):
    print("\n" + "=" * 60)
    print("STEP 4/4: Cleanup")
    print("=" * 60)

    if keep_temp:
        print("  Keeping temp files (as requested)")
        return

    items_to_clean = ["ff_temp_left", "ff_temp_right", "left.mp4", "right.mp4",
                      "left_swapped.mp4", "right_swapped.mp4"]

    for item in items_to_clean:
        item_path = os.path.join(temp_dir, item)
        if os.path.exists(item_path):
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)
                print(f"  Removed dir: {item}")
            else:
                os.remove(item_path)
                print(f"  Removed file: {item}")

    print("  Cleanup complete.")


def load_config(config_path):
    defaults = {
        "swapper_model": "inswapper_128_fp16",
        "enhancer_model": "codeformer",
        "enhancer_blend": 80,
        "frame_enhancer": None,
        "frame_enhancer_blend": 80,
        "detector_model": "retinaface",
        "detector_size": "640x640",
        "detector_score": 0.5,
        "encoder": "h264_nvenc",
        "quality": 20,
        "provider": "cuda",
        "threads": 8,
        "mask_types": "box occlusion",
        "mask_blur": 0.7,
        "mask_padding": "0 5 0 5",
        "fps": None,
    }

    if config_path and os.path.exists(config_path):
        import json
        with open(config_path) as f:
            user_config = json.load(f)
        defaults.update(user_config)

    return defaults


def main():
    parser = argparse.ArgumentParser(
        description="VR Deepfake Pipeline - Full automated workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (SBS video):
  python run_pipeline.py source.jpg target.mp4

  # With config file:
  python run_pipeline.py source.jpg target.mp4 -c pipeline_config.json

  # Trim first 30 seconds, top-bottom layout:
  python run_pipeline.py source.jpg target.mp4 -l tb --trim-start 0 --trim-end 30

  # Keep temp files for debugging:
  python run_pipeline.py source.jpg target.mp4 --keep-temp
        """
    )

    parser.add_argument("source_face",
                        help="Source face: an image, several joined by ';', or a short "
                             "video clip of the face (the best stills are extracted "
                             "from it automatically).")
    parser.add_argument("target_video", help="Path to target VR video")
    parser.add_argument("-o", "--output", default=None, help="Output video path (default: output/final/deepfake_<timestamp>.mp4)")
    parser.add_argument("-l", "--layout", default="sbs", choices=["sbs", "tb"], help="Stereo layout")
    parser.add_argument("-c", "--config", default=None, help="JSON config file for FaceFusion settings")
    parser.add_argument("--trim-start", type=float, default=None, help="Start time in seconds")
    parser.add_argument("--trim-end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    parser.add_argument("--refresh-source-faces", action="store_true",
                        help="Re-run the source-video analysis instead of reusing the "
                             "stills cached under output/source_faces/.")
    parser.add_argument("--merge-encoder", default="libx265", help="Final merge encoder")
    parser.add_argument("--merge-crf", type=int, default=24, help="Final merge CRF")
    parser.add_argument("--merge-preset", default="medium", help="Final merge preset")

    args = parser.parse_args()

    # A ';'-joined list and a video clip are both legal here, so the real
    # per-entry check happens in engine.resolve_source_faces below.
    if not any(os.path.exists(p.strip()) for p in args.source_face.replace(";", ",").split(",")):
        print(f"ERROR: Source face not found: {args.source_face}")
        sys.exit(1)
    if not os.path.exists(args.target_video):
        print(f"ERROR: Target video not found: {args.target_video}")
        sys.exit(1)

    config = load_config(args.config)
    temp_dir = os.path.join(PROJECT_DIR, "output", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_DIR, "output", "final"), exist_ok=True)

    if not args.output:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(PROJECT_DIR, "output", "final", f"deepfake_{ts}.mp4")

    print("\n" + "#" * 60)
    print("#  VR DEEPFAKE PIPELINE")
    print("#" * 60)
    print(f"  Source face : {args.source_face}")
    print(f"  Target video: {args.target_video}")
    print(f"  Layout      : {args.layout.upper()}")
    print(f"  Output      : {args.output}")
    print(f"  Swapper     : {config['swapper_model']}")
    print(f"  Enhancer    : {config['enhancer_model']}")
    print(f"  Provider    : {config['provider']}")
    print("#" * 60)

    start_time = time.time()

    # A clip is a legitimate source here as well as in the GUI. Resolved once,
    # up front, so both eyes swap against the same identity and a clip with no
    # usable face fails before the split rather than after it.
    import engine
    source_faces = engine.resolve_source_faces(
        args.source_face, config, log_cb=lambda m: print(f"  {m}"),
        refresh=args.refresh_source_faces,
    )

    left_video, right_video = step_split(
        args.target_video, temp_dir, args.layout,
        config.get("fps"), args.trim_start, args.trim_end
    )

    left_swapped, right_swapped = step_facefusion(
        source_faces, left_video, right_video, temp_dir, config
    )

    final_output = step_merge(
        left_swapped, right_swapped, args.output,
        args.layout, args.merge_encoder, args.merge_crf, args.merge_preset
    )

    step_cleanup(temp_dir, args.keep_temp)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE")
    print(f"  Output: {final_output}")
    print(f"  Time  : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
