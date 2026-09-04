import subprocess
import sys
import os
import argparse


def split_video(input_path, output_dir, layout="sbs", fps=None, trim_start=None, trim_end=None):
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    left_out = os.path.join(output_dir, "left.mp4")
    right_out = os.path.join(output_dir, "right.mp4")

    if layout == "sbs":
        probe = probe_video(input_path)
        width = probe["width"]
        half_w = width // 2

        filters_left = f"crop={half_w}:{probe['height']}:0:0"
        filters_right = f"crop={half_w}:{probe['height']}:{half_w}:0"

    elif layout == "tb":
        probe = probe_video(input_path)
        height = probe["height"]
        half_h = height // 2

        filters_left = f"crop={probe['width']}:{half_h}:0:0"
        filters_right = f"crop={probe['width']}:{half_h}:0:{half_h}"

    else:
        print(f"ERROR: Unknown layout: {layout}. Use 'sbs' or 'tb'.")
        sys.exit(1)

    base = "-y -i \"{input_path}\" -vf \"{filters}\" -c:v libx264 -preset fast -crf 18 -c:a aac -b:a 192k"

    ff_args_left = f"{base}\" \"{left_out}\"".format(
        filters=filters_left
    )
    ff_args_right = f"{base}\" \"{right_out}\"".format(
        filters=filters_right
    )

    if fps:
        ff_args_left = ff_args_left.replace("-c:v", f"-r {fps} -c:v")
        ff_args_right = ff_args_right.replace("-c:v", f"-r {fps} -c:v")

    if trim_start is not None and trim_end is not None:
        ff_args_left = ff_args_left.replace("-y -i", f"-y -ss {trim_start} -to {trim_end} -i")
        ff_args_right = ff_args_right.replace("-y -i", f"-y -ss {trim_start} -to {trim_end} -i")

    print(f"Splitting {layout.upper()} video...")
    print(f"  Left eye  -> {left_out}")
    print(f"  Right eye -> {right_out}")

    for label, cmd in [("Left eye", ff_args_left), ("Right eye", ff_args_right)]:
        print(f"\nProcessing {label}...")
        result = subprocess.run(
            f"ffmpeg {cmd}",
            shell=True,
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"ERROR: Failed processing {label}")
            sys.exit(1)

    return left_out, right_out


def probe_video(path):
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of csv=p=0:s=x "{path}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    parts = result.stdout.strip().split("x")
    width = int(parts[0])
    height = int(parts[1])
    return {"width": width, "height": height}


def main():
    parser = argparse.ArgumentParser(description="Split stereoscopic VR video into left/right eye streams")
    parser.add_argument("input", help="Path to stereoscopic VR video")
    parser.add_argument("-o", "--output-dir", default="output/temp", help="Output directory for split videos")
    parser.add_argument("-l", "--layout", default="sbs", choices=["sbs", "tb"], help="Stereo layout: side-by-side or top-bottom")
    parser.add_argument("--fps", type=float, help="Override output FPS")
    parser.add_argument("--trim-start", type=float, help="Start time in seconds")
    parser.add_argument("--trim-end", type=float, help="End time in seconds")

    args = parser.parse_args()
    split_video(args.input, args.output_dir, args.layout, args.fps, args.trim_start, args.trim_end)
    print("\nSplit complete.")


if __name__ == "__main__":
    main()
