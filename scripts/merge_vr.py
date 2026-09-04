import subprocess
import sys
import os
import argparse


def merge_videos(left_path, right_path, output_path, layout="sbs", encoder="libx265", crf=24,
                 preset="medium", add_vr_metadata=True, fps=None):
    if not os.path.exists(left_path):
        print(f"ERROR: Left eye video not found: {left_path}")
        sys.exit(1)
    if not os.path.exists(right_path):
        print(f"ERROR: Right eye video not found: {right_path}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    left_info = probe_video(left_path)
    right_info = probe_video(right_path)

    if layout == "sbs":
        hpad = ""
        filter_complex = f"[0:v][1:v]hstack=2[vout]"
        output_width = left_info["width"] + right_info["width"]
        output_height = left_info["height"]
        print(f"Merging side-by-side: {output_width}x{output_height}")

    elif layout == "tb":
        filter_complex = f"[0:v][1:v]vstack=2[vout]"
        output_width = left_info["width"]
        output_height = left_info["height"] + right_info["height"]
        print(f"Merging top-bottom: {output_width}x{output_height}")

    else:
        print(f"ERROR: Unknown layout: {layout}")
        sys.exit(1)

    stereo_meta = ""
    if add_vr_metadata and layout == "sbs":
        stereo_meta = '-metadata:s:v:0 "stereo_mode=fb"'
    elif add_vr_metadata and layout == "tb":
        stereo_meta = '-metadata:s:v:0 "stereo_mode=tb"'

    fps_arg = ""
    if fps:
        fps_arg = f"-r {fps}"

    cmd = (
        f'ffmpeg -y '
        f'-i "{left_path}" -i "{right_path}" '
        f'-filter_complex "{filter_complex}" '
        f'{fps_arg} '
        f'-map "[vout]" -map 0:a '
        f'-c:v {encoder} -crf {crf} -preset {preset} '
        f'-c:a aac -b:a 192k '
        f'{stereo_meta} '
        f'"{output_path}"'
    )

    print(f"\nMerge command:")
    print(f"  Encoder: {encoder} (CRF {crf}, preset {preset})")
    print(f"  Output:  {output_path}")
    print(f"  VR metadata: {'yes' if add_vr_metadata else 'no'}")

    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: Merge failed")
        sys.exit(1)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nMerge complete. Output size: {file_size:.1f} MB")
    return output_path


def probe_video(path):
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=x "{path}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    parts = result.stdout.strip().split("x")
    width = int(parts[0])
    height = int(parts[1])
    return {"width": width, "height": height}


def postprocess(input_path, output_path, denoise=False, color_correct=False,
                target_framerate=None, encoder="libx265", crf=24, preset="medium"):
    vf = []

    if denoise:
        vf.append("hqdn3d=strong=1.5:fast=1.5")

    if color_correct:
        vf.append("eq=brightness=0.02:contrast=1.05:saturation=1.05")
        vf.append("unsharp=5:5:1.5")

    if target_framerate:
        vf.append(f"fps={target_framerate}")

    vf_str = ",".join(vf) if vf else "null"

    cmd = (
        f'ffmpeg -y '
        f'-i "{input_path}" '
        f'-vf "{vf_str}" '
        f'-c:v {encoder} -crf {crf} -preset {preset} '
        f'-c:a aac -b:a 192k '
        f'"{output_path}"'
    )

    print(f"Post-processing...")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: Post-processing failed")
        sys.exit(1)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Merge left/right eye videos into stereoscopic VR output")
    parser.add_argument("left", help="Path to left eye video")
    parser.add_argument("right", help="Path to right eye video")
    parser.add_argument("-o", "--output", required=True, help="Output VR video path")
    parser.add_argument("-l", "--layout", default="sbs", choices=["sbs", "tb"], help="Stereo layout")
    parser.add_argument("--encoder", default="libx265", help="Video encoder (libx265, h264_nvenc, etc)")
    parser.add_argument("--crf", type=int, default=24, help="CRF quality (lower=better)")
    parser.add_argument("--preset", default="medium", help="Encoding preset")
    parser.add_argument("--no-vr-metadata", action="store_true", help="Skip VR stereo metadata")
    parser.add_argument("--fps", type=float, help="Output FPS override")

    args = parser.parse_args()
    merge_videos(
        left_path=args.left,
        right_path=args.right,
        output_path=args.output,
        layout=args.layout,
        encoder=args.encoder,
        crf=args.crf,
        preset=args.preset,
        add_vr_metadata=not args.no_vr_metadata,
        fps=args.fps,
    )
    print("\nMerge complete.")


if __name__ == "__main__":
    main()
