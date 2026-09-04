import subprocess
import sys
import os
import argparse
import json
import glob


def run_facefusion(source_path, target_path, output_path, temp_path, face_swapper_model="inswapper_128_fp16",
                   face_enhancer_model="codeformer", face_enhancer_blend=80,
                   face_detector_model="retinaface", face_detector_size="640x640",
                   face_detector_score=0.5, output_video_encoder="h264_nvenc",
                   output_video_quality=20, output_video_fps=None,
                   execution_providers="cuda", execution_thread_count=8,
                   face_selector_mode="one", face_selector_order="large-small",
                   face_mask_types="box occlusion", face_mask_blur=0.7,
                   face_mask_padding="0 5 0 5", frame_enhancer_model=None,
                   frame_enhancer_blend=None):

    # FaceFusion averages the ArcFace embeddings of every source image it is
    # given, so several stills of one person are a better identity than one -
    # which is also how a source *video* reaches this script: the caller runs
    # engine.resolve_source_faces() first and passes the stills it picked.
    source_paths = [source_path] if isinstance(source_path, str) else list(source_path)
    missing = [p for p in source_paths if not os.path.exists(p)]
    if missing:
        print(f"ERROR: Source face not found: {missing[0]}")
        sys.exit(1)
    if not os.path.exists(target_path):
        print(f"ERROR: Target video not found: {target_path}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    os.makedirs(temp_path, exist_ok=True)

    processors = ["face_swapper"]
    if face_enhancer_model:
        processors.append("face_enhancer")
    if frame_enhancer_model:
        processors.append("frame_enhancer")

    cmd = [
        sys.executable, "facefusion.py", "headless-run",
        "--source-paths", *source_paths,
        "--target-path", target_path,
        "--output-path", output_path,
        "--temp-path", temp_path,
        "--processors", *processors,
        "--face-detector-model", face_detector_model,
        "--face-detector-size", face_detector_size,
        "--face-detector-score", str(face_detector_score),
        "--face-swapper-model", face_swapper_model,
        "--face-selector-mode", face_selector_mode,
        "--face-selector-order", face_selector_order,
        "--face-mask-types", *face_mask_types.split(),
        "--face-mask-blur", str(face_mask_blur),
        "--face-mask-padding", *face_mask_padding.split(),
        "--output-video-encoder", output_video_encoder,
        "--output-video-quality", str(output_video_quality),
        "--execution-providers", execution_providers,
        "--execution-thread-count", str(execution_thread_count),
        "--video-memory-strategy", "tolerant",
    ]

    if face_enhancer_model:
        cmd.extend(["--face-enhancer-model", face_enhancer_model])
        cmd.extend(["--face-enhancer-blend", str(face_enhancer_blend)])

    if frame_enhancer_model:
        cmd.extend(["--frame-enhancer-model", frame_enhancer_model])
        if frame_enhancer_blend is not None:
            cmd.extend(["--frame-enhancer-blend", str(frame_enhancer_blend)])

    if output_video_fps:
        cmd.extend(["--output-video-fps", str(output_video_fps)])

    print(f"Running FaceFusion...")
    print(f"  Source:     {', '.join(os.path.basename(p) for p in source_paths)}")
    print(f"  Target:     {target_path}")
    print(f"  Output:     {output_path}")
    print(f"  Swapper:    {face_swapper_model}")
    print(f"  Enhancer:   {face_enhancer_model} (blend={face_enhancer_blend})")
    print(f"  Encoder:    {output_video_encoder}")
    print(f"  Provider:   {execution_providers}")

    proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)) + "\\..\\facefusion")
    if proc.returncode != 0:
        print(f"ERROR: FaceFusion failed with code {proc.returncode}")
        sys.exit(1)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run FaceFusion face swap on a video")
    parser.add_argument("source", nargs="+",
                        help="Path(s) to source face image. Several are averaged into one "
                             "identity, which is what a source video resolves to.")
    parser.add_argument("target", help="Path to target video")
    parser.add_argument("-o", "--output", required=True, help="Output video path")
    parser.add_argument("-t", "--temp-dir", default="output/temp/ff_temp", help="Temp directory for FaceFusion")
    parser.add_argument("--swapper-model", default="inswapper_128_fp16", help="Face swapper model")
    parser.add_argument("--enhancer-model", default="codeformer", help="Face enhancer model")
    parser.add_argument("--enhancer-blend", type=int, default=80, help="Face enhancer blend percent")
    parser.add_argument("--frame-enhancer", default=None, help="Frame enhancer model (e.g. real_esrgan_x2)")
    parser.add_argument("--frame-enhancer-blend", type=int, default=80, help="Frame enhancer blend percent")
    parser.add_argument("--detector-model", default="retinaface", help="Face detector model")
    parser.add_argument("--detector-size", default="640x640", help="Face detector size")
    parser.add_argument("--detector-score", type=float, default=0.5, help="Face detector score threshold")
    parser.add_argument("--encoder", default="h264_nvenc", help="Video encoder")
    parser.add_argument("--quality", type=int, default=20, help="Output video quality")
    parser.add_argument("--fps", type=float, default=None, help="Output FPS override")
    parser.add_argument("--provider", default="cuda", help="Execution provider")
    parser.add_argument("--threads", type=int, default=8, help="Thread count")
    parser.add_argument("--mask-types", default="box occlusion", help="Face mask types")
    parser.add_argument("--mask-blur", type=float, default=0.7, help="Face mask blur")
    parser.add_argument("--mask-padding", default="0 5 0 5", help="Face mask padding")

    args = parser.parse_args()

    run_facefusion(
        source_path=args.source,
        target_path=args.target,
        output_path=args.output,
        temp_path=args.temp_dir,
        face_swapper_model=args.swapper_model,
        face_enhancer_model=args.enhancer_model,
        face_enhancer_blend=args.enhancer_blend,
        frame_enhancer_model=args.frame_enhancer,
        frame_enhancer_blend=args.frame_enhancer_blend,
        face_detector_model=args.detector_model,
        face_detector_size=args.detector_size,
        face_detector_score=args.detector_score,
        output_video_encoder=args.encoder,
        output_video_quality=args.quality,
        output_video_fps=args.fps,
        execution_providers=args.provider,
        execution_thread_count=args.threads,
        face_mask_types=args.mask_types,
        face_mask_blur=args.mask_blur,
        face_mask_padding=args.mask_padding,
    )

    print("\nFaceFusion complete.")


if __name__ == "__main__":
    main()
