import subprocess
import sys
import os
import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_facefusion(eye_label, input_path, output_path, source_path, facefusion_args=""):
    """Run FaceFusion on a single eye video."""
    facefusion_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "facefusion")

    cmd = (
        f'cd "{facefusion_dir}" && '
        f'..\\venv\\Scripts\\python.exe -m facefusion '
        f'--source-paths "{source_path}" '
        f'--target-path "{input_path}" '
        f'--output-path "{output_path}" '
        f'{facefusion_args}'
    )

    print(f"\n[{eye_label}] Starting FaceFusion processing...")
    print(f"[{eye_label}] Input:  {input_path}")
    print(f"[{eye_label}] Output: {output_path}")

    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\n[{eye_label}] ERROR: FaceFusion failed!")
        return False, elapsed

    print(f"\n[{eye_label}] Completed in {elapsed/60:.1f} minutes")
    return True, elapsed


def process_vr_parallel(left_input, right_input, left_output, right_output,
                        source_path, facefusion_args="", parallel=True):
    """Process both eyes, optionally in parallel."""

    os.makedirs(os.path.dirname(left_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(right_output) or ".", exist_ok=True)

    if parallel:
        print("\n=== Processing BOTH eyes in PARALLEL ===")
        print("This uses more GPU memory but processes 2x faster.\n")

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(run_facefusion, "LEFT", left_input, left_output,
                              source_path, facefusion_args): "LEFT",
                executor.submit(run_facefusion, "RIGHT", right_input, right_output,
                              source_path, facefusion_args): "RIGHT",
            }

            results = {}
            for future in as_completed(futures):
                eye = futures[future]
                success, elapsed = future.result()
                results[eye] = (success, elapsed)

        left_ok = results["LEFT"][0]
        right_ok = results["RIGHT"][0]
        total_time = max(results["LEFT"][1], results["RIGHT"][1])

    else:
        print("\n=== Processing eyes SEQUENTIALLY ===")

        left_ok, left_time = run_facefusion("LEFT", left_input, left_output,
                                            source_path, facefusion_args)
        if not left_ok:
            return False

        right_ok, right_time = run_facefusion("RIGHT", right_input, right_output,
                                              source_path, facefusion_args)
        total_time = left_time + right_time

    if left_ok and right_ok:
        print(f"\n=== VR Processing Complete ===")
        print(f"Total time: {total_time/60:.1f} minutes")
        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Process VR stereo video through FaceFusion (both eyes in parallel)"
    )
    parser.add_argument("--left-input", required=True, help="Left eye input video")
    parser.add_argument("--right-input", required=True, help="Right eye input video")
    parser.add_argument("--left-output", required=True, help="Left eye output video")
    parser.add_argument("--right-output", required=True, help="Right eye output video")
    parser.add_argument("--source", required=True, help="Source face image")
    parser.add_argument("--sequential", action="store_true",
                        help="Process sequentially instead of parallel (uses less GPU memory)")
    parser.add_argument("--facefusion-args", default="",
                        help="Additional FaceFusion arguments")

    args = parser.parse_args()

    success = process_vr_parallel(
        left_input=args.left_input,
        right_input=args.right_input,
        left_output=args.left_output,
        right_output=args.right_output,
        source_path=args.source,
        facefusion_args=args.facefusion_args,
        parallel=not args.sequential,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
