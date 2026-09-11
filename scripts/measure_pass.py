"""
Measure one FaceFusion pass: wall time, job peak commit, GPU/NVENC utilisation, VRAM, free RAM/commit, machine CPU.

Used for MEMORY_AND_SPEED_PLAN.md (Sep 10 2026). Put a short clip named clip3.mp4 beside this file
(e.g. ffmpeg -ss 0 -t 3 -i output/temp/right.mp4 -c copy -an clip3.mp4), then run it with facefusion/venv python
from the facefusion/ directory. Env: VRDEEP_THREADS overrides execution threads.
"""
import sys, os, time, threading, subprocess, json, glob, ctypes
from ctypes import wintypes
sys.path.insert(0, r"B:\youtubeProjects\Buzzcaf_Media\VrDeep\scripts")
import engine, guardrails
from guardrails import ResourceGuard

S = os.path.dirname(os.path.abspath(__file__))
clip = os.path.join(S, "clip3.mp4")
out = os.path.join(S, "clip3_swapped.mp4")
tmp = os.path.join(S, "ff_tmp")
os.makedirs(tmp, exist_ok=True)
cfg = engine.load_config(); cfg['threads'] = int(os.environ.get('VRDEEP_THREADS', cfg['threads']))
faces = sorted(glob.glob(r"B:\youtubeProjects\Buzzcaf_Media\VrDeep\output\source_faces\data_src1_afa77cad94\face_0*.png"))
source = ";".join(faces)

k32 = ctypes.windll.kernel32
def sys_times():
    idle, kern, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    k32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
    f = lambda t: (t.dwHighDateTime << 32) | t.dwLowDateTime
    return f(idle), f(kern) + f(user)

samples = []
stop = threading.Event()
def sampler():
    last = sys_times()
    while not stop.is_set():
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,utilization.encoder,utilization.decoder,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, creationflags=0x08000000)
        t, a, p = guardrails.memory_status()
        now = sys_times()
        d_idle, d_tot = now[0] - last[0], now[1] - last[1]
        cpu = 100.0 * (1 - d_idle / d_tot) if d_tot else 0
        last = now
        try:
            g = [float(x.strip()) for x in r.stdout.strip().split(",")]
        except Exception:
            g = [0, 0, 0, 0, 0]
        samples.append((time.time(), *g, a / 2**30, p / 2**30, cpu))
        stop.wait(1.0)

th = threading.Thread(target=sampler, daemon=True); th.start()
log = []
def lg(m):
    log.append(f"{time.strftime('%H:%M:%S')} {m}"); print(m, flush=True)
guard = ResourceGuard(tmp, guardrails.load_guard_config(), log_cb=lg).start()
t0 = time.time(); ok = True; err = ""
try:
    engine.run_facefusion(source, clip, out, tmp, cfg, log_cb=lg, progress_cb=lambda p, t: None, guard=guard, stage_name="Measure")
except Exception as e:
    ok = False; err = str(e); lg("FAILED: " + err)
wall = time.time() - t0
peak = guard.peak_memory_bytes(); lowc = guard.min_available_commit_bytes()
guard.close()
stop.set(); th.join(timeout=3)

# summarise only the steady-state window (skip first 25% = model load) if enough samples
n = len(samples)
body = samples[int(n*0.3):] if n > 20 else samples
def col(i): return [s[i] for s in body]
def mean(v): return sum(v)/len(v) if v else 0
summary = {
    "ok": ok, "error": err[-600:], "wall_s": round(wall,1), "frames": 183,
    "job_peak_commit_gb": round(peak/2**30,2), "min_avail_commit_gb": round(lowc/2**30,2),
    "samples": n,
    "gpu_util_mean": round(mean(col(1)),1), "gpu_util_max": max(col(1)) if body else 0,
    "gpu_memctl_mean": round(mean(col(2)),1),
    "nvenc_util_mean": round(mean(col(3)),1), "nvdec_util_mean": round(mean(col(4)),1),
    "vram_used_mb_max": max(col(5)) if body else 0, "vram_used_mb_mean": round(mean(col(5))),
    "free_ram_gb_min": round(min(col(6)),2) if body else 0,
    "avail_commit_gb_min": round(min(col(7)),2) if body else 0,
    "cpu_total_mean": round(mean(col(8)),1), "cpu_total_max": round(max(col(8)),1) if body else 0,
}
json.dump({"summary": summary, "samples": samples, "log": log}, open(os.path.join(S, "measure_result.json"), "w"), indent=1)
print("SUMMARY", json.dumps(summary, indent=1))
