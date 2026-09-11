# Memory management + speed plan (Sep 10 2026)

Implementation plan for the next agent (Opus 4.8). Every step names the file,
the function, the exact behaviour, and the test that proves it. Do the steps
in order; each one is independently shippable and each one is measured before
the next starts. The "Verification" section at the end is what the reviewing
agent (Fable) will run after implementation.

---

## 0. What was measured, and what it means

All runs: one 4096x4096 eye (the `right.mp4` split in `output/temp`), 183
frames at 60 fps, the "Good" preset (hyperswap_1a_256, pixel boost 256,
gpen_bfr_512), CUDA, guard rails on. Wall time includes ~25 s of model load.
GPU utilisation is the steady-state mean from `nvidia-smi`; "CPU" is the
whole machine (28 logical cores). VRAM figures are system-wide; the idle
baseline was 1.3-3.4 GB depending on what else was open.

| variant | wall | job peak commit | VRAM used (system) | GPU util | CPU |
|---|---|---|---|---|---|
| today's defaults, threads 3 | 99.3 s | 14.5 GB | 11.2 GB (3.4 idle) | 70% | **82%** |
| + `CUDA_MODULE_LOADING=LAZY` | 96.3 s | 13.7 GB | 8.4 GB (1.3 idle) | 66% | 80% |
| + ORT session options (intra 2, no spinning) | 98.8 s | 13.8 GB | 8.4 GB | 72% | **20%** |
| + threads **6** | **89.2 s** | 14.9 GB | 9.3 GB | 76% | 21% |
| threads 3 + `kSameAsRequested` arena | 103.4 s | 13.5 GB | 8.3 GB | 61% | 22% |

Per-process CPU during the default run: the FaceFusion **python process alone
used 92% of the machine**; each ffmpeg reader used ~1%, the writer ~0%.
Software HEVC decode is *not* a cost here - do not chase NVDEC.

Conclusions that drive the plan:

1. **The "no available committed memory" refusal is real and correct.** The
   preflight needs `commit_need` (19.7 GB by the model) against
   `avail_commit - 3 GB floor` (16.5 GB at the time). The machine's commit
   limit is 63.8 GB and **44.4 GB of it was already charged** by other
   programs before the job started. See Part A.
2. **The model over-estimates commit by ~5 GB at 4K.** Measured 14.5 GB
   (13.7 with lazy loading) where `estimate_commit_mb()` says 19.7 GB. The
   refit in B7 alone would have let that job start with 2 GB to spare - but
   with the ambient 44 GB commit it would have run with ~3.6 GB of system
   commit free, i.e. one allocation from the watchdog's abort line. The real
   fix is Part A; B7 stops the estimate lying on top of it.
3. **62% of the machine's CPU was spin-waiting.** FaceFusion builds every
   `InferenceSession` with default `SessionOptions`, so each of its ~8 models
   gets its own intra-op pool of 14 threads that busy-spin between GPU ops.
   Setting `intra_op_num_threads=2` and `session.intra_op.allow_spinning=0`
   drops the machine from 82% to 20% CPU at identical speed. This is what makes
   any overlap (B9) or parallel lanes (B10) possible at all: with the old
   behaviour a second FaceFusion process would have been starved of CPU by the
   first one's spinning, and the 75% job CPU cap would have bound.
4. **A single process is Python-serial-bound, not GPU-bound.** GPU sits at
   70-76%. Six threads instead of three buys ~10% wall (more in steady state)
   and costs 0.9 GB of VRAM and 1.1 GB of commit. Thread count was clamped at
   4 only because the spinning pools made more threads pointless.
5. **Two full FaceFusion processes do NOT fit this card at 4K with an
   enhancer.** One pass holds ~7.0 GB of VRAM (8.3-8.4 GB used minus the
   1.3-1.4 GB idle baseline) regardless of arena strategy. Two lanes plus the
   baseline is ~15.4 GB of 16.3 GB; WDDM starts paging VRAM into host RAM
   near full, which is the failure mode that freezes the machine (see
   PROGRESS.md "cuDNN algorithm search"). So left+right in two processes is
   built as an *automatic* option gated by measured VRAM/commit/RAM headroom
   (B10). On this card with these settings the gate will say no; it will say
   yes for lighter presets (no enhancer: ~5 GB per pass) and smaller frames.
   The sure-fire speed-ups are B1-B3 and B9; the highest ceiling is B11.

---

## Part A - machine fixes (user does these; the app only reports them)

**A1. Reboot to apply the pagefile.** Registry
`HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management\PagingFiles`
says `c:\pagefile.sys 73728 73728` (72 GB), but `Win32_PageFileUsage` reports
the live file at 32,768 MB and the machine has been up since Sep 5 17:29.
After a reboot the commit limit goes from 63.8 GB to ~104 GB and this class
of refusal disappears for a single lane. C: has 76 GB free with the 32 GB
pagefile already on it, so the grown file leaves ~36 GB free on C:. If that is
too tight, 64 GB (65536) is enough headroom for two lanes at 4K.

**A2. Close the commit hogs before an overnight run** (measured at 01:20):

| process | commit | note |
|---|---|---|
| 3x `llama-server.exe` (Ollama runners, same model) | 8.4 GB | `ollama stop <model>` or set `OLLAMA_KEEP_ALIVE=0`; they also hold VRAM |
| `pythonw.exe` dexter `desktop_app.py` | 3.2 GB | another project's app |
| `NVIDIA Overlay.exe` x2 | 2.5 GB | disable the in-game overlay in the NVIDIA app |
| `bdservicehost` (Bitdefender) | 1.0 GB | leave it |
| `sqlservr`, `mysqld` | 1.2 GB | stop the services if unused overnight |
| 3x `claude.exe`, chrome, explorer | ~3 GB | |

The app must never kill these itself. B5 makes it *name* them in the refusal.

---

## Part B - code changes, in order

### B1. ONNX Runtime session options (the CPU fix) - `scripts/patch_facefusion.py`

Add a third idempotent patch, `PATCHES` entry
`"bounded onnxruntime thread pools"`, targeting
`facefusion/facefusion/inference_manager.py::create_inference_session`.

Replace

```
\t\tinference_session = InferenceSession(model_path, providers = inference_providers)
```

with (tabs, like the file):

```
\t\t# VrDeep: bounded, non-spinning thread pools. ...(comment with the numbers above)
\t\tsession_options = onnxruntime.SessionOptions()
\t\tsession_options.intra_op_num_threads = VRDEEP_ORT_INTRA_THREADS
\t\tsession_options.inter_op_num_threads = 1
\t\tsession_options.add_session_config_entry('session.intra_op.allow_spinning', '0')
\t\tsession_options.add_session_config_entry('session.inter_op.allow_spinning', '0')
\t\tinference_session = InferenceSession(model_path, sess_options = session_options, providers = inference_providers)
```

and add at module level, next to the imports (`import onnxruntime` +
`import os`):

```
VRDEEP_ORT_INTRA_THREADS = int(os.environ.get('VRDEEP_ORT_INTRA_THREADS', '2'))
```

- Keep the existing `from onnxruntime import InferenceSession` import; add
  `import onnxruntime` and `import os` in the same block. The OLD/NEW pair for
  the import must be a separate patch tuple so `status()` reports each half.
- `engine.run_facefusion()` sets `VRDEEP_ORT_INTRA_THREADS` in the child env
  from `guard_config["ort_intra_op_threads"]` (new key, default 2).
- Add `"ort_intra_op_threads": 2` to `default_guard_config()` with a comment
  carrying the 82% -> 20% measurement.
- `prepare_run()` already warns when patches are missing; nothing to add.

Test: `python scripts/patch_facefusion.py --check` shows three `applied`
lines; a 183-frame pass on `output/temp/right.mp4` (3 s trim) shows machine
CPU <= 30% (was 82%) and >= 1.8 frame/s steady. Regression check: run
`--check` twice; the second run changes nothing.

### B2. CUDA lazy module loading - `scripts/engine.py`

In `run_facefusion()` build `env = dict(os.environ)`, set
`env.setdefault("CUDA_MODULE_LOADING", "LAZY")` and the B1 variable, and pass
`env=env` to `guard.spawn`. Also for `rife_worker.py` and `esrgan_worker.py`
spawns (`interpolate_video`, `upscale_video_realesrgan`) and the CUDA probe in
`check_gpu()`. Measured: -0.75 GB commit per pass, no speed change.

Test: job peak commit in the `[Guard] Peak memory used by pipeline` line drops
by 0.5-1.0 GB against a run without the variable.

### B3. Execution threads 3 -> 6 - `scripts/pipeline_config.json`, `scripts/app.py` presets, `scripts/guardrails.py`

- `default_guard_config()["max_execution_threads"]`: `max(2, min(8, cores // 3))`
  (= 8 on 28 cores). Update the comment: the old cap of 4 was fitted while the
  ORT pools were spinning.
- `pipeline_config.json` `"threads": 6`; the "Good"/"Better"/"Best" presets in
  `app.py` likewise. Keep the Settings slider range 1-8.
- `estimate_vram_mb()`: replace the `1 + 0.35 * (threads - 1)` multiplier with
  `+ 300 MB per thread above 1` (measured: 3 -> 6 threads added 0.9 GB total).
  Same shape change in `estimate_commit_mb()` (see B7) and
  `estimate_resident_mb()` (its 160 MB/thread figure is already measured and
  stays).
- `clamp_config()`: unchanged in logic, but re-read its thread-reduction
  messages so they do not still claim "4 threads -> 1 saves 0.48 GB and costs
  2x" as if 4 were the ceiling.
- `guardrails.json` on disk carries `max_execution_threads: 4`; bump it in the
  file too (it is user-editable and loaded over the defaults).

Test: the same 183-frame pass finishes >= 8% faster than B1 alone; VRAM used
stays <= 10 GB on an idle desktop; preflight VRAM estimate for threads 6 stays
under the 92% block line.

### B4. Guard CPU cap and priority for unattended runs - `scripts/guardrails.py`, `scripts/app.py`

With B1 the pipeline no longer needs 75% of the machine, so the cap is no
longer what limits it - but a *second* lane (B10) or an overlapped split/merge
(B9) will push encode/decode CPU up again. Add `"cpu_percent_cap_unattended": 90`
to the guard config; `ResourceGuard.start(unattended=False)` picks the cap by
that flag. `app._launch_job()` passes `unattended=not interactive`. Log which
cap applied.

Test: log line `[Guard] Active - memory cap ..., CPU cap 90%` for a queued
job, 75% for an interactive Run.

### B5. Preflight says *who* holds the commit, and whether a reboot is pending - `scripts/guardrails.py`

1. `top_commit_processes(limit=6, exclude_pids=())`: same walk as
   `top_memory_processes()` but sorting on `process_memory(pid)[1]`
   (`PrivateUsage`, which *is* the process's commit charge). Group duplicates
   by exe name (three Ollama runners read as one line with a count).
2. In `preflight()`, when the available-commit check *blocks*, append to the
   message: `The commit is held by: llama-server.exe x3 8.4 GB, pythonw.exe
   3.2 GB, NVIDIA Overlay.exe x2 2.5 GB ...` and the sentence
   `Closing those frees N GB - enough for this run.` when it is.
3. `pagefiles()`: extend the PowerShell query to also read
   `Win32_PageFileSetting` (`InitialSize`/`MaximumSize`) and return
   `configured_mb` per pagefile. New `pagefile_reboot_pending()` returns a
   message when `configured_mb` differs from `allocated_mb` by > 512 MB:
   `The pagefile is set to 72 GB but the live file is 32 GB - Windows applies
   the new size at the next reboot. Rebooting raises the commit limit from
   63.8 GB to ~104 GB.` Print it at app startup (next to
   `slow_pagefile_warning()`), in preflight warnings, and in the blocking
   commit message.
4. `explain_memory_pressure()`: add the top-commit line as well as the
   working-set line, so a watchdog abort on commit names the holder too.

Test: with a fake `avail_commit` (monkeypatch `memory_status`) the blocking
message contains the process list; `pagefile_reboot_pending()` returns a
string on this machine before the reboot and `None` after.

### B6. A queued job that is refused for a *transient* reason waits instead of failing - `scripts/app.py`, `scripts/engine.py`

Today `_launch_job()` marks the job `failed` when `prepare_run()` raises, and
the overnight queue moves on (or stops). The job on Sep 10 01:16 was refused
because Ollama was still holding 8 GB - a condition that clears by itself.

1. `guardrails.preflight()` tags each blocking problem: return
   `blocking` as a list of `(kind, text)` with `kind in
   {"commit", "ram", "disk", "vram", "cap"}`. `prepare_run()` raises
   `PipelineError` with a new attribute `.transient = all(kind in
   ("commit", "ram") for kind, _ in blocking)`. Keep the message text
   identical.
2. In `app._launch_job()` (unattended only): if the error is transient, mark
   the job `"deferred"` with note `Waiting for memory: <first line>`; record
   `job["retry_after"] = now + guard_config["queue_retry_minutes"]` (new key,
   default 5) and `job["deferred_since"]` (first deferral). `_queue_tick()`
   treats `deferred` like `waiting` once `retry_after` has passed. After
   `guard_config["queue_retry_hours"]` (default 6) of deferral the job is
   marked `failed` with the last reason.
3. Queue tab: `deferred` rows show in amber with the retry time; the status
   line reads `Waiting for memory - retrying at HH:MM`. `stop_on_failure` does
   not trigger on a deferral.
4. Between retries `guardrails.wait_for_commit_release()` is *not* needed;
   the retry is a fresh preflight.
5. `queue.json` schema gains `retry_after` / `deferred_since`; `job_not_before`
   is unchanged. Loading an old queue file with no such keys must work.

Test: set `queue_retry_minutes` to 0.2, start a queue while a dummy process
holds 15 GB of commit (a Python child doing `bytearray(15 * 2**30)` and
sleeping), see the job defer twice in the log, kill the dummy, see it start.

### B7. Refit `estimate_commit_mb()` to the measurements - `scripts/guardrails.py`

Anchor points, all "Good" preset, 3 threads, lazy loading on (B2):

| frame | measured peak commit |
|---|---|
| 3840x2160 Normal (8.3 MP) | 11.1 GB (PROGRESS.md, Aug 27) |
| 4096x4096 VR eye (16.8 MP) | 13.7 GB (this document) |
| 4096x4096, 6 threads | 14.9 GB |

New model (MB): fixed part unchanged (`5000` base + pixel-boost + enhancer
terms, = 7400 for Good); per-thread `threads * mpix * 22`; ffmpeg children
`3 * (mpix * 100 + 250)`; frame-enhancer term unchanged. Check: 8.3 MP ->
7400 + 548 + 3 * 1080 = 11.2 GB; 16.8 MP -> 7400 + 1109 + 3 * 1930 = 14.3 GB;
6 threads -> +1.1 GB. Add a 5% safety factor at the end and update the
docstring table. The "Best" anchor (19.5 GB at 8.3 MP, PROGRESS Aug 27) must
still come out within 1 GB - the fixed terms carry that difference, leave them.

Test: `python -c` printing the three anchors within +/-0.7 GB of measured; the
Sep 10 job (16.8 MP, Good) now estimates ~14.3 GB, and with 19.5 GB of
available commit the preflight *warns* rather than blocks.

### B8. Lanes: a guard that can freeze part of a run - `scripts/guardrails.py`

Foundation for B9 and B10. Today `ResourceGuard` freezes/thaws the whole job.

1. `ResourceGuard.spawn(cmd, lane=None, **kw)`: record `lane` per proc in
   `self._lanes: dict[str, set[int]]` (root pids). Existing callers pass no
   lane; `engine.run_facefusion()` gains a `lane` parameter and passes it.
2. `lane_pids(lane)`: root pids plus their descendants, found by walking the
   Toolhelp snapshot (`_PROCESSENTRY32W.th32ParentProcessID`) **restricted to
   pids returned by `_job_process_ids(self._job)`**. That restriction is what
   makes the parent-pid walk safe (recycled pids outside the job are never
   touched) - keep the comment on `_job_process_ids` honest by saying so.
3. `pause_lane(lane, by="guard")` / `resume_lane(lane)`: same parents-first /
   children-first ordering as `_set_suspended`, on `lane_pids(lane)`. Track
   `self._lane_paused: dict[lane -> (since, by)]`. A whole-job `pause()` still
   freezes everything and remembers which lanes were already held so
   `resume()` does not thaw a lane the watchdog is holding.
4. `lane_memory_now(lane)`: sum of working sets of `lane_pids(lane)`, for the
   log.
5. Watchdog policy, in `_watchdog_loop()`, evaluated before the existing
   whole-job freeze:
   - Trip points (new config keys, defaults): `lane_pause_free_ram_gb: 6.5`
     (above `auto_pause_free_ram_gb`), `lane_pause_free_commit_gb: 6.0`,
     `lane_pause_free_vram_mb: 1200`, `lane_resume_margin_gb: 2.0`,
     `lane_pause_max_sec: 1800`.
   - When >= 2 lanes are live and any trip point is crossed: freeze the lane
     registered **last** (`self._lane_order`), log
     `[Guard] Free VRAM down to 0.9 GB - holding the right-eye lane; the left
     eye keeps running.` and call `empty_working_set` on its pids (existing
     `free_memory_now` logic, per lane).
   - Resume that lane when every metric is back above its trip point plus
     `lane_resume_margin_gb` (VRAM: +800 MB) for 3 consecutive samples.
   - If the held lane has been frozen for `lane_pause_max_sec`, **kill only
     that lane** (`TerminateProcess` on `lane_pids`) and set
     `self.lane_failed[lane] = reason`; the caller (B10) re-runs it
     sequentially. Never abort the whole run for a lane problem.
   - If the trip is still crossed with one lane held, fall through to the
     existing whole-job freeze / abort path unchanged.
   - VRAM sampling: `gpu_memory_mb()` shells out to `nvidia-smi` (~50 ms);
     call it only when >= 2 lanes are live, and no more than once per
     watchdog interval.
6. `paused_seconds()` must count time when *all* live lanes are frozen; a
   single held lane does not stop the run clock (the other lane is working).

Test (unit, no GPU): spawn two `python -c "import time; time.sleep(60)"`
children in lanes `a`/`b`, call `pause_lane("b")`, verify via
`_set_process_suspended` return / `NtQueryInformationProcess` that only `b`'s
pid is suspended, `resume_lane("b")` thaws it, `pause()` then `resume()`
leaves a previously held lane held. Test (integration): run B10 with
`lane_pause_free_vram_mb` set absurdly high (e.g. 20000) and watch the right
lane freeze on the first sample, then lower the key at runtime (the watchdog
re-reads `self.config`) and watch it thaw.

### B9. Overlap split and merge with the swap across chunks - `scripts/engine.py`

Per 60 s chunk at 8192x4096 90 fps: split ~3.6 min, both swaps ~72 min,
merge ~8 min. Split and merge are NVENC/NVDEC + a little CPU; the swap leaves
the encoder at 12-16% and, after B1, the CPU at 20%. Running them under the
swap hides ~12 min of every 84 (~14%), and costs one extra chunk of split
files on disk (~1.2 GB at 4K60).

1. **Per-chunk temp dirs.** `run_vr_pipeline()` takes `temp_dir` as a
   parameter (default the current `output/temp`); `run_chunked_pipeline()`
   passes `output/temp/chunk_NNN`. `RunState` lives in that dir. `cleanup_temp`
   of a chunk dir is the chunk's own business; the top-level `cleanup_temp`
   removes `chunk_*` dirs on success.
2. **Split prefetch.** In `run_chunked_pipeline()`, before starting chunk `i`,
   if a `split_future` for chunk `i` exists, wait on it; then submit
   `split_video(...)` for chunk `i+1` to a single-worker
   `ThreadPoolExecutor` with `guard.spawn(..., lane="split")`. `run_vr_pipeline`
   gains `presplit=(left_path, right_path)`; when given, it records the
   `split_left/right` stages as complete (same `state.complete` calls as
   today) and skips its own split.
3. **Merge deferral.** `run_vr_pipeline` gains `defer_merge=True` for chunked
   runs: it returns after the swaps (and interpolation) with the two eye
   paths; `run_chunked_pipeline` submits `merge_videos + apply_stereo_metadata
   + chunk_is_complete` for chunk `i-1` to a single-worker executor with
   lane `"merge"` while chunk `i`'s swap runs. The join at the end waits for
   the last merge future. A merge failure raises at the next chunk boundary
   with the same message as today.
4. **Clock.** `RunClock` for a chunk is built from an estimate whose `split`
   and `merge` stages are marked `overlapped: True`; `RunClock.__init__`
   gives such stages 0% of the bar and excludes them from `budget`; the
   overall chunk pace in `run_chunked_pipeline` (`overall_left()`) is already
   measured from wall-clock and needs no change. Timing calibration for
   `split`/`merge` kinds is still recorded from their own wall time.
5. **Disk preflight.** `estimate_disk_need()` for a chunked run: two chunks of
   split intermediates + one chunk of swapped eyes + one merged chunk.
6. **Resume.** Chunk manifests are unchanged; a chunk dir with a finished
   split for `i+1` is reused via the existing stage keys.
7. `guard.check()` between every future handoff; a `GuardAbort` cancels the
   pending futures and waits for the running ones.

Test: a 3-chunk run on a 150 s trim of the 4K test file. Log shows
`[Chunks] split of chunk 2 finished under chunk 1's swap` and
`[Chunks] merge of chunk 1 finished under chunk 2's swap`; total wall is
>= 10% under the same run with `overlap_stages: false` (new guard key,
default true); output frame count equals the sequential run's; the joined
file plays.

### B10. Both eyes at once, when the machine can afford it - `scripts/engine.py`, `scripts/guardrails.py`, `scripts/app.py`

Two FaceFusion processes, one per eye, each its own lane, with the watchdog
from B8 holding the second one when memory runs short. **Gated**: it only
runs when the measured headroom says two lanes fit; otherwise the log says
why and the run is sequential, exactly as today.

1. Config: `guardrails.json` `"parallel_eyes": "auto"` (`"auto"|"on"|"off"`);
   Settings tab gets the same three-way control under "Chunked Processing".
   `"on"` still refuses when VRAM cannot fit (that is a freeze, not a slow
   run); it only skips the commit/RAM margin.
2. Gate, in `guardrails.parallel_eyes_verdict(pipeline_config, guard_config,
   pass_pixels)` -> `(ok, reason)`:
   - VRAM: `free_vram_mb >= 2 * estimate_vram_mb(cfg) + 1500`
     (`estimate_vram_mb` must first be refit to the measured ~7000 MB for
     Good/3 threads and ~7900 MB for 6 threads - today it says 5397 MB and is
     wrong by a third).
   - Commit: `avail_commit - floor >= 2 * estimate_commit_mb(...)`.
   - RAM: `avail_ram - floor >= 2 * estimate_resident_mb(...)`.
   - Threads: each lane runs `max(2, threads // 2)` execution threads (the
     two processes share the GPU; 3+3 is measured above, 6+6 is not).
   - The verdict and its numbers are logged in preflight:
     `[Guard] Both eyes at once: no - VRAM 14.9 GB needed for two lanes,
     16.3 GB on the card with 1.4 GB in use (needs 1.5 GB spare).`
3. `run_vr_pipeline()`: when the verdict is ok and both swaps are needed, run
   `run_facefusion(..., lane="left")` and `run_facefusion(..., lane="right")`
   on two threads; each keeps its own `progress_cb` slice: `RunClock` gets a
   combined stage `swap_both` whose seconds are `max(swap_left, swap_right)
   * 1.15` (the 15% is the GPU-sharing penalty until measured), and its
   progress is the mean of the two lanes' fractions. Stage records
   (`state.complete("swap_left"...)`) stay per eye so a resume after a
   failure reuses whichever eye finished.
4. Failure of one lane: if `guard.lane_failed["right"]` is set (B8 timeout) or
   the right process exits non-zero **while the left is still running**, let
   the left finish, then re-run the right eye sequentially with the full
   thread count, and log that the machine could not hold two lanes. Two
   failures = the normal `PipelineError` path.
5. `estimate_runtime()` gains `parallel_eyes` awareness so the up-front
   estimate reads `Face swap, both eyes together: 41m` instead of two 36 m
   lines.
6. Nothing here changes Normal (2D) mode.

Expected on this machine: with the enhancer on at 4K the gate refuses (7 GB
x 2 + 1.5 > 16.3 - 1.4). With `enhancer_model: null` (3.5 frame/s, ~5 GB
VRAM per lane) or a 2K-per-eye source it runs and should give ~1.25-1.35x on
the swap stage (GPU has 25-30% idle in one process). Measure before claiming
more.

Test: (a) unit test of the verdict against three fake `gpu_memory_mb` /
`memory_status` readings; (b) a 60 s chunk with the enhancer off on an idle
desktop: log shows both lanes, VRAM stays <= 15 GB, wall <= 0.8x sequential;
(c) same run with `lane_pause_free_vram_mb: 20000` shows the right lane held
and then killed after `lane_pause_max_sec` (set to 60 for the test) and
re-run sequentially, and the output is still complete.

### B11. (Later, highest ceiling) both eyes in ONE FaceFusion process

The models are the VRAM; a second *process* pays for them twice. A patch to
`facefusion/workflows/to_video.py::process_memory_frames` that takes two
targets (`VRDEEP_SECOND_TARGET`, `VRDEEP_SECOND_OUTPUT` env), interleaves
`process_memory_frame` calls for both in the same bounded window, and writes
to two `video_manager` writers would fill the 25-30% idle GPU for ~+1.5 GB
(one more reader pair, one more writer, double in-flight frames) instead of
+7 GB. It keeps identity resolution identical for both eyes by construction.
Per-eye pausing then becomes a control file the loop polls every N frames
rather than `NtSuspendProcess`. Do not start this until B1-B10 are measured;
it is the only step that touches FaceFusion's control flow rather than its
configuration.

---

## Part C - order of work and what to measure

| step | expected effect | measure with |
|---|---|---|
| A1 reboot | commit limit 63.8 -> ~104 GB | `guardrails.commit_limit_bytes()` |
| A2 close hogs | +12-15 GB available commit | `memory_status()[2]` |
| B1 ORT options | CPU 82% -> ~20%, same fps | scratch `measure_pass.py` pattern (see Verification) |
| B2 lazy CUDA | -0.75 GB commit/pass | `[Guard] Peak memory used` |
| B3 threads 6 | -10% wall per eye | same pass, wall time |
| B7 refit | preflight within 1 GB of measured | `estimate_commit_mb` vs peak |
| B5/B6 | overnight queue never dies on a transient refusal | queue log |
| B8 | lane freeze/thaw works | unit test |
| B9 overlap | -10-14% wall per chunk | 3-chunk run |
| B10 lanes | +25-35% on swap when the gate passes | 60 s chunk, enhancer off |

Do not change `ffmpeg_reader_threads` (readers are ~1% CPU) and do not add
NVDEC to the readers; both were checked and are not where the time goes.

---

## Verification (Fable runs this after implementation)

1. `python scripts/patch_facefusion.py --check` -> three `applied` lines, exit 0.
2. Reproduce the measurement: cut 3 s of `output/temp/right.mp4`
   (`ffmpeg -ss 0 -t 3 -i right.mp4 -c copy -an clip3.mp4`), run
   `engine.run_facefusion` on it under a `ResourceGuard` while sampling
   `nvidia-smi --query-gpu=utilization.gpu,memory.used` and
   `GetSystemTimes` once a second (the script used for this plan is
   `measure_pass.py` in the session scratchpad; recreate it from the numbers
   above if it is gone). Expect: CPU mean <= 30%, wall <= 92 s at 6 threads,
   job peak commit <= 15.5 GB, VRAM <= 10 GB on an idle desktop.
3. `python -c` the three B7 anchors; each within +/-0.7 GB.
4. Preflight the Sep 10 job (queue.json job_1788895690692's config, 4096x4096
   per eye) with `prepare_run(..., force=False)` on a machine state with
   >= 18 GB of available commit: no `PipelineError`; with < 14 GB: a
   `PipelineError` whose text lists the top commit holders and whose
   `.transient` is `True`.
5. Queue deferral test from B6 with a dummy 15 GB commit holder.
6. B8 unit test (two sleeping children, per-lane suspend).
7. B9: 3-chunk 150 s run, overlap on vs off, frame counts equal, wall >= 10%
   lower with overlap on.
8. B10: verdict unit test; one enhancer-off 60 s chunk with both lanes,
   VRAM <= 15 GB, wall <= 0.8x sequential; forced lane-hold test.
9. `git diff --stat` touches only `scripts/*.py`, `scripts/*.json`,
   `scripts/patch_facefusion.py`; nothing under `facefusion/` is committed
   (it is ignored) - the ORT change lives only in the patch script.
