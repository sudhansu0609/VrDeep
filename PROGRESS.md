# VR Deepfake Pipeline - Implementation Progress

**Last Updated**: Thu Sep 10 2026

---

## ⚡ 62% OF THE MACHINE WAS SPIN-WAITING, AND THE OTHER EYE WAS IDLE (Sep 10 2026)

A 4K VR job was refused at 01:16 with "no available committed memory" on a
machine that could have run it, and the overnight queue then failed the job and
sat idle until morning. Working out why turned into a measurement of where a
run's time and memory actually go, and then into ten changes
(`MEMORY_AND_SPEED_PLAN.md`, steps B1-B10). Everything below is measured on one
4096x4096 eye - 183 frames of `output/temp/right.mp4` at 60 fps, the "Good"
preset - unless it says otherwise.

| variant | wall | job peak commit | VRAM (system) | GPU | machine CPU |
|---|---|---|---|---|---|
| the defaults as they were, 3 threads | 99.3 s | 14.5 GB | 11.2 GB | 70% | **82%** |
| + `CUDA_MODULE_LOADING=LAZY` | 96.3 s | 13.7 GB | 8.4 GB | 66% | 80% |
| + bounded onnxruntime pools | 98.8 s | 13.8 GB | 8.4 GB | 72% | **20%** |
| + 6 execution threads | **89.2 s** | 14.9 GB | 9.3 GB | 76% | 21% |

### The CPU was not doing any work

FaceFusion builds every `InferenceSession` with default `SessionOptions`, so
onnxruntime gave each of its ~8 models an intra-op pool of one thread per core
and let those threads **busy-spin** while they waited for the GPU. The
FaceFusion python process alone held 92% of a 28-core machine; the machine sat
at 82% while the card was 70% busy. Two intra-op threads per session with
`session.intra_op.allow_spinning` off takes the machine to 20% CPU at the same
frame rate.

Nothing about the swap got faster. What changed is that there is now a machine
left over - which is what every other change here spends.

The patch lives in `scripts/patch_facefusion.py` (a third idempotent rewrite,
plus a fourth for the import it needs) because `facefusion/` is an untracked
clone and a `git pull` in there would silently undo a direct edit.
`engine._cuda_child_env()` passes the thread count in as
`VRDEEP_ORT_INTRA_THREADS`, and sets `CUDA_MODULE_LOADING=LAZY` while it is
there: without it the CUDA runtime materialises every kernel image of every
loaded module at context creation, which cost 0.8 GB of commit and 2.8 GB of
VRAM for kernels the run never calls.

With the pools bounded, the old cap of 4 execution threads made no sense - it
had been fitted while more threads only bought more spinning. It is 8 now
(`max(2, cores // 3)`), the presets ask for 6, and a thread is charged at a
measured 300 MB of VRAM rather than the 35% of the whole model stack the
estimate used to invent.

### The refusal was real, and the estimate was lying on top of it

`estimate_commit_mb()` said the job needed 19.7 GB. It needed 13.7. The
difference is that the old fit charged 100 MB per thread-megapixel and 130 MB
per ffmpeg-megapixel; the two measured frame sizes together say 22 MB and
80 MB. Refitted against four anchors it is now within 0.2 GB of every one of
them:

```
3840x2160  8.3 MP Good t3   model 11.0 GB   measured 11.1 GB
4096x4096 16.8 MP Good t3   model 13.6 GB   measured 13.7 GB
4096x4096 16.8 MP Good t6   model 14.8 GB   measured 14.9 GB
3840x2160  8.3 MP Best t3   model 19.5 GB   measured 19.5 GB
```

That job would still have been tight, because 44.4 GB of the machine's 63.8 GB
commit limit was already charged to other programs - three Ollama runners at
8.4 GB, another project's app at 3.2 GB, two NVIDIA overlays at 2.5 GB - and
because the pagefile had been enlarged to 72 GB but not yet applied. So the
preflight now **names them**: `The commit is held by: llama-server.exe x3
8.4 GB, pythonw.exe 3.2 GB ...` with `Closing those frees N GB - enough for
this run` when it is, and a separate line for a pagefile that is waiting on a
reboot. Sorting on `PrivateUsage` rather than working set matters here: a CUDA
process or a browser can hold three times its working set in commit, and the
thing worth closing never appeared in the old list.

### A queued job that is refused for memory now waits

Committed memory and free RAM are facts about what the machine is holding this
minute; disk, VRAM and the job cap are facts about the run. `preflight()` tags
every blocking problem with which kind it is, `PipelineError.transient` is true
only when all of them are the first kind, and an unattended job that hits one
is marked **deferred** rather than failed: amber in the queue, retried every
`queue_retry_minutes` (5) with a fresh preflight, and only failed once it has
waited `queue_retry_hours` (6). `stop_on_failure` does not fire on a deferral.
The night that was lost to an 8 GB Ollama runner would now have cost minutes.

### Lanes: a guard that can freeze part of a run

`ResourceGuard` could freeze the whole job or nothing. That is the right answer
for a run with one thing in flight and the wrong one for everything below, so
it now has **lanes**: `guard.spawn(cmd, lane="right")` records which part of
the run a process belongs to, and `pause_lane` / `resume_lane` / `kill_lane`
act on that part alone.

Finding a lane's processes needs a parent-PID walk, which is unsafe on its own
- Windows recycles PIDs and leaves a dead parent's ID on its children - so the
walk is intersected with `_job_process_ids()`, the kernel's own list of who is
in this run's job object. A stale PID can then only ever pull in another
process of this same run. Nothing outside the job is a candidate, which is the
entire safety argument for suspending, trimming and killing lanes.

The watchdog gets one policy above the whole-job one. When two or more lanes
are live and free RAM, free commit or **free VRAM** crosses a trip point
(`lane_pause_free_*` in `guardrails.json`, all set above the whole-job trip
points), it freezes the lane registered **last** - the older one is closer to
finishing, and finishing is what actually gives memory back - trims its working
set, and says so:

```
[Guard] free VRAM down to 0.9 GB - holding the right lane; the other lane keeps
        running. Froze 4 process(es), trimmed 4 and handed back 1.8 GB.
```

It thaws that lane once every metric has been clear of its trip point plus the
margin for three consecutive samples. If the memory never comes back it kills
**only that lane** and records `lane_failed`, and the caller re-runs that eye
on its own. A lane problem never aborts the run.

Two smaller things fall out of that. A user Pause no longer thaws a lane the
watchdog is holding, and a held lane does not stop the run clock: while one eye
is still swapping the run is still making progress, and charging the estimate
for that time would report an hour's swap as two. Only a moment when *every*
live lane is frozen counts as paused.

### The split and the merge now run underneath the swap

Per 60 s chunk at 8192x4096 90 fps: split ~3.6 min, both swaps ~72 min, merge
~8 min. The split and the merge are NVENC and NVDEC with almost no CPU; the
swap is CUDA inference that leaves the encoder at 12-16% and, now, the machine
at 20%. So the next chunk's split and the previous chunk's merge run
*underneath* this chunk's swap - about 12 minutes of every 84.

```
[Chunks] split of chunk 2 finished under chunk 1's swap
[Chunks] merge of chunk 1 finished under chunk 2's swap
```

Each chunk gets its own scratch directory (`output/temp/chunk_003`), because
before this, chunk 4's split and chunk 3's merge would both have been reading
and writing `output/temp/left.mp4`. `run_vr_pipeline` takes `temp_dir`, a
`presplit` pair for a split that has already been done, and `defer_merge`,
which stops after the swaps and hands the two eyes back to the caller.

The failure this could have introduced - a merge reading an eye that the next
chunk is still writing - is not a crash. `hstack` stops at the shorter input,
so it would have produced a file that plays at the right length and is missing
frames. Every merge therefore now checks that the merged file holds exactly as
many frames as each eye did, from the container header, and refuses to go on if
it does not. Duration alone would not have caught it.

Turn it off with `overlap_stages: false` in `guardrails.json`. The overlapped
stages keep their predicted seconds - the calibration still wants them - but
own no part of the progress bar and no place in the budget, because time hidden
under another stage is not time anybody waits for.

### Both eyes at once, when the machine can afford it

Two FaceFusion processes, one per eye, one lane each. It is **gated**, and on
this card at 4K with a face enhancer the gate says no and is meant to: one pass
holds ~7.0 GB of a 16.3 GB card, so two plus the desktop is 15.4 GB and WDDM
starts paging GPU memory into host RAM, which is the failure that freezes the
machine for minutes at a time (see "cuDNN algorithm search" below).

`parallel_eyes_verdict()` measures rather than assumes - free VRAM against
`2 x estimate_vram_mb + 1500`, available commit and RAM against two lanes'
worth - and the answer is logged before anything starts, either way:

```
[Guard] Both eyes at once: no - VRAM 15.1 GB needed for two lanes (6.8 GB each
        plus 1.5 GB spare), 15.9 GB on the card with 1.7 GB in use.
[Guard] Both eyes at once: yes - 5.2 GB of VRAM, 12.4 GB committed and 7.1 GB
        of RAM per lane, at 3 threads each - two of those fit.
```

`parallel_eyes` in Settings (under Chunked Processing) is `auto`, `on` or
`off`. `on` skips the commit and RAM margins - the user has decided they would
rather page than wait - but never the VRAM one, because that is not a slow run.
Each lane gets half the execution threads, since the two processes share one
GPU. The verdict is taken again immediately before the lanes start: VRAM
belongs to the whole machine and a split takes minutes.

The estimate follows the run rather than the other way round - one
`Face swap, both eyes together: 41m` line instead of two 36 m ones, at the
measured-until-proven 15% GPU-sharing penalty - and the two lanes share that
stage's slice of the bar, reporting the mean of their two positions.

If one lane dies, or the watchdog gives up on it, the other is allowed to
finish and the failed eye is then re-run on its own with the full thread count.
The per-eye checkpoints are unchanged, so a resume still reuses whichever eye
finished. Two failures is the ordinary error path.

### The second lane died instantly, and it was not memory

The first real two-lane run failed in a way worth writing down. The right eye
ran; the left exited at once with a non-zero code and **not one line of
output**, which surfaced as `FaceFusion failed: Unknown error`.

It was FaceFusion's job store. `headless-run` names its job after the clock to
the second (`job_helper.suggest_job_id`), writes it into a shared `.jobs`
directory beside the FaceFusion checkout, and `process_headless()` returns 1
without logging anything when the job already exists. Two lanes launched in the
same second - 12:48:22, both of them - therefore asked for the same job id, and
the loser died silently every time. Nothing to do with VRAM, and no amount of
guard-rail headroom would have fixed it.

`run_facefusion()` now passes `--jobs-path <temp path>/jobs`. Every pass
already has its own temp directory, so two lanes cannot collide, and a single
pass behaves exactly as before.

### What two lanes actually did on this card

10 s of the 8192x4096 60 fps test source (600 frames per eye), face enhancer
off, three execution threads per lane, nothing else running:

| | swap stage | peak VRAM (system) | GPU |
|---|---|---|---|
| one eye alone, 3 threads | 179.8 s | 6.2 GB | ~70% |
| both eyes at once, 3 threads each | **253 s for both** (248.9 + 249.3 in parallel) | **10.8 GB** | 92-96% |

Two lanes sequentially would be 359.7 s, so the swap stage came out at **0.70x
wall - a 1.42x throughput gain** - which is what the 25-30% of idle GPU in a
single pass was worth, and squarely inside the 1.25-1.35x the plan predicted.
Peak VRAM stayed 5.5 GB clear of the card, and the merged output holds exactly
600 frames in each eye. The whole 10 s pipeline ran in 5m 54s.

None of that changes the verdict at 4K **with** the enhancer on: 6.8 GB a lane
against a 16.3 GB card is still a no, and the gate still says so.

### What was tested

`scripts/tests/` is new. 25 tests, no GPU needed, `python -m unittest discover
-s scripts/tests`:

* **Lanes** (`test_lanes.py`): two sleeping children in two lanes under a real
  job object, with the suspend state read back out of the kernel's own process
  snapshot rather than from anything guardrails.py believes. Freezing one lane
  leaves the other running; a user Pause and Resume leaves a guard-held lane
  held; a user Pause over two already-frozen lanes still takes, so releasing
  them does not restart the run behind the user's back; killing a lane leaves
  the other alive; and the run clock only stops when both are frozen.
* **The gate** (`test_parallel_eyes.py`): the verdict against fixed machine
  readings, including this card's real numbers at 4K with an enhancer (no) and
  without one (yes); `on` and `off`; the forced lane hold, driven through the
  real watchdog with `lane_pause_free_vram_mb` set above the whole card, which
  freezes the newest lane and then kills only it; and `run_vr_pipeline` with
  both eyes in flight and FaceFusion stubbed - the two passes really do overlap
  in time, get half the threads each, and a lane that dies costs the run
  nothing but that eye.
* **The overlap** (`test_overlap.py`): a three-chunk run on a synthetic
  side-by-side clip with the swap replaced by a sleep and *everything else
  real* - real ffmpeg splits, merges and joins. It asserts on the clock that
  chunk 2's split finished while chunk 1 still had the GPU and chunk 1's merge
  finished while chunk 2 did, that each chunk had its own scratch directory,
  and that the joined output holds exactly the same number of frames as the
  same run with `overlap_stages: false`.

What is *not* covered by a test, and was not run: the three-chunk 150 s
overlap-on-versus-off timing from the plan, which is about four hours of GPU
per pass at 4K, and any run of the lane watchdog against two real FaceFusion
processes under genuine memory pressure. The forced-hold test proves the
policy; it does not prove what a 16 GB card does at the moment it runs out.

## 📋 A LIST OF JOBS, NOT ONE JOB AT A TIME (Sep 9 2026)

One run at a time is all this machine can do - a second FaceFusion would fight
the first for the same 16 GB of VRAM - but one run per sitting is a different
limit, and it was the one the app imposed. Every render meant coming back to
the keyboard. Now there is a **Queue** tab: a list of jobs that run one after
another, top to bottom, each one no earlier than a time written against it.

### Adding jobs

Every tab (VR, Normal, Upscale) has an **Add to Queue** button beside Run. It
freezes the tab's fields *and the settings in force at that moment* into a
job, so the Settings tab can be changed for the next job without touching the
ones already waiting. A job is checked when it is added - missing files, a bad
time - so a queue armed for the night is a queue of jobs that could actually
start; the files are checked again at start time, because a night is long
enough for a drive to be unplugged.

**Start At** still works on each tab, and the Upscale tab now has one too. Run
with a time in that box is the same as Add to Queue plus Start Queue: the job
waits in the list with its time against it.

The generated output names get a fresh timestamp after each add (and a `_2`,
`_3` suffix when two adds land in the same second), so two queued jobs never
point at the same file. A path typed by hand that another waiting job already
writes to is refused.

### Running it

**Start Queue** works down the list. A job with no time runs as soon as the
machine is free; a job with a time waits for it, and the progress line counts
down (`Next: VR swap: clip.mp4 at 02:30:00 - 1h 12m to go`). By default the
untimed jobs behind a timed one run first rather than leaving the GPU idle
until 02:30; tick *Keep strict order* to wait instead. A job in flight is never
interrupted for a timed one - "not before" is an earliest start, not a slot.

A queued job is unattended: it is never forced past the guard rails and never
blocks on a dialog nobody will answer. A failure is recorded against the job
and the queue moves on, unless *Stop the queue when a job fails* is ticked.
**STOP** kills the job in flight *and pauses the queue* - someone pressing STOP
wants the machine back, not the next job a second later. The waiting jobs stay
put; Start Queue carries on. **Pause Queue** lets the current job finish and
holds the next.

### The list

Each row shows the job, its portion, preset and output name, and what happened
to it (`Done  |  1h 12m`, or the error). Waiting jobs have a *Not before* box
(a clock time, `tomorrow 02:30`, `+90m`, or a date and time - type it and press
Enter), up/down arrows, **Edit** (puts the job back on its tab, settings and
all, to be changed and re-added) and **Remove**. Finished ones have
**Re-queue**. **Re-queue Failed** and **Clear Finished** work on the whole list.

An immediate Run also goes through the queue as a running job, so the tab is
a record of everything this window has done.

### It survives a restart

The list lives in `output/queue.json` beside `session.json`. A job that was
running when the app closed comes back as waiting, marked interrupted; its
chunks and finished eyes are still in `output/temp`, so it resumes from them.
The queue never starts itself on launch - a machine that rebooted overnight
should not begin rendering the moment the window opens - it says how many jobs
are waiting and leaves Start Queue to be pressed.

## 🧩 A RUN THAT IS NEVER MORE THAN A MINUTE FROM SOMETHING WATCHABLE (Sep 6 2026)

The checkpoints keep finished *stages*, and a stage is one eye of the whole
clip. So a run that died at 80% had a left eye, a right eye's frame dumps,
and nothing anyone could play. The whole selected range was one bet.

Now the range is cut into chunks of about a minute (`chunk_seconds` in the
settings, under "Chunked Processing"; 0 is the old single-run behaviour) and
each chunk goes through the entire pipeline on its own - split, both swaps,
merge, smoothing, upscale, VR tag - landing as a complete, playable file in
`<output>_chunks/part_NNN.mp4` beside the final output. When the last one is
done they are joined into the requested file and the chunk folder is removed.

```
CHUNKED RUN: 12 chunks of ~1m 00s covering 12m 00s (0s to 12m 00s)
  Chunks are written to: output/final/vr_deepfake_20260906_LR_180_chunks
[Chunks] Starting chunk 7/12: 6m 00s to 7m 00s
[Chunks] About 41m 10s left overall (done around 23:05)
...
[Chunks] Stopped during chunk 7/12: Free RAM down to 2.4 GB ...
[Chunks] 6 finished chunk(s) are kept in ... - each is a playable video of its own range:
  - part_001.mp4: 0s to 1m 00s
  ...
[Chunks] Start the run again with the same settings and it continues from chunk 7.
```

### You are told before it starts

The label beside Clip End reads `323.2s selected - 6 chunks of ~60s` as soon as
a range is picked (or the chunk length is changed in Settings), and pressing
Run lists every chunk with its timecodes and the folder they will land in
before the preflight even begins. The estimate line says `Estimated 1h 26m in
6 chunks - finishing around ...`.

### What a failure costs now

At most one chunk, and usually less: the inner checkpoints still work inside
a chunk, so a retry of chunk 7 skips whichever eye it had finished. The chunk
folder carries a `chunks.json` whose key covers the source, the target, every
setting and the chunk boundaries - a chunk is only reused when all of that is
unchanged, and a mismatch says so and starts over rather than splicing two
different renders together. A chunk is also only trusted at the length it was
meant to be; the upscale stage writes to its final name directly, and an abort
in the middle of that leaves a headerless file that ffprobe reads as 0 s.

### The join is a copy, and the audio is not from the chunks

Every chunk comes out of the same encoder with the same settings, so the
concat demuxer splices the video without re-encoding: seconds, not another
hour, and no extra generation of loss. Verified on a synthetic 135 s clip cut
into 60+60+15: the joined file has exactly 4050 frames at 30 fps - the
`-ss`/`-to` input trims partition frames without duplicating or dropping one
at a seam. If the copy is refused the join falls back to one re-encode and
says so.

Audio is cut from the *source* across the whole range in one piece and muxed
in alongside. Each chunk's own AAC track carries its encoder priming, and
splicing a dozen of those puts a tick at every boundary. The VR tag is
re-applied after the join because the join rebuilds the container header.

### Preflight and the progress bar

Temp usage is now one chunk's worth, and the disk check is sized to that -
which is what lets a full-length 8K clip start at all. The time estimate is
still for the whole range, because that is what the wait is. The bar runs
across the whole run in proportion to seconds of video done, chunks reused
from an earlier attempt count as done from the start, and the "left" figure
on it is for the whole run, measured from this run's own pace once a chunk
has finished. A tail shorter than 10 s is folded into the chunk before it
rather than paying FaceFusion's model load for three seconds of video.

## 🧊 A RUN THAT FREEZES INSTEAD OF DYING (Sep 6 2026)

The memory trip points were a cliff: above them the run continued, below them
it was killed and the stage in flight - up to an hour of 8K swapping - went
with it. There was nothing in between, and "the machine briefly got busy" is
not a good reason to lose an hour.

Now there is something in between. The watchdog gets a second, higher trip
point, and crossing it FREEZES the run rather than ending it:

```
[Guard] Free RAM down to 4.8 GB - FREEZING the run instead of letting it reach
        the 2.5 GB abort line.
[Guard] Of the 27.0 GB in use, this run's own processes hold 19.4 GB.
[Guard] Froze 6 process(es) and handed back 19.4 GB; free RAM is now 22.1 GB.
        The run resumes by itself once 7.5 GB is free.
[Guard] Free RAM is back to 22.1 GB - thawing the run after 4s. Nothing was
        lost; the estimate is not charged for the wait.
```

### Freezing alone would not have worked

Suspending a process frees no physical memory by itself. Windows trims a frozen
process eventually, under pressure, on its own schedule - and when the machine
is seconds from an abort, "eventually" is not a plan. So the freeze is followed
immediately by `EmptyWorkingSet` on every process in the job, which hands the
resident pages back now; they go to the modified and standby lists and become
available as the modified page writer drains them, which on an NVMe pagefile is
a second or two. Measured: a 4 GB run went from 15.4 GB free to 19.2 GB free
and thawed itself two seconds later.

That call is only ever made on a process that is already frozen. Doing it to a
running process would make it fault its whole world straight back in, which is
all cost and no benefit.

### The three ways out, in order

| free RAM | what happens |
|---|---|
| above 7.5 GB | runs normally; a frozen run thaws here |
| below 5.0 GB | frozen and trimmed, then waits |
| below 2.5 GB, or frozen for 30 min | aborted, keeping every finished stage |

The hold has a timer because a run that never gets its memory back was not
going to finish, and holding the work hostage indefinitely helps nobody. When
it does expire, the checkpoints mean the next attempt still starts from the
last finished stage.

Free RAM only, deliberately. Freezing returns *resident* pages; it does not
return committed address space, so it cannot fix a commit exhaustion, and
triggering on commit would only spend the hold timer before aborting anyway.
Commit is raised by the pagefile, and the app now says so at launch.

### Who is holding the freeze matters

`pause()` records whether it was the user or the guard. Only a freeze the guard
applied is ever lifted automatically - a run the user paused stays paused, no
matter how much memory comes free. The PAUSE button follows the guard's state
rather than its own, because the watchdog can now put a run into RESUME without
anyone having pressed anything, and a button that lied about that would resume
a run the user believed they were pausing.

`RunClock` already excluded frozen time from every span it measures, so a run
that spent twenty minutes frozen still reports its stages honestly and still
writes honest numbers into `output/timing.json`.

## 🔎 AN ABORT THAT NAMES A CULPRIT (Sep 6 2026)

A run died with `free RAM down to 0.0 GB` two and a half minutes into the
split, and the same log said `Peak memory used by pipeline: 3.1 GB`. Those two
lines together are the whole problem: the machine ran out of memory and the
pipeline was not using any.

### The split was measured, and cleared

The exact split command was run against the 8192x4096 source for 150 seconds
with per-second sampling. It is flat: ffmpeg holds 2.28 GB working set and 3.03
GB private for the entire run, the job's peak matches at 3.03 GB - the figure
`_decode_args` predicts - and free RAM never left the 18-22 GB band. The split
cannot take 23 GB because the split does not take 23 GB.

### What the log could not say

`peak_memory_bytes()` answers "did the pipeline do this?" and the answer was
no. Nothing answered "then what did?", so a user reading that abort would
reasonably reach for a smaller pixel boost or more RAM, and neither would have
changed anything.

The watchdog now attributes the memory before it kills anything - on the first
warning and again at the abort:

```
[Guard] Of the 8.5 GB in use, this run's own processes hold 0.6 GB.
[Guard] Largest processes outside this run: LEDKeeper2.exe 0.4 GB, ...
[Guard] Kernel pools hold 2.4 GB (nonpaged 1.3 GB); commit charge is 21.0 GB
        of 63.8 GB, peak 63.8 GB.
[Guard] The pipeline is NOT what filled the memory - closing whatever is
        listed above will do more than lowering any setting in this app.
```

The kernel-pool line is there because adding up every process's working set
accounts for only about half of what Windows calls "in use" on this machine,
and an abort you cannot attribute is an abort you cannot fix. `GetPerformanceInfo`
supplies the rest: pools, system cache, and the commit figures.

### What it found here

`commit_peak` is 63.78 GB against a `commit_limit` of 63.78 GB. This machine
has already driven commit charge to its ceiling since boot, and at that ceiling
Windows trims every working set hunting for pages - which is precisely what
"free RAM down to 0.0 GB" looks like from inside a process that is behaving.

The ceiling is RAM plus pagefile. `guardrails.json.pre_commitlimit.bak` records
a commit limit of 102.1 GB; it is 63.8 GB now, because the pagefile is a fixed
32 GB. The guard had already noticed and correctly lowered its own cap from 51
GB to 31.9 GB - but 31.9 GB of pipeline plus a ~21 GB idle baseline leaves
about 10 GB for everything else on a machine whose commit peak is the limit.

The app now says this at launch, before a job that may run for days, rather
than after one has been killed.

### And a resume hint that was lying

The same log ended with:

```
[Resume] These stages finished and were kept in output/temp:
  - left eye split
  - right eye split
```

while the line above it was `Processing Left eye...` - the left eye split had
just been interrupted. `_log_resume_hint` listed everything in `run_state.json`
without checking whether any of it was still reusable, and those two records
belonged to a different, 332-second target from an earlier job.

A record that cannot be reused is not merely useless, it is a trap: it names a
file the current attempt is about to overwrite, so an abort part-way through
that rewrite leaves the record standing beside a half-written file. Stale
records are now dropped up front, before anything is written, so the failure
hint only ever promises work that actually survived.

## ⏸️ PAUSE, AND START LATER (Sep 5 2026)

Two ways to run a job that owns the machine for hours without the job owning
the machine for hours.

### Pause freezes the run; Stop still ends it

There was one answer to "I need this computer back": Stop. Since the
checkpoints landed that at least keeps the finished stages, but the stage in
flight is still lost, and at 8K a single eye is the better part of an hour.

`ResourceGuard.pause()` suspends every process in the job instead. The work
stays exactly where it is - mid-frame, models loaded - and Resume carries on.

**How it finds the processes.** Not by walking the process tree. FaceFusion
spawns ffmpeg, a tree walk keyed on parent PIDs misses re-parented children,
and a recycled PID means freezing something that has nothing to do with this
run. The guard already owns a Job Object that every child is assigned to, so it
asks the kernel: `QueryInformationJobObject(JobObjectBasicProcessIdList)`
returns exactly the processes in the job, which by construction is exactly what
this run started. Each one gets `NtSuspendProcess`, parents first so nothing
can spawn behind the freeze - and `spawn()` suspends any process created during
a pause, which closes the one remaining race.

**What it costs.** A frozen run keeps its RAM and its VRAM. This gives back the
CPU and the GPU's *time*, not the card's memory. So: Pause to use the machine
for something else, Stop to get the graphics card back. Both are now stated on
screen when the button is pressed, because the two buttons look
interchangeable and are not.

**The watchdog holds its fire while paused.** A frozen run touches nothing and
Windows is free to trim and page out its whole working set, so it is not what
is eating the machine. Aborting there would destroy precisely the work the user
paused to protect. It keeps sampling and warning, does not accumulate strikes,
and arms again on resume.

**The clock does not charge for it.** `RunClock` takes the guard's pause meter
and subtracts frozen time from every span it measures: total elapsed, the pace
correction, each stage's duration and each decile. Without that, a run paused
overnight would report a nine-hour merge, halve its own pace estimate, and -
worst of the three - write "a merge on this machine costs nine hours" into
`output/timing.json`, where it would go on lying to every future estimate.

### Start At

A "Start At" box on the VR and Normal tabs. Empty means now; otherwise
`02:30`, `tomorrow 02:30`, `+90m`, `+2h30m`, or a full `2026-09-08 22:05`. A
bare clock time that has already gone today means tomorrow - typing 02:30 at
midnight means tonight, typing it at 03:00 means the next one.

The countdown is armed **after** the inputs are validated, so a run only ever
waits if it could actually have started, and the box is cleared as it arms:
the run that fires must not schedule itself again, and pressing Run tomorrow
must not silently wait until 02:30 all over again. STOP cancels a countdown
rather than killing anything, and closing the window asks first, since a
scheduled start needs the app open.

One deliberate asymmetry: when the guard rails refuse a run that started on a
schedule, it is **not** offered the usual "start anyway?" dialog. Nobody is at
the keyboard - that is the entire point - and a modal at 03:00 means the run
never begins and nothing on screen says why. It logs the refusal loudly and
stays idle instead.

## 🧠 THE APP REMEMBERS THE JOB (Sep 5 2026)

"Resume from where the last attempt stopped" had been ticked by default since
the checkpoint work went in, and output/temp really did keep the finished eyes.
It still felt like it had never worked, and for two reasons that had nothing to
do with the checkpoints.

### The window forgot everything at every launch

Every field was built empty: source face, target video, clip start, clip end,
FPS, layout, output path. Reopening the app after an overnight stop meant
browsing back to the source, browsing back to the target and retyping the in
and out points from memory.

That is not only tedious, it is the bug. The target file and the clip range are
*part of the resume key*, so a start time typed back as 61 where it had been
60.5 moved the key, and a moved key means the split is redone - three and a
half gigabytes per eye, several minutes at 8K, thrown away for half a second of
typing. The feature that was meant to save an hour of GPU time was being
defeated by the text box above it.

The window now writes its fields to `output/session.json` and reads them back
at launch. It saves on a one-second debounce as they are edited, when a run
starts - a job the watchdog kills never reaches the close handler, and its
inputs are exactly the ones needed to resume it - and on close.

Two details that are not just "dump the variables":

- **Re-extract is deliberately not remembered.** It is a one-shot that
  `finish_run` already clears; persisting it would re-run the source face
  analysis at every launch, which is the exact cost the extraction cache
  exists to avoid.
- **The generated output name is restored as a generated name.** It carries a
  launch timestamp, so a restored one has to be re-adopted as "the name this
  app made" or the layout radio stops renaming it. A run that *finishes* then
  retires that name and hands the next job a fresh timestamp, so a second run
  cannot silently overwrite the render just produced - while a run that
  *failed* keeps its name, so the retry writes to the file the resume is
  rebuilding.

### A resume that gave up said nothing about why

When a key did not match, the run simply redid the stage. No line in the log,
no mention of what had moved. From the outside that is indistinguishable from
the checkpoints not existing.

`run_state.json` now stores the named inputs behind each key, not just the
hash, and `RunState.why_not()` diffs them:

```
[Resume] Redoing the left eye swap: the quality settings (enhancer_model
  (gpen_bfr_512 -> gpen_bfr_1024)), the split (the clip range (whole video ->
  0.0s to 61.0s)) changed since it ran.
```

The same check runs at launch, against the fields just restored, so the state
of the drive is on screen before anything is started:

```
[Resume] output/temp still holds finished stages from an earlier attempt:
  - left eye split - finished 2026-09-05 18:13:03, 3.5 GB on disk
      WILL BE REDONE: the target video (other.mp4 -> raya.mp4) changed since it ran.
```

Which is how the 7 GB currently in `output/temp` turned out to be two eyes of a
332-second video that is no longer in the project - kept, and unusable, and
nothing had ever said so.

Unticking Resume now also names what it is about to discard, because throwing
away a finished eye is the better part of an hour of GPU time and should never
happen quietly.

The split key composition is untouched by any of this, on purpose: rearranging
it would have invalidated every split already on disk.

## ⏱️ HOW LONG WILL THIS TAKE (Sep 5 2026)

A run said "Progress: 34%" and nothing else. On a job whose honest answer is
"an hour and a half" - or seventy hours - that is not enough to decide whether
to start it, and the percentage itself was not even a percentage of the wait.

### The bar was not measuring time

The stage boundaries were hardcoded and had no relationship to what each stage
costs:

| stage | share of the bar | share of the clock |
|---|---|---|
| split, both eyes | 5% | 4% |
| swap, left eye | 38% | 43% |
| swap, right eye | 37% | 43% |
| merge | 9% | 10% |

Close enough by luck on the default settings, and wrong the moment anything
else is switched on: turning RIFE on adds hours and 6% of the bar. Worse, the
optional stages ran *backwards* - the merge set the bar to 91 and temporal
smoothing then set it to 87 - and the Real-ESRGAN upscale reported its own
0-100 into the same bar, so a VR run that reached 92% jumped back to 0.

`RunClock` now allocates the bar in proportion to each stage's estimated
seconds and refuses to let it move backwards, so a percentage is a percentage
of the wait.

### The estimate

`guardrails.estimate_runtime()` models every stage as

```
seconds per frame = fixed + per_megapixel * frame_megapixels
```

fitted to the measurements already in this document, all taken on this machine
against the 8192x4096 90 fps source (one eye = 16.78 MP):

| stage | measured | s/frame at the anchor |
|---|---|---|
| swap | 2.5 frame/s per eye | 0.400 |
| split | 50 frame/s per eye | 0.020 |
| merge | 11 frame/s at 33.55 MP | 0.091 |
| RIFE | 0.96 synthesised frame/s | 1.042 |
| smoothing | ~1 frame/s at 9.44 MP | 1.000 |

Checked against the two runs this project has actual figures for: the
60-second clip comes out at **1h 23m** and the 50-minute projection at
**2d 21h** - the same ~70 hours the projection section arrived at by hand.

RIFE gets its own two-size model, because the network runs at `infer_scale` of
the frame while the resizes, the 50 MB-per-frame pipes and the NVENC encode
all run at full size. Fitting the resolution table (6.1 / 3.0 / 1.6 frame/s at
2048 / 3072 / 4096 square) gives the model term, and the 0.96 frame/s measured
end to end at scale 0.75 gives the rest.

Quality settings move the *fixed* term only, since per-face work does not grow
with the frame: pixel boost 512 is a batch of four tiles (x2.2), gpen_bfr_1024
is x1.9 on top, the expression restorer is a second generative model per face
(x1.45), and no enhancer at all is x0.55. Those are ratios of model work, not
timings - they exist so that ticking Pixel Boost 512 does not leave the
estimate still claiming an hour and a half.

### It corrects itself, twice

**Within the run.** Every progress callback compares estimated-seconds-spent
against wall-clock-seconds-spent and scales what is left by the ratio, bounded
to [0.25, 4]. A model that is 40% wrong stops being 40% wrong a few minutes in.

**Between runs.** Each finished stage writes `actual / predicted` into
`output/timing.json` as an exponential average per stage kind, bounded the same
way, and the next estimate is multiplied by it. A stage that only did half its
work - a split where one eye was resumed - is explicitly not recorded, because
half a split is not a sample of what a split costs.

### What it says

Before anything spawns, next to the existing disk and RAM lines:

```
[Time] Estimated total: ~1h 23m (finishing around 21:47)
  - Split into left and right eyes: 3m 36s (4%)
  - Face swap, left eye: 35m 59s (43%)
  - Face swap, right eye: 35m 59s (43%)
  - Merge the eyes back into one stereo video: 8m 11s (10%)
  First run: this is the model's guess and it will be corrected from the
  real pace within the first few percent.
```

Then at every 10% of the run:

```
[Time] 60% - that 10% took 14m 02s, 42m 10s elapsed, about 33m left
       (done around 22:31)
```

and on every stage boundary, what that stage cost against what it was expected
to cost, and what a reused stage saved. The progress label carries the
remaining figure continuously, so the number is on screen without reading the
log.

### The split and the merge stopped being silent

Both are single blocking ffmpeg calls - three and seven hours respectively on a
full-length job - and neither reported anything, so the bar sat still and the
remaining figure had nothing to correct against. `-progress pipe:1` writes
`frame=N` to stdout independently of `-loglevel`, so both now report frames as
they are written. Verified on a 2-second trim: 21 progress callbacks from the
split, 35 from the merge, output identical (4096x4096 eyes, 8192x4096 merged).

### The preset names were promising times they could not keep

`Good (~31 min/1min video)` was a single 2D face-swap pass. A VR run swaps each
eye separately and splits and merges 8192x4096 frames around them, so a minute
of VR source at 'Good' is nearer ninety minutes than thirty-one. Four numbers
that are only right for one of the two tabs are worse than none, so the names
now say what the preset does - `Good (swap + 512 enhancer)` - and the time
comes from the estimator, which knows the clip, the resolution and the
settings.

### Nothing expensive is on by default

Asked in passing, and worth writing down, because three separate settings sound
like they might be:

| setting | default | what it would cost |
|---|---|---|
| Frame Rate Multiplier (RIFE) | **1, off** | 1h 33m per eye on a 60 s clip |
| Frame Enhancer | **none** | a generative model over every full frame |
| Output Upscaling | **none** | hours, and several times the file size |
| Temporal Smoothing | **off** | cannot run at VR resolution at all |

The whole 1h 23m of a default 60-second VR run is the face swap (86%) plus the
split and merge around it. There is nothing switched on to switch off - the
cost is 5,407 frames x two eyes at 16.78 MP each, and the ways to move it are
fewer frames (trim, or drop Output FPS to 45), a lighter preset, or a smaller
frame.

---

## 🚀 RIFE FRAME INTERPOLATION - BUILT (Sep 5 2026)

`minterpolate` could not run on VR frames at all, so real interpolation now
runs on RIFE. It is off by default (`interpolate_multi: 1`) because it is the
most expensive thing in the pipeline by a wide margin.

### PyTorch was a dead end, so it is ONNX

Every mainstream RIFE implementation is PyTorch. Not usable here:

```
NVIDIA GeForce RTX 5060 Ti with CUDA capability sm_120 is not compatible
with the current PyTorch installation (sm_50 ... sm_90).
```

`torch.cuda.is_available()` still returns **True** on this box, and then every
kernel launch fails - the same trap that stalled the Real-ESRGAN work.
onnxruntime's CUDA provider works on the card today (FaceFusion runs on it), so
RIFE goes through onnxruntime instead: vs-mlrt's `rife_v2` export of RIFE
v4.15-lite, 10 MB, fetched by `python scripts/rife_worker.py --install-model`.

The v2 interface is the one worth having - one float32 input of shape
(1, 7, H, W): frame0 RGB, frame1 RGB, and a constant timestep plane - and it
pads internally, so any frame size works with no rounding to a multiple of 32.

### It is genuinely interpolating, and that was checked

Leave-one-out against ground truth: predict frame n+1 from n and n+2 and score
against the frame that is really there, with a linear crossfade as the baseline
since that is what you get for free.

| | PSNR vs the real frame |
|---|---|
| crossfade | 42.73 dB |
| RIFE, inferred at 2048, upscaled | 45.05 dB |
| RIFE, inferred at 3072, upscaled | 46.58 dB |
| RIFE at full 4096 | 47.50 dB |

Structure was checked separately in the finished file: even output frames match
the source frames, and each synthesised frame sits between its neighbours
(0.581 / 0.573 against a 0.756 gap) rather than duplicating either.

### Why the default is 3/4 resolution, not full

| infer at | frame/s | peak VRAM (16.3 GB card) |
|---|---|---|
| 2048x2048 | 6.1 | 6.6 GB |
| 3072x3072 | 3.0 | 8.7 GB |
| 4096x4096 | 1.6 | **15.8 GB - 97% of the card** |

Full resolution leaves ~550 MB for the desktop, the NVENC encoder the worker is
feeding, and anything else on the GPU. The first run at full resolution did not
merely slow down - it thrashed to ~0.03 frame/s and died after 13 frames, which
is the WDDM spill this project has now hit three separate ways.

3/4 scale costs 0.92 dB, runs nearly twice as fast, and leaves real headroom.
Only synthesised frames go through the model; every original frame is passed
through untouched.

### What it costs

End-to-end, including the CPU resizes, the 50 MB-per-frame pipes and the NVENC
encode: **0.96 synthesised frame/s** on a 4096x4096 eye.

| job | frames to synthesise | time |
|---|---|---|
| 60 s VR clip, 2x, both eyes | 10,812 | **3.1 hours** |
| 50-minute file, 2x, both eyes | 540,000 | **156 hours** |

Preflight now says this before the run starts, and says it in those words. The
90 fps source this project is being used on has very little to gain from 180;
24-30 fps footage has a great deal.

### Where it sits

Per eye, after the swap, before the merge - and that is forced, not preferred:
the model needs 8.7 GB on a 4096x4096 eye and the merged 8192x4096 frame is
four times the pixels. It is a checkpointed stage like the others, so an abort
does not throw away a finished eye, and it runs in the guard's Job Object so
the STOP button and the watchdog reach it.

Verified end to end: a 1-second VR run produced 8192x4096 at **180 fps**, 179
frames, with the VR metadata intact, peak 6.96 GB RAM and 14.08 GB free
throughout.

---

## 🎞️ FRAME RATE: THE FPS BOX DOES NOT INTERPOLATE (Sep 5 2026)

The source is **90 fps** and the pipeline preserves it exactly - the finished
60-second output is 5407 frames over 60.077 s, i.e. 90.000 fps. Nothing is
being lost, and 90 already matches the Quest's display.

The FPS box in the GUI is a *resampler*, not an interpolator: it becomes
ffmpeg's `-r`, which duplicates or drops whole frames. Setting it to 120 yields
a 120 fps file containing 90 fps of motion - judder, plus a third more bitrate
for nothing. Its genuinely useful direction is downward: at 45 it halves the
frames the swap has to process, which halves the dominant cost of a long run.

### minterpolate cannot run on this content at all

`apply_temporal_smoothing()` uses ffmpeg's `minterpolate`, the only real
interpolation wired up. It has a hard ceiling that has nothing to do with the
machine:

| frame | megapixels | result |
|---|---|---|
| 3840x2160 | 8.29 | OK, 35 s for 43 frames |
| 3072x3072 | 9.44 | OK, 42 s for 43 frames |
| 4032x4032 | 16.26 | OK |
| 4064x4064 | 16.52 | OK |
| **4096x4096** | **16.78** | **FAIL** - "Cannot allocate memory", instantly |
| **8192x4096** | **33.55** | **FAIL** |

It failed with 17 GB of RAM free, in 0.2 s, during filter setup. The cause is
`minterpolate` allocating three per-pixel tables for the whole frame as single
blocks, and `av_calloc` refusing any one allocation over `INT_MAX`. The
predicted ceiling from the largest of those structs is 16.27 MP; measured, it
sits between 16.52 and 16.78. **One VR eye is 16.78 MP and the merged stereo
frame is 33.55**, so both are out of reach and always were.

Speed rules it out anyway: ~1 output frame/s at 3072x3072. A 50-minute VR job
is 540,000 frames - 150+ hours on top of the 70 the swap already costs.

### The bug this was hiding

With "temporal smoothing" ticked, a VR run would launch ffmpeg, have it fail
during filter setup, and then fall into `shutil.copy(input_path, output_path)`
- silently duplicating the whole merged file (45 GB on a 50-minute job) to hand
back something byte-identical to the input. The log line said "using original"
and nothing about why.

It now checks the frame size before doing any work, explains the limit and what
would fit, and **moves** rather than copies, since both callers treat the input
as dead the moment it returns. Verified: 8192x4096 skips with the explanation
and produces the output by move; 1920x1080 still really interpolates.

Real frame interpolation at VR resolution needs **RIFE**, not minterpolate.
That is now built - see "RIFE frame interpolation" above.

---

## 📏 WHAT A FULL-LENGTH SCENE COSTS (Sep 5 2026)

Projecting a 50-minute, 11 GB source (same ~30 Mbit/s 8K VR format as the test
clip) against this session's measured throughput and file sizes:

| | 8192x4096 @ 90 fps | @ 60 fps |
|---|---|---|
| frames to swap | 540,000 | 360,000 |
| split, both eyes | 3 h | 3 h |
| swap, both eyes | 60 h | 40 h |
| merge | 7 h | 5 h |
| **total** | **~70 h** | **~48 h** |
| peak temp disk | 109 GB | 109 GB |
| peak RAM | 7 GB | 7 GB |

Rates behind it: swap 2.5 frame/s per eye, split 1.8x realtime per eye, merge
11 frame/s at 8192x4096. **RAM does not move with length** - resolution sets it,
which is the whole point of the earlier section. Disk and time do.

### The disk estimator was wrong by 20x

Running the projection through `estimate_disk_need()` returned **2,362 GB** for
that job. Two independent errors:

1. It charged for FaceFusion extracting every frame to disk. That is the `disk`
   workflow; `run_facefusion()` pins `--workflow-strategy memory`, which pipes
   decoded frames straight to the encoder and writes no frames at all. That
   term alone invented ~2.2 TB.
2. It summed every intermediate as if they all coexisted, and sized them by
   bits-per-pixel. They do not coexist - each is deleted as soon as the stage
   consuming it finishes - and their size is set by the NVENC rate cap, not by
   resolution, which is why a 4096x4096 eye and an 8192x4096 stereo frame come
   out at similar bitrates.

Now it walks the run and takes the worst moment (the merge, holding both
swapped eyes plus the file being written from them), and sizes each encode as
`min(bits-per-pixel, rate cap)` so both regimes are right - 1080p never reaches
the cap, 4K-per-eye is pinned to it. Checked against the files this session
actually produced from the 60-second clip:

| | estimated | measured |
|---|---|---|
| split, one eye | 0.59 GB | 0.57 GB |
| swapped, one eye | 0.72 GB | 0.72 GB |
| merged stereo | 0.90 GB | 0.89 GB |

The 50-minute job is still blocked - 109 GB needed against ~101 GB free on B: -
but now for the true reason and with a number worth acting on.

### Known limits for a job this long

- **Resume is per stage, not per frame.** `RunState` checkpoints split / swap /
  merge, so an abort 20 hours into an eye re-does that whole eye. Fine for a
  60-second clip, expensive here.
- **The pagefile freeze compounds with runtime.** It hit roughly half of the
  short runs; over 70 hours it is close to certain. Still a Windows setting -
  see the pagefile section.

---

## 🥽 THE OUTPUT PLAYED AS FLAT 2D IN THE QUEST (Sep 5 2026)

A finished 60-second run played in the Quest 3 as an ordinary 2D video on a
curved screen. Nothing was wrong with the picture - a frame pulled from
`vr_deepfake_20260905_081929.mp4` is a correct 8192x4096 side-by-side VR180
stereo pair with the swap applied. The file simply never said what it was.

### What was missing

An MP4 declares itself as VR in two boxes inside the video sample entry:

| box | says |
|---|---|
| `st3d` | how the two eyes are packed - left-right, top-bottom, mono |
| `sv3d` | how pixels map onto a sphere, and how much sphere they cover |

Neither was present. Checked with ffprobe and by scanning the container:
no side data, no `st3d`, no `sv3d`, no `proj`, no `uuid`.

**The SLR source has none either.** It plays correctly on the headset purely
because it is named `..._LR.mp4`, and DeoVR, Skybox and Pigasus read that
convention before they look at anything else. Our output was called
`vr_deepfake_20260905_081929.mp4`, which tells a filename-driven player nothing
and a metadata-driven player (Quest's own media player, YouTube VR) nothing
either. So it fell back to flat 2D - which Quest renders on a curved screen.

`apply_stereo_metadata()` could never have fixed this. It wrote Matroska's
`stereo_mode`, which ffmpeg silently drops for MP4 - the code even logged
"*.mp4 cannot store the stereo_mode tag*" and moved on. And in an MKV that tag
only describes eye packing, never projection, so a tagged MKV would still have
been a flat stereo pair rather than a sphere.

### The fix

ffmpeg's avformat can read *and* write these boxes (`st3d`, `sv3d`, `equi`,
`prhd`, `svhd` are all in the DLL), but only by propagating spherical side data
from an input that already has it - there is no CLI option to originate it.
`-side_data_prefer_packet` only picks a precedence. So `scripts/vr_metadata.py`
writes the boxes directly, to Google's Spherical Video V2 layout:

```
st3d                       stereo_mode = 2 (left-right) or 1 (top-bottom)
sv3d
  svhd                     "VrDeep"
  proj
    prhd                   pose, all zero
    equi                   bounds left = right = 0.25, top = bottom = 0
```

Those `equi` bounds are what make it **VR180** rather than 360: the frame holds
the middle half of a full equirect sphere, so a quarter is cropped from each
side. 180-vs-360 is decided from the shape of one eye - a square eye is 180
degrees, a 2:1 eye is the whole sphere - so a 360 source still gets tagged
correctly.

Verified by round-trip through an independent parser. ffprobe on the patched
file now reports:

```
side_data_type=Stereo 3D          type=side by side
side_data_type=Spherical Mapping  projection=tiled equirectangular
                                  bound_left=4097  bound_right=4095
```

Those pixel bounds look odd but are right: ffmpeg reconstructs the full 360
width (8192 / 0.5 = 16384) and reports the middle 8192 as the region kept,
which is exactly VR180.

**Safety.** Only `moov` is read into memory - a few MB of what may be a
gigabyte file - and the rest is streamed through, so tagging a 891 MB output
costs no more RAM than tagging a small one (which, after the previous section,
is not a detail to be careless about). Inserting bytes inside `moov` shifts
anything after it, so every `stco`/`co64` chunk offset at or past the insertion
point is corrected; when `moov` follows `mdat`, which is how this pipeline's
files come out, nothing needs moving. 64-bit box sizes in the `moov` chain are
detected and refused rather than mis-written.

Applied to the existing 60-second output: file grew from 891,183,764 to
891,183,864 bytes - exactly the 100 bytes of new boxes - still decodes to the
final frame, still 5407 frames.

### Filenames still matter

Metadata alone does not cover DeoVR, Skybox or Pigasus, which decide from the
name first. The GUI's generated output name now carries the convention
(`..._LR_180.mp4`, following the layout radio), and a run whose output name
lacks it logs the exact name to use. A path the user typed or browsed to is
never rewritten underneath them.

---

## 🧊 WHY THE MACHINE FREEZES: IT IS THE PAGEFILE, NOT THE RAM (Sep 5 2026)

The previous section capped what the pipeline holds and made preflight size a
run against physical RAM. It was not enough: a 60-second clip still froze the
box solid for 10-20 seconds and then got killed by the watchdog, on about half
of all attempts. This is what that actually was.

### Measured first

A 3-second trim of the 8K test clip, sampled every 0.3-0.5 s across the whole
process tree, with a realtime-priority sampler so it kept recording while the
desktop was unresponsive:

| stage | peak resident | peak committed |
|---|---|---|
| split (per eye) | 3.79 GB | 4.58 GB |
| FaceFusion pass (per eye) | 7.09 GB | 15.0 GB |
| merge | 5.46 GB | 6.86 GB |

Steady state during a run: **17.3 GB of RAM still free, 41.8 GB committed.**
The run's own footprint is flat and never grew - across five full runs the peak
resident figure was 7.06, 7.07, 7.09, 7.11 and 7.16 GB. Nothing leaks.

The two runs that died did so with the whole tree still at 7 GB and free RAM
going 17 -> 11 -> 1.6 -> 0.0 in under two seconds, with no process holding the
missing memory. That is not a program using too much RAM. That is Windows
unable to get paged-out memory *back*.

### The cause

```
C:\pagefile.sys  24 GB  WD_BLACK SN770 1TB      SSD   (NVMe)
D:\pagefile.sys  24 GB  ST1000DM010-2EP102      HDD   <- 7200 rpm
E:\pagefile.sys  24 GB  ST1000DM010-2EP102      HDD   <- same physical disk
```

Two thirds of this machine's 72 GB of pagefile lives on a single mechanical
disk, and D: and E: are two partitions of that *same spindle*, so they also
contend with each other. Peak usage recorded on them: 4.4 GB and 2.5 GB - so
those pagefiles are not decorative, they are actively used.

Windows does not prefer the fast pagefile. It stripes paging across all of
them. And paging is not avoidable here: 32 GB of RAM against 41.8 GB committed
means the system pages during every run no matter how tidy the pipeline is.
Roughly two thirds of that traffic goes to a disk that does ~120 random IOPS,
so the instant anything needs those pages back, everything stops - the desktop,
the app, and the watchdog thread that is supposed to notice.

Windows had already said so once, on Aug 27:
*"Your system is low on virtual memory... memory requests for some applications
may be denied."*

**The fix is a Windows setting, not code**: System > About > Advanced system
settings > Performance > Advanced > Virtual memory - set D: and E: to *No
paging file* and leave the pagefile on C: (NVMe) only.

Because that is a machine setting the app cannot change on the user's behalf,
the app now *names* it. `guardrails.pagefiles()` resolves every pagefile to its
physical disk and media type; `slow_pagefile_warning()` turns a spinning one
into a warning with the exact fix. It is printed at startup, again in preflight
before a run, and appended to the watchdog's abort message when free RAM is
what tripped it - so "the machine ran out of RAM" stops being the last word.

### What was reduced anyway

ffmpeg still defaulted to one decode thread per core in the pipeline's own
stages (the FaceFusion patch only covered FaceFusion's readers). At 4096x4096
and 8192x4096 each of those threads holds its own decoded frames.
`engine._decode_args()` caps them, and the cost of doing so is nil:

| | default | -threads 4 | -threads 2 | time |
|---|---|---|---|---|
| split 8192x4096 -> one eye | 4.58 GB | 3.25 GB | 2.88 GB | 36.3 s vs 36.1 s |
| merge two eyes -> 8192x4096 | 6.86 GB | 5.33 GB | 4.86 GB | 24.5 s vs 24.3 s |

Identical wall-clock in every case - the decoders were never the bottleneck,
NVENC is. Applied to the split, merge, Lanczos upscale, temporal smoothing,
the normal-pipeline trim and the source-face sampler. End-to-end this took the
split stage from 3.79 to 2.47 GB resident and the merge from 5.46 to 3.69 GB.

Two other things came out of the same measurements:

- **`estimate_resident_mb()` had the wrong shape.** It charged ~350 MB per
  execution thread; measured, a thread is worth ~160 MB (4 threads 7.09 GB,
  1 thread 6.61 GB) while the three ffmpeg children are ~1.45 GB *each* and
  nearly flat. Refitted, it now predicts 7.4 GB where 7.1 GB is measured -
  conservative in the right direction, and correct about where the memory is.
- **`clamp_config()` was making a bad trade.** It reduced execution threads to
  claw back headroom, but 4 threads -> 1 saves 0.48 GB and takes a pass from
  73 s to 150 s. It now only reduces threads when the run does not otherwise
  fit at all, stops the moment it does, and says in the log how much slower the
  run just became and that closing other programs would avoid it.

A patch that did *not* survive measurement: capping FaceFusion's hard-coded
`set_thread_count(16)` on its video *writer*. 16 threads vs 1 made no
difference to anything (2.46 GB vs 2.45 GB, same runtime), so it was reverted
rather than left in someone else's codebase as decoration.

---

## 🔁 RESUME AFTER A FAILURE + WHERE THE RAM GOES (Sep 4 2026)

Two complaints, one investigation: a 1-minute VR clip runs the machine out of
RAM, and the retry after that starts again from scratch.

### The "only 20 GB" is free RAM, not total RAM

`GlobalMemoryStatusEx` reports 31.8 GB total and **19.9 GB available** on this
machine. The header read `RAM free 19.9 GB`, which looks like a 20 GB box. The
missing 12 GB is other programs - during this investigation that included two
of this user's other Python projects (a uvicorn server and BuzzEdit) plus the
usual desktop. Both the header and the startup limits block now spell out
total / in use / free, and say which of those numbers a run is measured
against.

That distinction matters because there are **two different memory ceilings**
and they had drifted apart:

| ceiling | value here | measured against |
|---|---|---|
| `job_memory_cap_gb` | 51 GB | *committed* memory (RAM + pagefile = 102 GB) |
| watchdog `min_free_ram_gb` | 2.5 GB | *physical* RAM (31.8 GB, ~20 GB free) |

The job cap was raised to half the commit limit in August to stop premature
kills, and nothing has sized a run against physical RAM since. So preflight
happily passed runs the watchdog then aborted part-way through.
`estimate_resident_mb()` fills that gap, and preflight refuses (with the
numbers, and with "close other programs frees X GB") before anything starts.

### Where the RAM actually goes: resolution, not length

The test clip is **8192x4096 at 90 fps**. Per eye that is a 4096x4096 frame -
**48 MB of raw pixels each**, sixteen times a 1080p frame - and 5400 frames per
eye for 60 seconds. Sampled during a real left-eye pass:

| process | resident | committed |
|---|---|---|
| FaceFusion python | 2.35 GB | 9.0 GB |
| ffmpeg raw-frame reader | 2.35 GB | 2.36 GB |
| ffmpeg raw-frame reader | 1.38 GB | 2.29 GB |
| ffmpeg writer | 2.30 GB | 2.31 GB |
| **total** | **8.4 GB** | **16.0 GB** |

**The ffmpeg children were 72% of it**, and for a dull reason: they ran with
ffmpeg's default thread count, which on 28 cores is 16 frame-decoding threads
each holding its own pool of 48 MB frames. `scripts/patch_facefusion.py` caps
the readers at 4 threads - still an order of magnitude faster than the swapper
consumes frames.

The same script fixes a second thing, which was not what bit here but is a
loaded gun: `process_memory_frames()` submitted a future for **every frame of
the video** before draining any of them. `ThreadPoolExecutor`'s queue is
unbounded, so there was no backpressure at all - any stall on the writer let
the workers run arbitrarily far ahead, and every frame of lead holds another
48 MB result alive. 5400 frames x 48 MB is 260 GB of worst case. It now submits
inside a window of `threads * 2 + 2`.

> `facefusion/` is an untracked clone (see `.gitignore`), so these live as an
> idempotent script rather than as edits nobody can see. `prepare_run` logs a
> warning if they are missing; `--check` reports status without changing
> anything.

### Resume: a failed run keeps its finished stages

`run_vr_pipeline` had `cleanup_temp(temp_dir)` in a `finally`, so an abort
during the *right* eye threw away a *left* eye that had rendered perfectly -
45 minutes of GPU time, every retry.

Each stage that produces a file now records it in `output/temp/run_state.json`
with a key covering exactly its own inputs, and a retry skips any stage whose
key still matches and whose file is still intact:

- **the split does not depend on the face**, so re-extracting the source face
  invalidates both eyes but not the (multi-minute, 8K) split
- **upscaling and smoothing are post-swap settings**, so turning them on does
  not throw two finished eyes away
- **an interrupted encode is not reused**: the recorded duration is compared
  against the file's actual duration, because an aborted encode leaves a file
  that opens perfectly well and holds half the frames

Only a *successful* run clears temp now. A failed one keeps its finished
stages and logs what survived; the frame dumps, which are the part that fills
the drive, are dropped either way. There is a **Resume from where the last
attempt stopped** checkbox on the VR tab (on by default) to force a clean run.

### Also fixed: every error dialog was a NameError

`except PipelineError as e: self.after(0, lambda: ...str(e))` - Python unbinds
`e` at the end of the `except` block, so by the time Tk ran the lambda the name
was gone:

```
NameError: cannot access free variable 'e' where it is not associated
with a value in enclosing scope
```

Every pipeline failure in the GUI showed *that* instead of what actually went
wrong, which is why none of this was diagnosable from the log. Four call sites,
now `lambda err=e:`.

### Worth knowing

- **90 fps triples everything.** The preset timings ("~31 min/1min video") were
  measured on 30 fps footage. This source is 90 fps, so a 1-minute clip is
  5400 frames per eye, ~47 GB of temp and ~45 minutes for the left eye alone.
  Setting Output FPS to 30 cuts time, temp and memory by three.
- Resolution, not clip length, is what decides whether a run fits in RAM.
  Trimming a clip does not reduce peak memory at all.

---

## 🧱 THE PIPELINE'S OWN UPSCALE STAGE (Sep 3 2026)

`upscale_vr.py` is the standalone upscaler. This is the *other* one - the
Output Upscaling stage inside the deepfake pipeline (`engine.upscale_video`,
used by both VR mode's stage 6 and Normal mode's stage 4). Three ways it
failed, all fixed.

### 1. Neither NVENC ceiling was checked

`upscale_video_ffmpeg` passed whatever Target Width said straight to the
encoder. Past the limit, ffmpeg says:

```
[hevc_nvenc] No capable devices found
Error while opening encoder - maybe incorrect parameters such as bit_rate,
rate, width or height
```

which names neither the cause nor a fix - and in the pipeline it arrived
*after* the swap, i.e. half an hour in.

Measured on this card by encoding one black frame at each size:

| encoder | 4096x4096 | 4098x2160 | 5120x2880 | 8192x4096 | 8192x8192 |
|---|---|---|---|---|---|
| `h264_nvenc` | OK | **FAIL** | **FAIL** | **FAIL** | **FAIL** |
| `hevc_nvenc` | OK | OK | OK | OK | OK |
| `av1_nvenc` | OK | OK | OK | OK | OK |

So the cap is **per dimension, not on total pixels**, and H.264's is 4096 -
half of HEVC's 8192. New `ENCODER_MAX_DIMENSION` plus `check_upscale_encoder`,
which refuses with the measured limit and the ways out. Called from
`prepare_run`, so a bad Target Width now fails in the first second.

`upscale_output_size()` is shared by the preflight and the upscale itself: a
check that validates a different number from the one reaching the encoder is
worse than no check.

### 2. `h264_nvenc` was the default encoder on all three upscale entry points

Latent - every caller passes `config["encoder"]` - but it is the encoder with
the *lower* ceiling, so the default was the one most likely to fail. Now
`hevc_nvenc`.

### 3. The Real-ESRGAN preflight gave advice that would not have worked

`prepare_run` checked only for the weights and told the user to install them.
It would still have failed afterwards: **torch is 2.5.1+cu121, built for
sm_50-sm_90, and this card is sm_120.** `esrgan_worker.py` already knew that
and checked it properly - but only once spawned, at the upscale stage.

The worker now takes `--preflight`: same checks, no video touched, exits
immediately. `prepare_run` spawns it and reports what it says. One copy of the
logic, and the run is refused before the swap rather than after it.

### Also: the disk estimate ignored the upscale entirely

`estimate_disk_need` counted temp frames and intermediates but not the
upscaled result, which is the largest single file a run produces - 4x the
source area at scale 2, 16x at scale 4. It now takes the pipeline config and
includes it, so the preflight stops under-estimating exactly the runs most
able to fill the drive.

### Still true, not fixed here

- **VR mode upscales the *merged* frame** (`engine.py`, stage 6), so lanczos
  reads across the eye seam - the exact thing `upscale_vr.py` was written to
  avoid. Upscaling per eye before the merge would fix it and drop a re-encode
  generation.
- **`nvenc_maxrate_mbps` is still 120** for every size. At 1440p that is
  generous; at 8K the extra resolution is thrown away at the cap.

---

## 🎥 A SOURCE VIDEO CAN NOW BE THE FACE INPUT (Sep 2 2026)

The Source Face box takes a short clip as well as photos. Mixing the two works
and is often the best input there is - a sharp portrait for detail plus a clip
for the range of poses.

### Why this is worth having at all

FaceFusion's `--source-paths` accepts images only, and for each one it takes the
largest face and **averages the ArcFace embeddings**
(`face_creator.average_face_identity`). So multiple stills were always better
than one; the awkward part was finding several good photos of the same person.

A clip is a better way to fill that list - but only if the wrong frames are
thrown away first. Averaging is unweighted, so a blurred or badly lit or
turned-away frame does not merely rank low, it drags the mean embedding off the
person. That selection is the entire feature.

### What it does with a clip

`scripts/face_source_worker.py`, spawned through the guard like
`esrgan_worker.py`, so onnxruntime and its CUDA context leave with the process
and the memory cap, CPU cap, watchdog and Stop button all reach it.

1. **Samples** the clip through ffmpeg's `fps` filter - sequentially, not by
   seeking, because seeking a long-GOP file lands on the wrong frames and costs
   a full decode anyway. One frame is held at a time; a 4K frame is 25 MB as
   bgr24 and four hundred of them would be 10 GB.
2. **Measures** every sampled frame with FaceFusion's own analyser - the same
   detector, landmarker, recogniser and classifier the swap uses, so what is
   measured is the face it would actually pick.
3. **Gates** on the raw measurements, before anything is ranked: too blurred
   (variance of the Laplacian below 35% of the clip's own median - relative,
   because the absolute value depends on resolution and codec), too dark or
   blown out, or turned more than ~30 degrees away.
4. **Clusters by identity** and keeps only the largest group. One frame of
   somebody walking behind the subject is enough to average two people into a
   third who is neither.
5. **Picks for spread**, not for score: farthest-point selection over yaw, roll,
   mouth openness and time, weighted by quality. Top-N by quality alone returns
   the same moment six times, and six copies of one view average to what one
   view does.
6. **Saves face-centred crops**, not whole frames. FaceFusion's detector resizes
   whatever it is given to 640x640, so a 400px face inside a 4K frame arrives 66
   px across - cropped, it arrives at full size. The crop also guarantees the
   intended face is the largest one in the still.

Verified on a built test clip (sharp / blurred / second-identity / dark
segments, 4 fps sampling):

```
blur gate      16 of 40 sampled frames rejected; no pick from the blurred or
               the dark segment
identity       two-person clip, gates disabled to force both through:
               8 frames rejected as a different identity, all picks from the
               dominant one
```

### Wiring

| | |
|---|---|
| `engine.resolve_source_faces()` | photos pass through, videos are replaced by their stills. Called **once per run** - VR calls `run_facefusion` twice, and both eyes must swap against the same identity or they disagree in the headset. |
| Cache | `output/source_faces/<clip>_<hash>/`, keyed on the file plus the settings that change the result. Cold 6.7 s, warm 0.0 s. Not under `output/temp`, which `cleanup_temp` wipes. |
| Re-extract | a checkbox on both tabs, cleared after each run, for a clip edited in place under the same name and timestamp. |
| `run_facefusion()` | now **refuses** a video source with a sentence naming the fix. FaceFusion silently filters non-images out of `--source-paths` and then swaps nothing. |
| `run_pipeline.py` / `run_facefusion.py` | the CLI takes a clip too; `source` became `nargs="+"`. |

Settings tab: a **Source Face from Video** group - stills to keep (6), sample
rate (2 fps), max frames sampled (400), min face size (160 px), identity
threshold (0.45).

End-to-end verified: a source clip through the full VR pipeline, both eyes
logging the same three stills, PSNR 33.3 dB against the untouched target (a
no-op re-encode would be 45-55 dB).

### Also fixed

**`save_settings` and `apply_settings` held byte-identical 35-line copies of the
same widget-reading loop.** That is precisely the shape where a new setting gets
added to one and silently never reaches a run. `save_settings` is now
`apply_settings` plus the write. Verified that a bare `apply_settings()` against
the on-disk config changes no key at all.

---

## 🧊 UPSCALE FROZE PARTWAY THROUGH (Sep 2 2026)

Reported as: start an upscale, watch it stop at 6% and never move again. It was
a pipe deadlock in the progress reporting added the same day, and it was two
mistakes compounding.

### What happened

`run_ffmpeg_with_progress` spawned ffmpeg with `stderr=subprocess.PIPE` and
**never read it**. Separately, `-nostats` was inserted at the *front* of the
command while the base command still carried `-stats` further along - and with
ffmpeg the last flag wins, so stats stayed switched on and streamed to that
stderr the whole run.

Once the OS pipe buffer filled, ffmpeg blocked forever on the write. Not slow -
stopped, with the progress bar frozen wherever it happened to be.

It survived testing because every test clip was fifty frames, which never
produces enough output to fill the buffer. Only a real encode does.

Reproduced deliberately with the old command shape on a 2003-frame clip:

```
OLD SHAPE: last frame seen = 229 of 2003 (11%) after 100s
  -> stopped advancing 82s before we gave up
```

The exact stall point varies with content and encode speed, which is why it
showed as 6% rather than 11%.

### Fixed

- `-stats` is **removed** from the command rather than countermanded, so there
  is nothing continuously writing to stderr. (It is still used on the CLI path
  that has no progress callback, where the console wants it.)
- stderr is **merged into stdout**, so one reader drains one pipe. That cannot
  deadlock by construction.
- Non-progress lines are kept as a short tail, so a failure can now report what
  ffmpeg actually said instead of only a return code.

Verified on the same 2003-frame clip: **rc=0, 153 s, 294 monotonic progress
updates, 1.5 GB out** - far past where it used to hang.

### The same latent bug in two other places

An audit of every `spawn`/`Popen` in the project found two more with an unread
`stderr=PIPE`, both of which write for the life of a long job:

| | |
|---|---|
| `face_source_worker.decode_frames` | reads frames off stdout one at a time for the whole decode, so nothing can drain stderr concurrently |
| `esrgan_worker` encoder | writes frames into stdin for the whole run and only calls `communicate()` at the end |

Neither could merge stderr into stdout - the first has raw frame data coming
back on stdout, and merging would corrupt it. Both now send stderr to a
`tempfile.TemporaryFile()`. **A file never blocks**, which is the general fix
whenever the reader cannot run concurrently. ffmpeg is quiet at `-loglevel
error`, but "usually quiet" is not a guarantee and the failure mode here is a
permanent hang rather than an error.

---

## 🔍 STANDALONE VR UPSCALER (Sep 2 2026)

**`scripts/upscale_vr.py`** - point it at any video, get a bigger one. Nothing
to do with the deepfake pipeline; it only borrows `engine`'s encoder helpers and
the resource guard.

```
python scripts/upscale_vr.py in.mp4 -o out.mkv --target 8k
python scripts/upscale_vr.py in.mp4 -o out.mkv --target 8k --layout tb
python scripts/upscale_vr.py in.mp4 -o out.mkv --target 4k --layout mono
python scripts/upscale_vr.py in.mp4 -o out.mkv --target 16k --split-eyes
```

### Two things that make it more than `ffmpeg -vf scale`

**The eyes are scaled separately.** A resize filter reads a neighbourhood around
every output pixel - lanczos reaches three source pixels either side - so
scaling a side-by-side frame as one image pulls left-eye pixels into the right
eye along the seam. It is a narrow band, but it is the one place the two views
are guaranteed to disagree, which is exactly the signal used to judge depth.

**One encode, not three.** Split-to-disk / upscale / merge is three lossy
generations for a job that needs one. For the resize methods the crop, the two
independent scales and the restack are a single ffmpeg filtergraph, so the
source is decoded once and encoded once.

### Measured, 50 frames of 4K SBS

| | |
|---|---|
| 8K SBS, lanczos, `hevc_nvenc` | **10.8 frame/s** |
| 8K TB, lanczos | 10.9 frame/s |
| 16K as two per-eye files, `hevc_nvenc` | 2.95 frame/s |
| 16K merged, `libx265` | 1.14 frame/s |
| 8K, `real_esrgan_x2_fp16` | ~0.03 frame/s (see below) |

### The encoder ceiling is checked before any work starts

NVENC on this card stops at 8192x8192 and NVDEC will not *decode* wider than
8192 either, so 16K crosses the line in both directions. Asking for it used to
mean reaching ffmpeg and dying with `No capable devices found`, which names
neither the cause nor the fix. Now it refuses up front with the three real
options, and `--split-eyes` is one of them: **each eye at 16K is 7680x7680,
which is inside the hardware limit**, so two per-eye files both encode and
decode normally where the merged file does neither.

### The guard caught a genuine problem

The first 16K merged software encode drove free RAM to 0.6 GB and the watchdog
aborted it - correctly. x265's defaults assume a frame you can hold a lot of,
and one 15360x7680 yuv420p frame is 177 MB:

```
preset veryfast, 8 threads (the default)    1.68 frame/s   23.6 GB peak  -> aborted
+ frame-threads=1, rc-lookahead=10, 4 thr   1.46 frame/s   14.8 GB peak  -> completes
preset ultrafast, frame-threads=1, la=5     4.64 frame/s    9.6 GB peak
```

`encoder_args_for()` now re-tunes libx265 whenever the frame exceeds HEVC level
6.2's sample limit. Serialising the frame threads costs ~13% of the speed and
halves the memory, which is the right trade when the alternative is a run that
does not finish.

### Neural super-resolution is wired but gated

`--method esrgan` runs FaceFusion's ONNX frame enhancer per eye - the only
super-resolution that works on this machine's stack. It **must** pass
`--output-video-scale` or the entire upscale is discarded (see the plan
document); this does.

Because that flag only moves in 0.25 steps, the reachable size is not always the
requested one, so the geometry is snapped to what will actually come out
*before* the summary prints it. Asking for 1920 wide from an 852-wide source
reports 1916 and delivers 1916, rather than announcing one size and writing
another.

It also prints a wall-clock estimate from the measured rate and stops for
confirmation past an hour. For a 60 s 4K SBS clip that estimate is **16.6
hours**, against about 3 minutes for lanczos.

### It is a GUI tab now, not just a CLI script

The upscaler was CLI-only, which meant it did not exist as far as the app was
concerned. There is now an **Upscale** tab beside VR Mode and Normal Mode.

`upscale_vr.main()` was split into `build_parser()` + `run_upscale(args, ...)`,
and the tab calls `run_upscale()` directly with a `SimpleNamespace` carrying the
same attribute names argparse produces. So both front ends drive one code path -
no shelling out, and no second copy of the logic to drift.

**A live preview** sits above the Run button and updates as the settings change,
because the preflight's whole value is answering "will this work, and will it
even be visible" *before* an encode rather than after one:

```
Layout: MONO - nothing declares a stereo layout, so treating it as flat 2D.
1920x1080 -> 7680x4320  (4x)
Meta Quest 3: 129,600 of 138,240 blocks, up to 60 fps at this size (source is 25)
Estimated time: about 0.1 minutes (50 frames at ~10 frame/s).
```

`describe()` runs the same checks the run does, so the preview cannot promise
one thing and the run deliver another.

### The Upscale tab owns its Encoder and Quality

They were being read from the Settings tab, which configures the *deepfake
pipeline* - a different job with different trade-offs. A software encoder here
is a deliberate choice to exceed 8192px, not a fallback. The tab now has its own
Encoder dropdown and Quality slider, and Settings is left alone (verified: the
Settings quality stays 95 while the tab runs at 90).

Encoders offered: `hevc_nvenc`, `av1_nvenc`, `h264_nvenc`, `libx265`, `libx264`.
The live preview updates on every change, so an impossible combination shows
before the encode rather than after it.

### Two real bugs this surfaced

**`av1_nvenc` could not run at all.** `headset_encoder_args` hardcoded
`-profile:v main`, which is an HEVC/H.264 profile name. `av1_nvenc` has **no
`-profile` option whatsoever** and refused to open:

```
[av1_nvenc] Undefined constant or missing '(' in 'main'
[av1_nvenc] Unable to parse "profile" option value "main"
```

`-pix_fmt yuv420p` is what actually guarantees the 8-bit 4:2:0 the hardware
decoder wants, so the profile pin was belt-and-braces anyway. It is now emitted
only for HEVC, along with the `hvc1` tag, which is equally HEVC-specific -
ffmpeg tags AV1 and H.264 correctly on its own.

**`h264_nvenc` stops at 4096x4096, not 8192.** The ceiling check used one
constant for all of NVENC. Measured per encoder by trying them:

```
hevc_nvenc  8192x8192 OK   10240x5120 refused
av1_nvenc   8192x4096 OK   15360x7680 refused
h264_nvenc  4096x4096 OK    5120x2560 refused
```

H.264 is the odd one out at half the limit, and every NVENC encoder fails the
same unhelpful `No capable devices found` way - so picking H.264 at 8K produced
exactly the error the ceiling check exists to prevent. `ENCODER_MAX_DIMENSION`
is now per-encoder, and when the encoder is what is holding things back the
advice leads with the free fix rather than with splitting eyes:

```
CANNOT ENCODE: 7680x3840 cannot be encoded by h264_nvenc: it stops at 4096x4096.
Switch to hevc_nvenc or av1_nvenc - they reach 8192px, h264_nvenc stops at 4096px.
```

All four encoders verified end to end at their own limits.

### Two bugs the GUI work exposed

1. **`run_upscale` ignored a caller-supplied guard.** It always built its own
   `ResourceGuard`. The GUI starts one and its **STOP button acts on that
   instance**, so the real work would have gone into a Job Object the Stop
   button could not reach, with two guards fighting over the same limits. It
   now reuses the caller's and only closes what it opened.
2. **The progress bar never moved.** A single ffmpeg call reported nothing, and
   an 8K encode of any real length runs for minutes. `run_ffmpeg_with_progress`
   parses `-progress pipe:1` (machine-readable `key=value`, far more reliable
   than the self-rewriting `-stats` line). Verified monotonic 0->99 on the
   merged path and across both files on the per-eye path, which gets half the
   bar each rather than jumping back to zero halfway.

Verified end to end through the GUI's own event loop: 8K SBS output written,
progress bar reached 1.0, guard released, button re-enabled.

### Flat 2D video, and layout auto-detection

Ordinary 2D video was always handled (`--layout mono`), but the default was
`sbs`, so pointing the tool at a normal file without remembering the flag
cropped half the frame away and duplicated it - a silently wrong result that
looks plausible until you watch it. `--layout auto` is now the default.

**Comparing the two halves of the frame does not work as a detector**, which is
worth recording because it is the obvious first idea. Measured across real
files, PSNR between halves:

```
genuinely stereo footage   ~13 dB
ordinary flat footage      ~13 dB
an exact duplicated eye     47 dB
```

Parallax between two eyes is about as large as the difference between two
unrelated halves of an ordinary shot, so the distributions sit on top of each
other. Only an exact duplicate separates, and that is not the case worth
detecting. `detect_layout()` therefore uses declarations, not pixels:

1. the container's `stereo_mode` tag - authoritative, and verified readable
2. the file-name markers VR libraries use (`_LR`, `_SBS`, `_TB`, `_3dh`, …)
3. otherwise **mono**, with a note when the aspect is ~2:1 and it might be VR

The fallback direction is deliberate and asymmetric: reading stereo as flat
scales the packed frame as one image and costs only a little filter bleed at
the seam, while reading flat as stereo destroys the frame. An unsure answer
goes to the harmless side. `--content` now follows the layout too, so a flat
file is not measured against a headset's field of view.

```
wedding.mp4      MONO: nothing declares a stereo layout
real_vr4k.mp4    MONO: ... the 2.00:1 aspect is typical of VR180 side-by-side,
                       so if this IS stereo pass --layout sbs
tagged.mkv       SBS:  the file declares stereo_mode=left_right
holiday_LR.mp4   SBS:  the file name contains '_lr'
```

A 1920x1080 clip goes to **7680x4320** - true 8K UHD - at 10.1 frame/s, which
is 129,600 blocks and inside the Quest 3's per-frame limit with 60 fps of
headroom.

### `--headset quest3`, with limits read off the actual device

The published "Quest 3 does 8K30" figure is a floor, not the limit. Pulled from
the headset itself over adb - `/vendor/etc/media_codecs.xml`,
`c2.qti.hevc.decoder` (and `c2.qti.av1.decoder`, which is identical):

```
<Limit name="size"              max="8192x8192"/>
<Limit name="block-size"        value="16x16"/>
<Limit name="block-count"       range="36-138240"/>
<Limit name="blocks-per-second" max="7776000"/>
<Limit name="bitrate"           range="1-160000000"/>
```

The decoder has a **throughput budget in 16x16 blocks per second**, not a fixed
frame rate, so resolution and frame rate trade against each other and 8K30 is
one point on a curve rather than the end of it:

| frame | blocks | fits | max fps |
|---|---|---|---|
| 4K SBS 3840x1920 | 28,800 | yes | 270 |
| 6K SBS 5760x2880 | 64,800 | yes | 120 |
| **8K SBS 7680x3840** | 115,200 | yes | **67.5** |
| 8192x4320 (the true max SBS) | 138,240 | yes | 56.2 |
| 12K SBS 11520x5760 | 259,200 | **no** | - |

So the real numbers are **67 fps at 8K, not 30**, and **160 Mbit/s, not 100**.
Both were guessed conservatively before and are now measured. `wm size` also
reports 4128x2208, confirming the 2064x2208 per-eye panel the sampling maths
assumed.

`decoder_budget()` models this properly, so a 60 fps 8K source no longer warns
(it is inside the budget) while a 90 fps one does, and says what to do:

```
Decoder : 115,200 blocks/frame of 138,240 max; headroom is 67.5 fps (source is 90)
WARNING : 90 fps at 7680x3840 needs 10.4M blocks/s, over the 7.8M the decoder has.
          Drop to 68 fps at this size, or pick a smaller --target.
```

New `--target 8.2k` (8192 wide) exposes the extra headroom, and `--content
vr180|vr360` reports whether the resolution is even visible:

```
6k  vr180  0.85x  Below the panel - more resolution would genuinely help
8k  vr180  1.14x  Well matched to the panel
12k vr180  1.71x  Past what the panel can show
8k  vr360  0.57x  Below the panel
```

Only ~61% of a VR180 frame is in view at once (110 degrees of a 180 arc), so at
8K about 2347 source px per eye land on a 2064 px panel. **For VR180 the panel,
not the decoder, is the limit** - which is why 8K is the right answer and 12K
is not. VR360 spreads the same pixels over twice the arc and is genuinely
short.

### Encoder settings the headset needs

There are two ceilings and the lower one is not this PC. NVENC stops at 8192;
the headset's own decoder stops before that, so a file can come out of NVENC
perfectly and still not play. `--headset` applies the second one:

| | why |
|---|---|
| rate cap 100 Mbit/s (not the project default 120) | sustained read while also decoding 8K |
| GOP 2 s (NVENC defaults to 250 frames = 10 s at 25 fps) | verified: 8 s clip goes from **1 keyframe to 4**. A 10-second GOP makes scrubbing feel broken. |
| `-tag:v hvc1` on mp4 (ffmpeg writes `hev1`) | signalling only, but it is the difference between "plays" and "black screen" in several players |
| `-pix_fmt yuv420p -profile:v main` | pinned rather than inherited; 10-bit or 4:2:2 drops the device to software decode |
| warns past 7680x3840, and past 30 fps at that size | verified: a 60 fps source warns at 8K and is silent at 6K |

**8K is the sweet spot because the two ceilings line up.** 7680 is just inside
NVENC's 8192, and 8K is the largest the XR2 Gen 2 is specified to decode. That
is not a coincidence worth pushing past: 12K (11520) and 16K (15360) are over
the line in both directions, so they are not merely slower, they are unplayable
as a single file. Measured for completeness:

```
12K merged, libx265        0.99 frame/s   level 189 (out of spec), no hw decode
12K per eye, hevc_nvenc    4.97 frame/s   level 180 (in spec), hw decode OK
```

### Also
- Named targets describe the **per-eye** resolution, so "8K" is 7680x3840
  side-by-side and 3840x7680 top-bottom - one eye is 3840x3840 either way.
  Sizing from the packed width instead would have made 8K TB mean four times
  the pixels of 8K SBS.
- Audio is stream-copied, never re-encoded.
- `stereo_mode` is stamped on the result (Matroska only), same as the pipeline.

---

## 📄 8K / 16K UPSCALING INSIDE THE DEEPFAKE PIPELINE: PLAN, NOT BUILT (Sep 2 2026)

This is about the *pipeline's own* output resolution. Upscaling an existing
file is a separate, finished thing - see the standalone upscaler above.

See **`UPSCALE_8K_16K_PLAN.md`**. Measured on this machine, the headlines:

- `hevc_nvenc` and `av1_nvenc` both stop at **8192x8192**, and NVDEC will not
  decode wider than 8192 either. 8K VR SBS (7680x3840) is fine; 16K SBS
  (15360x7680) has no hardware path at all, and x265 encodes it only by
  stamping an out-of-spec level.
- **Per eye, both resolutions fit in hardware** - 7680x7680 per eye encodes
  fine. Only the final merge crosses the line, so the architecture already has
  the right tiling.
- **A live defect found while measuring**: `to_video.py` sizes the output
  encoder from the *target* video times `--output-video-scale` (default 1.0),
  and this pipeline never passes that flag. So selecting an x2/x4/x8 frame
  enhancer runs the full super-resolution and then **scales the result back
  down to the source size** - measured, 426x226 in, 426x226 out. The note in the
  Aug 26 entry recommending `real_esrgan_x2_fp16` as the working alternative to
  the broken torch path does not do what it says.
- Passing `--output-video-scale` is therefore most of the 8K/16K feature,
  already built and unexposed: 2.0 on a 1920x1920 eye gives 8K after the merge,
  4.0 gives 16K.
- Neural super-resolution is not viable here regardless: `real_esrgan_x2_fp16`
  measured **0.027 frame/s at 4K input**, i.e. ~18.5 hours per eye per minute of
  footage. Lanczos to 8K runs at 9.08 frame/s.

---

## 💥 "BEST" PRESET KILLED NORMAL-MODE RUNS (Aug 27 2026)

Reported as: pick **Best** from the quality dropdown in Normal mode, and the
run - and the app with it - falls over. Root cause is the guard rails' own
memory cap, which was sized against the wrong number.

### The job memory cap was measured against the wrong thing

`ResourceGuard` puts every pipeline child in a Windows Job Object with a
`JobMemoryLimit`. That limit is on **committed** memory. `default_guard_config`
derived it from **physical RAM** (`min(total_ram * 0.62, total_ram - 10 GB)`),
which on this 31.8 GB machine gave **19.7 GB**.

Those two numbers are nowhere near each other for a CUDA process. Measured
during a run:

```
FaceFusion python process   13.1 GB committed   2.3 GB working set
```

The CUDA runtime and the WDDM driver reserve address space they never make
resident, and all of it counts against the cap.

### How close Best actually ran to it

Peak job commit (`ResourceGuard.peak_memory_bytes`), Normal mode, 4K target,
3 execution threads:

| preset | clip | peak commit | % of the 19.7 GB cap |
|---|---|---|---|
| Good (pb256 + gpen_bfr_512) | 18 frames | 11.1 GB | 56% |
| Best (pb512 + gpen_bfr_1024) | 18 frames | 15.5 GB | 79% |
| Best | 480 frames (20 s) | **19.5 GB** | **99%** |
| Best | 1440 frames (60 s) | 17.2 GB | 87% |

Best sits between 79% and 99% of the ceiling and the peak is spiky rather than
monotonic - the 20 s clip peaked higher than the 60 s one, because the spikes
come from model loads and from ffmpeg readers being torn down and recreated on
every seek, not from clip length. Real footage seeks far more than these
static-image test clips do. So Best crossing the cap is a coin flip, which is
exactly what "it failed when I chose Best" looks like.

Once the cap is reached every allocation inside the job fails, so FaceFusion
dies mid-run with an onnxruntime or numpy memory error that says nothing about
the guard rails. Normal mode is the worst case: VR mode processes one eye per
pass, i.e. half the pixels.

### Fixes

| # | Fix | Where |
|---|-----|-------|
| 1 | Cap derived from the **commit limit** (RAM + pagefile, 102.1 GB here), keeping half in reserve and never going below physical RAM. 19.7 GB -> **51.0 GB**. Physical-memory pressure - the thing that actually freezes Windows - is the watchdog's job, not this cap's. | `guardrails.default_guard_config`, `guardrails.json` |
| 2 | New `commit_limit_bytes()`; `_machine.commit_limit_gb` recorded in the config. | `guardrails.py` |
| 3 | New `estimate_commit_mb()` + a preflight check against the cap. A run that cannot fit is now refused **before** it starts, naming the fix. Model vs measurement: Good 13.5 GB predicted, Best 19.4 GB predicted against 19.5 GB measured. | `guardrails.estimate_commit_mb`, `guardrails.preflight` |
| 4 | Watchdog warns once when job commit passes 85% of the cap. | `guardrails._watchdog_loop` |
| 5 | A FaceFusion exit that looks like an allocation failure - or that happened within 3% of the cap - now raises a `PipelineError` that says so, with the peak, the cap and the three ways out. It used to surface ten lines of onnxruntime noise. | `engine.run_facefusion` |

The old config is kept at `scripts/guardrails.json.pre_memcap.bak`.

### Also fixed in the same pass

- **The Settings tab was write-only for the four settings a preset owns.**
  `run_vr`/`run_normal` re-apply the selected preset immediately before
  starting, so any hand-edit to Swapper Model, Pixel Boost, Face Enhancer or
  Frame Enhancer was overwritten a moment before the run - silently, since the
  dropdown still said "Good". Editing any of them now flips both preset
  dropdowns to **Custom**.
- **"Custom (Settings Tab)" did nothing at all.** It left the previous preset's
  models sitting in the widgets, so Custom was a copy of whatever had been
  chosen before it. Selecting it now reloads `pipeline_config.json`.
- **The two preset dropdowns drifted apart.** They drive the same settings, so
  choosing Best on the Normal tab left the VR tab reading "Good" over Best's
  settings. They are kept in step now.
- **The UI thread no longer waits on `nvidia-smi`.** The resource read-out
  polled it from the Tk timer every 3 s (47 ms idle, worse under load), and
  `ResourceGuard._report_released` blocked the UI for ~1.2 s at the end of
  every run (five calls plus four 0.25 s sleeps). Both are on background
  threads now; measured worst UI-thread stall is 65 ms.
- **A full-size temp encode leaked when temporal smoothing was on.** The
  cleanup test was `current_output != ff_output`, which is never true at that
  point, so FaceFusion's own render stayed in `output/temp`.

---

## 🧹 MEMORY RELEASE AFTER A RUN (Aug 26 2026)

Checked, measured, and one real leak found and fixed.

### What was already fine

Every model-bearing stage runs in a **child process** - FaceFusion, the CUDA
probe, ffmpeg - so when a run ends the OS reclaims everything wholesale. Three
back-to-back Normal-mode runs in one process:

```
after import engine          RSS 16.4 MB
after normal pipeline run 1  RSS 16.6 MB
after normal pipeline run 2  RSS 16.6 MB
after normal pipeline run 3  RSS 16.6 MB
```

and a full VR run, sampling once a second throughout:

```
idle                 RSS 17.7 MB   GPU in use   944 MB
during the run       RSS 18.5 MB   GPU in use  8819 MB   <- peak
after the run        RSS 18.5 MB   GPU in use   935 MB
                     delta +0.6 MB              delta -9 MB
```

8.8 GB of VRAM taken and given straight back. No drift.

### The leak: Real-ESRGAN ran inside the GUI process

It was the only stage that did, and it cost three things at once:

1. **torch never goes away.** `import torch` alone took the process from 16 MB
   to 364 MB RSS, and Python cannot unload a native extension - so the app
   stayed ~350 MB heavier for the rest of the session after a single upscale.
2. **The CUDA context is process-lifetime too**, and torch's caching allocator
   does not hand freed device blocks back to the driver without an explicit
   `empty_cache()`. So VRAM stayed reserved as well.
3. **The GUI process is not in the guard's Job Object.** This stage therefore
   ran with no memory cap, no CPU cap, and out of reach of the watchdog and the
   STOP button - the one stage most able to exhaust the card was the one stage
   the guard rails did not cover.

Fixed by moving it to **`scripts/esrgan_worker.py`**, spawned through the guard.
Everything it allocates leaves with the process, and it inherits job membership
so the caps and STOP now cover it (ffmpeg started from inside it inherits too).
Progress comes back as `ESRGAN_PROGRESS done total` lines and fatal problems as
a single `ESRGAN_ERROR` line, so the GUI shows a sentence instead of a
traceback.

### And `engine.py` no longer imports cv2 or numpy

They were there only for the in-process Real-ESRGAN loop. With that gone they
were pure weight in the GUI:

```
import engine   before   RSS 39.4 MB   commit 763.2 MB
import engine   after    RSS 16.4 MB   commit   9.9 MB
```

The commit figure matters beyond tidiness: the watchdog trips on free commit
charge (`min_free_pagefile_gb`), so 753 MB of it was being spent before the
pipeline started.

### Real-ESRGAN could never have run anyway

Three separate defects, all pre-existing, found while moving the code:

| | |
|---|---|
| `RealESRGANer(...)` was called with no `model=` | it defaults to `None`, then dies inside `model.load_state_dict()` |
| `upsampler(rgb, None)` | `RealESRGANer` has no `__call__`; the method is `.enhance()` |
| `facefusion/weights/realesr-general-x4v3.pth` | the directory does not exist |

The first two are fixed in the worker: it builds the right architecture
(`SRVGGNetCompact`, 32 conv blocks, native 4x) and calls
`.enhance(rgb, outscale=...)`, with the ffmpeg pipe sized from what the model
actually returns rather than what we assumed. The third is now caught in
**preflight**, before the swap runs, instead of after half an hour of work:

```
Output Upscaling is set to Real-ESRGAN but its weights are not installed.
Expected: ...\facefusion\weights\realesr-general-x4v3.pth
Either drop realesr-general-x4v3.pth there, or set Output Upscaling to
'ffmpeg', or use FaceFusion's own frame enhancer instead - set Frame Enhancer
to real_esrgan_x2_fp16 ...
```

Note that even with the weights present, **Real-ESRGAN cannot run on this
machine**: torch is `2.5.1+cu121`, built for sm_50-sm_90, and this card is
sm_120. The worker reports that as one clean line rather than "no kernel image
is available for execution on the device". FaceFusion's ONNX frame enhancers
(`real_esrgan_x2_fp16`, …) are unaffected - they run on the onnxruntime stack
that already works.

### Smaller things

- **The log pane grew forever.** A Tk text widget keeps every line it is ever
  given, and `log_messages` kept a parallel copy. Both are now trimmed to 1500
  lines once they pass 2000, in step so they cannot drift apart.
- **`finish_run()` now collects** after closing the guard, so the idle
  footprint is flat between runs rather than drifting until a generation
  happens to fire.
- **The guard says what it gave back.** Closing it now logs
  `[Guard] Memory released. VRAM now in use across the whole system: 0.9 GB of
  15.9 GB.` The wait for the driver to settle is bounded to about a second and
  stops as soon as the number stops falling, because `close()` runs on the UI
  thread.

---

## 🎬 SECOND QUALITY PASS (Aug 26 2026)

The first pass fixed the things that were badly wrong (CPU fallback, EXHAUSTIVE
algo search, a swapper that washed colour out). This pass went looking for what
was still being left on the table. Two of the four findings are settings
FaceFusion exposes and this pipeline never passed; the other two are the
pipeline's own encoding, which turned out to be capped in a way nobody asked
for.

### 1. `-cq` was doing nothing (the big one)

`_encoder_args()` built `-rc vbr -cq N` with no `-maxrate`. Under VBR with no
explicit rate, NVENC falls back to **its own** default maximum bitrate, and that
cap - not `-cq` - decides quality. Measured on this card, 3 s of 4K:

| encoder args | output |
|---|---|
| `-rc vbr -cq 1` | 8 346 855 bytes |
| `-rc vbr -cq 10` | 8 346 855 bytes |
| `-rc vbr -cq 20` | 8 331 541 bytes |
| `-rc vbr -cq 30` | 4 880 413 bytes |
| `-rc vbr -cq 10 -maxrate 200M` | 30 139 211 bytes, +0.15 dB PSNR |

Every value below about cq 20 collapsed onto the same operating point - and
cq 16-20 is the entire range this pipeline uses. Adding `-maxrate` (default
120 Mbit/s, `nvenc_maxrate_mbps` in `guardrails.json`) makes the knob behave.
`-b:v 0` is **not** the fix; it changes nothing, because ffmpeg already implies
it when `-cq` is set.

### 2. The final merge ignored the Quality setting

The merge is the last encode the output goes through, and it was pinned at
`crf=20` regardless of what Quality was set to - so asking for 95 bought a
better FaceFusion intermediate and then threw the difference away in the last
step. It now maps Quality through the same curve FaceFusion uses internally
(`quality_to_cq()`), runs at NVENC preset `p7` instead of `p5`, and gets the
rate cap from (1). On the same input:

```
OLD  p5 cq20, no cap    3 793 173 bytes   PSNR 48.05 dB
NEW  p7 cq3,  cap 120M  48 998 477 bytes  PSNR 57.01 dB
```

**Expect much larger output files.** At Quality 95 the merge now runs up
against the 120 Mbit/s cap, so a 1-minute 4K-per-eye VR clip lands around
900 MB rather than 70 MB. Turn Quality down, or lower `nvenc_maxrate_mbps`, if
that is not wanted - the point is that the slider now controls it.

The same fix went to the split (cq 18 -> 16, and it is the *input* to the swap,
so anything it discards is gone for good), the temporal-smoothing pass, the
Normal-mode trim, and the upscaler - all four hardcoded 18 and ignored the crf
they were handed. The upscaler was worse than that: it was passed
`crf=config['quality']`, i.e. **cq 95**, which is the worst setting the encoder
has. Quality runs 0-100 upwards, cq runs 51-0 downwards.

### 3. `.mkv` outputs were actually MP4s

The merge always wrote `merged_vr.mp4` and the pipeline then `shutil.move()`d
it onto whatever the user named the output. Choosing `.mkv` produced an **MP4
with an .mkv extension**, so the `stereo_mode` tag - which is a Matroska
element and the whole reason to pick .mkv - was silently dropped every time.
The note in the Aug 12 entry saying "choose .mkv if you need headset-readable
stereo metadata" was therefore not achievable.

Now the intermediates take the output's own extension, and `stereo_mode` is
stamped by a new `apply_stereo_metadata()` that runs **last**, as a `-c copy`
remux - because temporal smoothing and upscaling each re-encode into a fresh
file and would drop the tag again. Choosing `.mp4` logs a line saying the
container cannot carry the tag instead of pretending it did.

> **Superseded (Sep 5 2026).** `.mp4` is now the container to pick: it carries
> `st3d` + `sv3d`, which describe eye packing *and* the 180/360 projection.
> `stereo_mode` in an MKV only ever described the packing, so an MKV was never
> going to be enough on its own.

Verified: the output is now `format_name=matroska,webm` with
`TAG:stereo_mode=left_right`.

### 4. Settings FaceFusion has and this pipeline never passed

| flag | was | now | why |
|---|---|---|---|
| `--face-tracker-score` | never passed (0 = off) | `0.2` | See below. |
| `--target-frame-amount` | never passed | `2` | The tracker's window. |
| `--face-swapper-weight` | never passed (0.5) | exposed | Identity strength. 0.5 is neutral; above it the result is pushed away from the original face's embedding, sharpening resemblance to the source. |
| `--face-enhancer-weight` | never passed | exposed | codeformer's fidelity knob. Inert on gpen/gfpgan - their graphs have no `weight` input. |
| `--face-landmarker-model` | never passed | exposed, stays `2dfan4` | Measured below. |
| `--face-detector-margin` | never passed | exposed | A box that clips the jaw drags the landmarks in with it. |
| `--face-occluder-model` | never passed | exposed | xseg_1/2/3 or `many`. Decides what is "in front of" the face. |
| `--face-parser-model` | never passed | exposed | Used by the `region` mask. |
| `--output-video-preset` | never passed | `slow` | FaceFusion's default is `veryfast`, which it maps to the **lowest** NVENC quality tier. `slow` maps to the highest, on silicon the inference is not using. |
| `--workflow-strategy` | inherited from ini | pinned `memory` | `disk` writes every frame out as an image first - an extra lossy generation on top of hundreds of GB. |
| `--reference-face-distance`, `--reference-frame-number` | never passed | exposed | `reference` selector mode was selectable but had no controls. |
| `expression_restorer` processor | not wired | optional, off | LivePortrait re-applies the original expression to the swapped face. **5x** the runtime (198.7 s vs 39.5 s / 100 frames), so it is opt-in. |

### What the face tracker actually does

Do not expect it to fix landmark jitter - **it does not smooth landmarks.**
Read `face_tracker.py`: `refill_faces()` only interpolates frames that have
*no* detection. Where the detector succeeds on every frame, tracked output is
byte-identical to untracked output. Verified on two 100-frame clips, one with
synthetic head roll, jitter, noise and 4-frame motion blur - identical file
sizes, all three window settings.

What it does do is recover frames the detector drops, which otherwise come out
unswapped - the original face flashing back for a single frame, which is very
obvious in a headset:

```
100-frame clip, detector loses the face on frames 82-83
  tracker off   frames 82-83 delta 3.10  (unswapped)
  tracker 0.2   frames 82-83 delta 5.79  (swapped)
```

Cost is only the extra decode - detections are cached per frame - so 39.5 s ->
42.3 s per 100 frames, **+7%**. On at that price.

`target_frame_amount` 4 instead of 2 costs 46.0 s and bridges longer gaps; it
changed nothing on the test clips, whose gaps were single frames.

### Why the landmarker stays at 2dfan4

`many` runs 2dfan4 *and* peppa_wutz on every face and keeps the higher-scoring
set. On ordinary front-facing footage 2dfan4 wins every frame, so it costs 85%
more for nothing:

```
2dfan4   39.0 s / 100 frames
many     72.0 s / 100 frames   - byte-identical output
```

It is in the GUI for footage that actually needs it: hard profiles, heavy head
roll, faces looking down.

### `temp_frame_format` is inert

`workflow_strategy` is `memory`, and in memory mode FaceFusion never writes
frames out at all - it decodes, processes and pipes straight to the encoder.
There is no image round-trip to lose quality to, so the jpeg/png choice does
nothing. It is still honoured, and still correct, for the `disk` strategy.

### Still not fixed

**Left and right eyes are still processed independently.** The tracker helps
each eye individually but does not make them agree with each other. Processing
the whole SBS frame in one pass is not a fix at 4K per eye either: FaceFusion
resizes the frame to 640x640 for detection, so a 300 px face in a 7680-wide
frame arrives at the detector 25 px across.

---

## 🎯 FACE SELECTION + NORMAL (2D) MODE (Aug 26 2026)

### Single face, filtered by gender

`selector_mode` was `many`, which replaces **every** face in the frame. It is
now `one`, with two new config keys behind it:

| key | values | meaning |
|---|---|---|
| `selector_mode` | `one` / `many` / `reference` | `one` swaps a single face per frame |
| `selector_order` | `large-small`, `best-worst`, `left-right`, … | which face `one` picks |
| `selector_gender` | `auto` / `male` / `female` | which faces are eligible at all |

FaceFusion applies these in a specific order — sort by `selector_order`, filter
by gender, then take the first survivor — so `one` + `large-small` + `female`
means *"the largest female face in the frame"*.

`auto` (the default) copies the gender of the **source photo**, so a female
source only ever replaces female faces. Pin it to `male`/`female` when the
source is ambiguous or the classifier gets it wrong. Gender comes from the
`fairface` classifier, which also fills in age and race.

Verified on a still with one female face:

```
selector_gender=female  ->  1,361,540 pixels changed (16.4% of frame)
selector_gender=male    ->          0 pixels changed  (face excluded, no swap)
```

Exposed in Settings under a new **Face Selection** group (Mode / Which Face /
Gender).

### Normal (2D) mode

Normal mode shares `run_facefusion()` with VR, so it already picked up the CUDA
fix, the model changes and the selector work. Three things were specific to it:

1. **The Output FPS box did nothing.** `run_normal_pipeline()` took an `fps`
   argument and never referenced it — the VR path uses it for split/merge, but
   this path has neither, so the value was read from the GUI and dropped. It now
   reaches FaceFusion as `--output-video-fps`.
2. **Still images were not supported.** FaceFusion decides image-vs-video from
   the *output* extension, and the output was hardcoded to `ff_result.mp4`, so a
   photo target came back as a one-frame video. Image targets now work
   end-to-end (`IMAGE_EXTS` / `is_image_path()`), the output extension is
   corrected to match, `--output-image-quality` is sent instead of the video
   quality flag, and trim / fps / temporal smoothing / upscale are skipped with
   a log line rather than silently mis-applied.
3. It is otherwise the same pipeline minus split and merge, so it has **two
   fewer generations of lossy re-encoding** than VR mode.

### Also fixed

Renaming the quality presets left `"Good (~30 min/1min video)"` hardcoded in
both tab builders, which would have been a `KeyError` at window construction.
Both now reference a single `DEFAULT_PRESET` constant.

---

## ⚡ SPEED & QUALITY FIX (Aug 26 2026)

The pipeline was slow and the swaps were poor. Three separate causes, all fixed.

### 1. The GPU was never being used (the big one)

`onnxruntime-gpu` ships **no CUDA libraries of its own**. The venv had
`onnxruntime-gpu 1.20.2` and only `nvidia-cuda-runtime-cu12` — no cuDNN, no
cuBLAS — so `onnxruntime_providers_cuda.dll` failed to load with `error 126`
and onnxruntime **silently** fell back to the CPU provider. Every run completed
normally, just on 28 CPU threads instead of the RTX 5060 Ti.

Measured on the original settings, 4K per eye:

| | throughput |
|---|---|
| CPU (what was actually happening) | **0.22 frame/s** |
| CUDA, same settings | **1.39 frame/s** |
| CUDA + new settings | **1.95 frame/s** |

That is **~9x end-to-end**.

Fixes:
- Installed `onnxruntime-gpu==1.24.4` (the version FaceFusion 3.8.2's own
  installer pins for `cuda@12`; it has Blackwell/`sm_120` kernels, which 1.20.2
  did not) plus `nvidia-cudnn-cu12`, `nvidia-cublas-cu12`, `nvidia-cufft-cu12`,
  `nvidia-curand-cu12`.
- `facefusion/venv/Lib/site-packages/_vrdeep_cuda_dlls.{py,pth}` registers those
  DLL directories at interpreter start-up. Without it a plain
  `import onnxruntime` still lands on the CPU — onnxruntime only finds them if
  something calls `preload_dlls()`, and FaceFusion never does.
- `engine.check_gpu()` builds a real CUDA session and runs one convolution
  through it during preflight. `prepare_run()` now **refuses to start** if that
  fails, with the pip command to fix it. Listing `CUDAExecutionProvider` in
  `get_available_providers()` is not proof it works.
- `--execution-providers` no longer has `cpu` appended behind it. A silent
  fallback is worse than an error.

> PyTorch is still `2.5.1+cu121`, which has no `sm_120` kernels either — it only
> matters for the Real-ESRGAN upscale path, which now raises a clear error
> instead of dying with "no kernel image is available for execution on the
> device". FaceFusion's own ONNX frame enhancers (`real_esrgan_x2_fp16`, …) run
> on the working onnxruntime stack.

### 2. cuDNN algorithm search was exhausting VRAM

`execution.py` carried a comment saying it used `HEURISTIC`, but the code
returned `EXHAUSTIVE` for every card except the GTX 16xx series. EXHAUSTIVE
benchmarks every convolution algorithm and keeps the largest workspace any
candidate asked for. With hyperswap resident that pinned VRAM at **15.9 of
16.3 GB**, so the allocator spilled into WDDM shared host memory:

| cudnn_conv_algo_search | throughput |
|---|---|
| EXHAUSTIVE | 0.12 frame/s |
| DEFAULT | 1.54 frame/s |

It also caused intermittent `CUBLAS_STATUS_ALLOC_FAILED … cublasCreate` at
start-up, after which FaceFusion logs `loading model <name> failed` and the run
dies. Now `DEFAULT`, with `FACEFUSION_CUDNN_CONV_ALGO_SEARCH` still honoured as
an override, and a second escape hatch
(`FACEFUSION_ARENA_EXTEND_STRATEGY=kSameAsRequested`, ~20% slower) for the
allocation failures.

### 3. Settings that were costing quality

| setting | was | now | why |
|---|---|---|---|
| `swapper_model` | `simswap_unofficial_512` | `hyperswap_1a_256` | Newest swapper in FF 3.8; holds identity far better than simswap, which is soft and washes colour out. Also **faster** (1.95 vs 1.39 frame/s). |
| `pixel_boost` | `512x512` | `256x256` | Pixel boost tiles the crop, so 512 on a 256 model is a batch of 4. Native is the default; `512x512` is available as the "Better"/"Best" presets. |
| `enhancer_model` | `gpen_bfr_1024` | `gpen_bfr_512` | ~2x cheaper per face for detail a face a few hundred pixels wide cannot show. |
| `enhancer_blend` | `85` | `70` | Above ~75 GPEN's own face prior starts overriding the swapped identity — waxy skin, and flicker frame to frame that is very visible in a headset. |
| `detector_score` | `0.25` | `0.5` | Back to FaceFusion's own default. The raw detector returns 8 boxes at 0.25 vs 6 at 0.5 on the test frame, but NMS and the landmarker prune both down to the same 1 real face - so this is defensive, not a fix for observed bad swaps. |
| `detector_angles` | hardcoded `0 90 180 270` | config key, default `0` | Extra angles are extra chances at a rotated false positive. Set to `"0 90 180 270"` only for footage where the camera itself rolls. |
| `mask_padding` | `2 2 2 2` | `0 0 0 0` | Padding shrinks the mask inward per side, leaving a rim of the original face around the swap. |
| `mask_blur` | `0.4` | `0.3` | Softer edge was compensating for the padding rim. |

Measured alternatives, same 100-frame 4K clip, ~6 faces per frame:

```
simswap_512 / pb512 / gpen1024  (old)   1.39 frame/s
hyperswap   / pb256 / gpen512   (new)   1.95 frame/s
hyperswap   / pb256 / gpen1024          1.52 frame/s
hyperswap   / pb512 / gpen512           1.60 frame/s
hyperswap   / pb256 / no enhancer       3.50 frame/s
inswapper128/ pb512 / gpen512           1.03 frame/s
```

### 4. GUI model lists were wrong

The Face Detector dropdown offered `mediapipe_face_detection`, `fast` and
`yoloface` — **none of which exist in FaceFusion 3.8.2**; picking one killed the
run at launch (the real name is `yolo_face`). The size list offered `800x800`,
which no detector accepts. Swapper and enhancer lists were missing the
hyperswap, ghost_1/2, hififace, simswap_256 and gpen_bfr_512 models. All four
lists now match `facefusion/choices.py` and
`facefusion/processors/modules/*/types.py`.

### Still worth knowing

- **`selector_mode` is still `many`**, which swaps *every* face in frame. For
  single-subject footage set it to `one` or `reference`: fewer wrong swaps, and
  most of the cost is per-face so it is also much faster.
- **Left and right eyes are processed independently**, so the two swaps can
  differ slightly. In a headset that reads as a shimmering, unstable face. There
  is no fix short of processing the SBS frame as a single image, which costs
  detector resolution.
- The old `pipeline_config.json` is kept at `pipeline_config.json.bak`.

---

## 🛡️ RESOURCE GUARD RAILS (Aug 12 2026)

Fixed the bug where starting a conversion froze the whole machine and forced a
hard restart. Three resource exhaustions were responsible:

| # | Cause | Fix |
|---|-------|-----|
| 1 | FaceFusion extracted every frame to **PNG** (default) and the VR split wrote near-lossless x264 (`-crf 6 -preset slow`). A 5-min 4K VR clip needed **~104 GB** of temp on a drive with 66 GB free. | `--temp-frame-format jpeg`, split at cq 18, per-eye temp cleanup. Same clip now needs **~20 GB**. |
| 2 | `--video-memory-strategy tolerant` + `threads 8` + 1024 pixel-boost + `gpen_bfr_2048` needed **~46 GB VRAM** on a 16 GB card. Windows then paged GPU memory into system RAM and thrashed the pagefile. | Strategy `moderate`, threads clamped, safer defaults (**~9 GB**), preflight blocks overcommit. |
| 3 | `libx264 -preset slow` / `libx265 -preset medium` pinned all 28 cores at normal priority, so the shell never got scheduled. | NVENC preferred; software fallback is thread-capped; whole job runs below-normal with a 75% CPU cap. |

**New: `scripts/guardrails.py`** — tunable via `scripts/guardrails.json`.

- **Preflight** estimates temp-disk and VRAM need, and refuses (with an
  explanation and an override prompt) runs that cannot fit.
- **Job Object** caps the pipeline at ~19.7 GB committed memory and 75% CPU,
  runs it below-normal priority, and kills every child when the app closes.
- **Watchdog** polls free RAM / disk / commit charge every 3s and aborts the
  run before Windows becomes unresponsive.
- **STOP button** in the UI terminates the whole process tree mid-run.

Also fixed along the way: torch/CUDA no longer loads into the GUI at startup
(launch is now ~0.5s), Real-ESRGAN uses tiled inference instead of whole-frame
(`tile=0` blew up VRAM) and streams frames to ffmpeg instead of staging PNGs,
the FaceFusion output reader no longer busy-spins a core, slider settings reach
the CLI as ints, invalid frame-enhancer model names were corrected, and the
Matroska `stereo_mode` tag now uses valid values (`left_right`/`top_bottom`;
the old `fb` was silently dropped).

> Note: `stereo_mode` is a Matroska tag — it is **not** written for `.mp4`
> outputs.
>
> **Superseded (Sep 5 2026): prefer `.mp4`.** `stereo_mode` only ever described
> eye packing, never projection, so an MKV still played as a flat stereo pair.
> MP4 now gets Google's `st3d` + `sv3d` boxes, which describe both — see
> "The output played as flat 2D in the Quest".

---

## ✅ COMPLETED

### Phase 1 - Tool Selection
- [x] FaceFusion installed and configured as primary engine (`facefusion/venv/`)

### Phase 3 - FaceFusion Pipeline (Core Workflow)
- [x] `split_vr.py` — Split stereoscopic video (SBS/TB) into left/right eyes
- [x] `run_facefusion.py` — Face swap on each eye with configurable models/enhancers
- [x] `merge_vr.py` — Merge back into stereoscopic VR + post-processing (denoise, color correction)
- [x] `run_pipeline.py` — Full CLI orchestrator (split → swap → merge → cleanup)
- [x] `engine.py` — Python API for programmatic pipeline calls
- [x] `app.py` — Full GUI with VR and Normal pipeline modes, settings panel, progress tracking, logging

### Phase 5 - Installation & Launchers
- [x] Python + CUDA environment set up (`facefusion/venv/`)
- [x] Config system in place (`pipeline_config.json`)
- [x] Fixed PowerShell launcher (`launch_app.ps1`)
- [x] Created Windows batch file launchers (`launch_app.bat`, `launch.bat`)
- [x] Fixed path resolution bugs in `scripts/app.py`
- [x] Fixed FaceFusion `--processors` argument CLI syntax in `scripts/engine.py` and `scripts/run_facefusion.py`

### Phase 6 - VR Handling
- [x] SBS and TB layout support
- [x] VR stereo metadata injection
- [x] Trim, FPS override, temp file management

---

## ⚠️ PARTIALLY IMPLEMENTED

### Phase 4 - Quality Enhancement Pipeline
- [x] Base deepfake (FaceFusion)
- [x] Face Enhancement: CodeFormer built in
- [x] Post-processing: denoise, color correction in `merge_vr.py`
- [ ] Real-ESRGAN video upscaling — not wired in
- [x] RIFE frame interpolation — built Sep 5 2026 (scripts/rife_worker.py)
- [ ] Noise matching — not implemented

---

## ❌ NOT DONE

### From the Plan
- [ ] DeepFaceLab integration (Option B, never started)
- [ ] Benchmarking / quality vs speed testing
- [ ] Scaling to full production (no test runs logged)

### Missing Features
- [ ] Video upscaling step (Real-ESRGAN) - code path is now correct and runs in
      its own process, but it needs `facefusion/weights/realesr-general-x4v3.pth`
      (not present) **and** a PyTorch build with sm_120 kernels (installed one is
      cu121, sm_50-sm_90). FaceFusion's ONNX frame enhancers work today and need
      neither.
- [x] Frame interpolation (RIFE) - built Sep 5 2026, see below
- [ ] Equirectangular support (plan mentions it, code only handles SBS/TB)
- [ ] Batch processing multiple videos
- [ ] Output validation / quality metrics

---

## SUMMARY

**Core pipeline: ~80% complete and functional.**

Remaining work is quality enhancements (upscaling, interpolation) and production hardening.
