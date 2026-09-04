# VR Deepfake Pipeline - Implementation Progress

**Last Updated**: Thu Sep 3 2026

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
> outputs. Choose `.mkv` if you need headset-readable stereo metadata.

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
- [ ] RIFE frame interpolation — not implemented
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
- [ ] Frame interpolation (RIFE)
- [ ] Equirectangular support (plan mentions it, code only handles SBS/TB)
- [ ] Batch processing multiple videos
- [ ] Output validation / quality metrics

---

## SUMMARY

**Core pipeline: ~80% complete and functional.**

Remaining work is quality enhancements (upscaling, interpolation) and production hardening.
