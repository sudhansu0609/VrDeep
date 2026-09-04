# Plan: upscaling VR output to 8K and 16K

**Written**: Wed Sep 2 2026
**Status**: plan only — nothing here is implemented yet.
**Measured on**: RTX 5060 Ti 16 GB, driver 595.97, ffmpeg 8.1-full (gyan), this repo's venv.

Every number below was measured on this machine during the writing of this
document, not looked up. Where something could not be measured it says so.

---

## 0. What "8K" and "16K" mean here

VR resolutions get quoted per-eye and total interchangeably, and the difference
decides whether a thing is possible at all. This document always means:

| name | total frame (SBS) | per eye | megapixels |
|---|---|---|---|
| 4K VR (what the pipeline does today) | 3840 x 1920 | 1920 x 1920 | 7.4 |
| **8K VR** | **7680 x 3840** | 3840 x 3840 | 29.5 |
| **16K VR** | **15360 x 7680** | 7680 x 7680 | 118 |

16K is **4x the pixels of 8K and 16x the pixels of 4K.** Almost every awkward
result below follows from that one number.

---

## 0.5. Found while measuring this: the frame enhancer's upscale is discarded

This is a live defect, not a planning note, and it changes the shape of the
plan — so it goes first.

`facefusion/workflows/to_video.py` sizes the output encoder from the **target
video's** resolution, scaled by `--output-video-scale` (default `1.0`):

```python
output_video_resolution = scale_resolution(
    detect_video_resolution(state_manager.get_item('target_path')),
    state_manager.get_item('output_video_scale'))
```

`engine.run_facefusion` never passes `--output-video-scale`. So when a frame
enhancer is selected, FaceFusion runs the full tiled super-resolution, produces
a frame twice (or four or eight times) the size — and then the encoder, set up
for the target's size, scales it straight back down.

Measured, 426x226 input, `real_esrgan_x2_fp16`, blend 100:

| flags | output |
|---|---|
| frame enhancer only (what the pipeline does today) | **426 x 226** — the 2x is computed and thrown away |
| frame enhancer + `--output-video-scale 2.0` | **852 x 452** |
| `--output-video-scale 2.0`, no frame enhancer | **852 x 452** (plain resize) |

Two consequences:

1. **PROGRESS.md's recommended workaround does not do what it says.** It tells
   the reader to "use FaceFusion's own frame enhancer instead - set Frame
   Enhancer to `real_esrgan_x2_fp16`" as the working substitute for the broken
   torch Real-ESRGAN path. As wired, that pays the entire super-resolution cost
   — 0.13 frame/s at 1080p — and returns a frame the same size it started.
   It is not worthless (super-sampling then downscaling is a mild sharpen) but
   it is not the upscale the note implies, and nobody would knowingly buy it at
   that price.
2. **`--output-video-scale` is the whole 8K feature, already built.** Its range
   is 0.25-8.0 in 0.25 steps. On a 1920x1920 eye, `2.0` gives 3840x3840 (8K SBS
   after the merge) and `4.0` gives 7680x7680 (16K SBS) — both under the NVENC
   ceiling, both produced by the swap pass itself with no separate upscale
   stage, no second encode and no extra models.

Exposing that one flag is a much smaller change than anything else in this
document, and it is Stage 1 below.

---

## 1. The hard limits, measured

These are not tuning knobs. They are walls.

| limit | measured result |
|---|---|
| `hevc_nvenc` max frame | **8192 x 8192.** 8192x4096 and 8192x8192 encode fine; 10240x5120 and 15360x7680 both fail with `No capable devices found`. |
| `av1_nvenc` max frame | **Same 8192 ceiling.** 8192x4096 OK, 15360x7680 refused. |
| NVDEC (hardware decode) max width | **8192.** Decoding a 15360-wide HEVC file with `-hwaccel cuda`: `Video width 15360 not within range from 144 to 8192`, then it falls back to software. |
| `libx265` at 15360x7680 | Encodes, and dav1d/libhevc decode it in software. But it stamps **level 210 (= level 7.0)**, and HEVC's highest *defined* level is 6.2. The file is out of spec. |
| `libsvtav1` at 15360x7680 | Prints `8K and higher resolution support is currently a work-in-progress project ... should not be used for benchmarking until fully implemented`, silently forces the preset down to M9, runs at ~0.9 frame/s, and **produced a 0-byte file on one of two attempts.** Not usable. |
| PyTorch Real-ESRGAN | **Cannot run at all.** torch is 2.5.1+cu121, built for sm_50-sm_90; this card is sm_120. The weights (`realesr-general-x4v3.pth`) are also not installed. Already documented in PROGRESS.md. |

### The one good structural consequence

The pipeline already splits VR into two eyes and processes them separately.
That split *is* the tiling this problem needs, and it holds all the way up:

```
8K SBS  per eye   3840x3840   hevc_nvenc: OK
16K SBS per eye   7680x7680   hevc_nvenc: OK     <- still under the 8192 ceiling
8K SBS  merged    7680x3840   hevc_nvenc: OK
16K SBS merged   15360x7680   hevc_nvenc: REFUSED
```

**Every stage of a 16K run fits in hardware except the final merge.** The
architecture does not need to change to reach either resolution. One stage does.

---

## 2. Speed, measured

50 frames of real 4K SBS footage (`target-2160p.mp4` scaled to 3840x1920),
upscaled and encoded:

| path | rate | 60 s @ 30 fps (1800 frames) |
|---|---|---|
| lanczos -> 8K, `hevc_nvenc` p7 | **9.08 frame/s** | 3.3 min |
| lanczos -> 16K, `libx265` ultrafast | **4.62 frame/s** | 6.5 min |
| lanczos -> 16K, `libx265` medium | **1.14 frame/s** | 26 min |

So the *resize and encode* is cheap even at 16K. It is not the problem.

### Neural super-resolution is the problem

FaceFusion's ONNX frame enhancers are the only super-resolution that runs on
this machine at all (they use the onnxruntime CUDA stack that already works,
unlike the torch path). Measured, `real_esrgan_x2_fp16`, 3 execution threads,
both at the same settings:

| input | per-frame | throughput |
|---|---|---|
| 1920x1080 (12 frames) | 7.45 s/frame steady state | **0.099 frame/s** |
| 3840x2160 (4 frames) | 40.7 s/frame | **0.027 frame/s** |

4K is 4x the pixels of 1080p and cost 3.7x the time, so this scales with **input
pixel count**, as expected: the model is tiled at 256px with 8px overlap, so a
1920x1080 frame is ~40 tiles and a 3840x2160 frame is ~144.

Against 9.08 frame/s for lanczos, that is **340x slower** at 4K input. What it
means for a real run:

```
60 s clip @ 30 fps, 4K per eye, real_esrgan_x2_fp16
  = 1800 frames at 0.027 frame/s
  = ~18.5 hours per eye, ~37 hours for both
```

**Conclusion: neural upscaling to 8K is not a feature this machine has.** It is
something to leave running overnight on a clip of a few seconds. Plan for
lanczos, and treat neural SR as an opt-in with an honest time estimate in front
of it.

---

## 3. The thing worth deciding before any of it

Upscaling the merged output is the worst possible place to spend the work.

A face in VR footage is a few hundred pixels across. Enlarging the whole frame
from 7.4 to 118 megapixels to recover detail in a 400px face spends 99% of the
compute on background that has no detail to recover — the sky does not get
sharper, it gets bigger.

Three places to spend it, cheapest first:

**(a) Make the face better, not the frame bigger.** `pixel_boost: 512x512` and
`enhancer_model: gpen_bfr_1024` (the "Best" preset) already work on the face
crop only. `gpen_bfr_2048` costs ~2x *per face* — not ~16x per frame. This is
already available today and is almost certainly the better buy.

**(b) Upscale per eye, before the merge.** Half the pixels per pass, stays under
the NVENC ceiling, and it removes an entire generation of lossy re-encode
compared with today's merge-then-upscale order.

**(c) Enlarge the container last,** and only because the headset wants a
particular frame size.

The plan below is built around (b) and (c). It is worth trying (a) first and
seeing whether 8K is wanted at all.

---

## 4. The plan

### Stage 0 — stop the existing path failing silently at these sizes

This is the smallest piece and should go first, because the GUI already lets
you ask for it and the failure is unreadable.

`upscale_video_ffmpeg` passes whatever `upscale_target_width` says straight to
`config["encoder"]`, which is `hevc_nvenc`. Set Target Width to 15360 today and
the run dies with:

```
[hevc_nvenc] No capable devices found
Error while opening encoder - maybe incorrect parameters such as bit_rate, rate, width or height
```

which names neither the real cause nor the fix.

- Add an `ENCODER_MAX_DIMENSION` table (`hevc_nvenc`/`h264_nvenc`/`av1_nvenc`:
  8192; software: no practical limit).
- Check the *computed output size* against it in `prepare_run`, not at the
  upscale stage — that is half an hour into a run.
- When the target exceeds the ceiling, either fall back to `libx265` with a
  logged line saying why, or refuse with the three ways out. Prefer refusing:
  a silent switch from a 9 frame/s hardware encoder to a 1.14 frame/s software
  one is the kind of surprise this codebase has already been bitten by.
- Warn whenever the output will be wider than 8192 that **it will not
  hardware-decode**, on this machine or in a headset.

Also: `guardrails.estimate_disk_need` assumes the output is about the size of
the input. At 8K it is 4x and at 16K 16x, so the preflight currently
under-estimates exactly the runs most able to fill the drive.

### Stage 1 — expose `--output-video-scale` (the actual feature; small)

This is the change section 0.5 points at, and it is most of the job.

Add one config key, `output_video_scale` (float, 0.25-8.0 in 0.25 steps), and
pass it through in `run_facefusion`:

```python
cmd += ["--output-video-scale", str(config.get("output_video_scale", 1.0))]
```

Because VR mode runs FaceFusion **per eye**, the scale applies per eye, and the
merge produces the target automatically:

| source per eye | scale | eye out | merged SBS |
|---|---|---|---|
| 1920x1920 | 1.0 | 1920x1920 | 3840x1920 (4K, today) |
| 1920x1920 | **2.0** | 3840x3840 | **7680x3840 (8K)** |
| 1920x1920 | **4.0** | 7680x7680 | **15360x7680 (16K)** |

Every intermediate stays under the 8192 NVENC ceiling. There is **no separate
upscale stage, no second encode and no extra model** — the swap pass emits the
final size directly, which is one fewer generation of lossy re-encode than
today's merge-then-upscale order.

With `frame_enhancer` left at `none` this is a plain resize inside FaceFusion
and costs almost nothing. With a frame enhancer set it is the neural path, at
the price measured in section 2.

Also in this stage:

- **Add a GUI control that sets both together.** The two settings are only
  meaningful in combination: a scale of 2.0 with no frame enhancer is a resize,
  and an x2 frame enhancer with a scale of 1.0 is the discarded-work bug from
  section 0.5. A single "Output Resolution" dropdown (`source` / `8K` / `16K`)
  that writes both, and refuses the combinations that make no sense, is safer
  than two independent knobs.
- **Fix the guard's estimates.** `estimate_disk_need` assumes the output is
  about the size of the input; at scale 2.0 it is 4x and at 4.0 it is 16x. So
  the preflight currently under-estimates exactly the runs most able to fill
  the drive.
- **Raise `nvenc_maxrate_mbps`.** It is 120 Mbit/s in `guardrails.json`, and
  PROGRESS.md records that a 4K VR minute already runs into that cap. An 8K
  frame at the same cap is being thrown away — the bitrate has to scale with
  the pixel count or the extra resolution buys nothing.

### Stage 2 — the lanczos fallback for the merged path

Stage 1 covers the per-eye case, which is VR. Normal (2D) mode and any
"upscale something that already exists" use has to go through
`upscale_video_ffmpeg`, which is the path Stage 0 makes safe. Once the encoder
ceiling is handled there, this stage is just wiring the same
`output_resolution` choice to it, at the measured 9.08 frame/s.

Worth keeping separate from Stage 1 because it is the only route that can
produce an output wider than 8192 in one file, and therefore the only one that
has to deal with software encoding.

### Stage 3 — 16K (mastering only, and probably two files)

Note that Stage 1 already gets each *eye* to 7680x7680 in hardware. It is only
the merged 15360-wide file that has no hardware encode, no hardware decode, and
no conformant container. Given that, the recommendation is:

**Do not merge.** Write `<name>_left.mkv` and `<name>_right.mkv` at 7680x7680
each. Both are under every ceiling, both hardware-encode on NVENC, both
hardware-decode. This costs nothing in quality and loses only the convenience of
one file.

If a single merged 16K file is genuinely required:

- `libx265` at ultrafast, 4.62 frame/s — the only thing measured that reliably
  produced a valid file.
- It will carry an out-of-spec level. Note it in the log, do not pretend
  otherwise.
- `stereo_mode` tagging still works (it is a Matroska element, independent of
  the codec).
- Do not use SVT-AV1 until its own 8K+ warning goes away.

Treat the result as a master to downscale from, not something to put in a
headset. No consumer headset decodes 16K, and whether a given headset manages 8K
depends on the device — worth confirming against the actual target hardware
before spending a run on it.

---

## 5. Time and size budget, 60 s clip at 30 fps

| output | how | rate | added wall clock | hardware decode? |
|---|---|---|---|---|
| 4K SBS (today) | — | — | — | yes |
| 8K SBS | `output_video_scale 2.0`, no frame enhancer | resize inside the swap pass | **~0** (no extra stage) | yes |
| 8K SBS | lanczos as a separate merged stage | 9.08 frame/s | 3.3 min | yes |
| 8K SBS | `output_video_scale 2.0` + `real_esrgan_x2_fp16` | 0.027 frame/s | **~18.5 h per eye** | yes |
| 16K, two per-eye files | `output_video_scale 4.0`, no merge | resize inside the swap pass | ~0 | yes |
| 16K SBS merged | x265 ultrafast | 4.62 frame/s | 6.5 min | **no** |
| 16K SBS merged | x265 medium | 1.14 frame/s | 26 min | **no** |

Sizes are governed by `nvenc_maxrate_mbps`, not by the resolution, until that
cap is raised — which is the point made in Stage 1.

---

## 6. Suggested order

1. **Stage 0** (half a day). Stops a wrong Target Width producing an unreadable
   failure and fixes the disk estimate. Worth doing whether or not 8K happens.
2. **Fix section 0.5 first, as a bug** — either pass `--output-video-scale`, or
   correct PROGRESS.md and the GUI so nobody buys 0.027 frame/s of
   super-resolution and gets a same-size frame for it. This is not optional
   work; it is currently wrong.
3. **Try (a) from section 3** — `gpen_bfr_2048` on the existing 4K pipeline —
   and look at the result in a headset before building anything else. Given
   that the face is a few hundred pixels wide, it may settle the question.
4. **Stage 1** (about a day). This is real 8K *and* real 16K per eye, for one
   flag plus a dropdown, and it removes a re-encode generation from the 4K path
   as a side effect.
5. **Stage 3's two-file variant** (half a day on top of Stage 1) if 16K is
   wanted; it is mostly a naming change and skipping the merge.
6. **Stage 2** (the merged 16K software encode) last, and only if a single file
   is genuinely required.

The neural super-resolution option is deliberately not on this list as anything
but an opt-in. At 18.5 hours per eye per minute of footage it is not a pipeline
stage, it is an experiment.
