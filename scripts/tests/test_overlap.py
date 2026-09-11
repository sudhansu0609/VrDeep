"""
B9: the split and the merge run underneath the face swap.

The face swap is the only part of a chunk that needs the GPU, and it is 85% of
the chunk's wall time - so it is the part this test replaces with a sleep. The
splits, the merges, the joins, the per-chunk temp directories, the background
executors, the handoff and the frame-count check are all the real thing,
running real ffmpeg over a real (tiny) side-by-side clip.

That is deliberate: what B9 changed is not the swap, it is everything around
it, and the failure mode it has to be proved against - a merge reading an eye
that the next chunk is still writing - lives entirely in that machinery.

Run:  python scripts/tests/test_overlap.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine
import guardrails

# chunk_plan folds a tail shorter than CHUNK_MIN_TAIL (10 s) into the
# previous chunk, so a three-chunk run needs a clip of at least this shape.
CHUNK_SECONDS = 15.0
CLIP_SECONDS = 45.0
SWAP_SECONDS = 1.2      # long enough for a background split or merge to land


def have_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=30,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def make_sbs_clip(path, seconds=CLIP_SECONDS):
    """A 640x240 side-by-side clip: two different 320x240 halves, 30 fps."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=30:duration={seconds}",
        "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=30:duration={seconds}",
        "-filter_complex", "[0:v][1:v]hstack=2[v]", "-map", "[v]",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        path,
    ]
    subprocess.run(cmd, check=True, capture_output=True,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


class _Timeline:
    """When each stage of each chunk started and finished."""

    def __init__(self):
        self.events = []
        self.start = time.time()

    def mark(self, what, index):
        self.events.append((what, index, time.time() - self.start))

    def at(self, what, index):
        for name, i, when in self.events:
            if name == what and i == index:
                return when
        raise AssertionError(f"{what} of chunk {index} never happened: {self.events}")

    def happened(self, what, index):
        return any(name == what and i == index for name, i, _ in self.events)


class OverlapTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not have_ffmpeg():
            raise unittest.SkipTest("ffmpeg is not on PATH")
        cls.work = tempfile.mkdtemp(prefix="vrdeep-overlap-")
        cls.clip = os.path.join(cls.work, "sbs.mp4")
        make_sbs_clip(cls.clip)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def setUp(self):
        self.real_inner = engine.run_vr_pipeline
        self.real_split = engine.split_video
        self.real_finish = engine.finish_deferred_merge
        self.timeline = _Timeline()
        self.merge_dirs = []

        # run_chunked_pipeline derives its scratch directory from PROJECT_DIR,
        # so the test gets its own PROJECT_DIR and never goes near the real
        # output/temp. This is not tidiness: an earlier version of this file
        # shared output/temp with the app, and running it while a real job was
        # mid-split deleted that job's finished left eye. A test must not be
        # able to destroy hours of somebody's render.
        self.real_project_dir = engine.PROJECT_DIR
        self.sandbox = tempfile.mkdtemp(prefix="vrdeep-sandbox-", dir=self.work)
        engine.PROJECT_DIR = self.sandbox
        self.temp_root = os.path.join(self.sandbox, "output", "temp")
        os.makedirs(self.temp_root, exist_ok=True)

    def tearDown(self):
        engine.run_vr_pipeline = self.real_inner
        engine.split_video = self.real_split
        engine.finish_deferred_merge = self.real_finish
        engine.PROJECT_DIR = self.real_project_dir
        shutil.rmtree(self.sandbox, ignore_errors=True)

    # -- the stand-in for the face swap ----------------------------------

    def _fake_pipeline(self, timeline):
        """
        run_vr_pipeline with the two FaceFusion passes replaced by a sleep.

        Everything else it does is what the real one does: it splits when it
        has not been handed a split, it writes the same checkpoints, and it
        either merges in line or hands the eyes back exactly as `defer_merge`
        asks.
        """
        def fake(source_face, target_video, output_path, config, layout="sbs", fps=None,
                 trim_start=None, trim_end=None, merge_encoder=None, merge_crf=None,
                 merge_preset=None, keep_temp=False, progress_cb=None, log_cb=None,
                 guard=None, force=False, prepared=None, refresh_source_faces=False,
                 resume=True, temp_dir=None, presplit=None, defer_merge=False,
                 on_split_done=None):
            # chunk_file() names them part_001.mp4 upwards; chunk indices are 0-based.
            index = int(os.path.basename(output_path).split("_")[-1].split(".")[0]) - 1
            temp_dir = temp_dir or os.path.join(engine.PROJECT_DIR, "output", "temp")
            os.makedirs(temp_dir, exist_ok=True)
            guard_config = (prepared or (None, {}, None))[1] or {}
            merge_crf = 20 if merge_crf is None else merge_crf

            if presplit:
                left, right = presplit
                timeline.mark("presplit-used", index)
            else:
                timeline.mark("split-inline-start", index)
                left, right = engine.split_video(
                    target_video, temp_dir, layout, fps, trim_start, trim_end,
                    log_cb=None, guard=guard, encoder="libx264",
                    guard_config=guard_config)
                timeline.mark("split-inline-end", index)

            # The real pipeline tells the caller here, and the caller uses it
            # to start the next chunk's split. Getting this wrong is what made
            # two splits run at once on the first chunk.
            if on_split_done:
                on_split_done()

            timeline.mark("swap-start", index)
            left_out = os.path.join(temp_dir, "left_swapped.mp4")
            right_out = os.path.join(temp_dir, "right_swapped.mp4")
            time.sleep(SWAP_SECONDS)
            # A "swap" that keeps every frame, so the frame-count check has
            # something true to assert.
            shutil.copyfile(left, left_out)
            shutil.copyfile(right, right_out)
            timeline.mark("swap-end", index)

            state = engine.RunState(temp_dir)
            keys, parts = engine.vr_stage_keys(
                engine.split_sources(source_face), target_video, config, layout, fps,
                trim_start, trim_end, merge_encoder, merge_crf, merge_preset)
            for eye, path in (("left", left_out), ("right", right_out)):
                state.complete("swap_" + eye, path, keys["swap_" + eye],
                               parts["swap_" + eye])
            handoff = {
                "left": left_out, "right": right_out,
                "temp_dir": temp_dir, "output_path": output_path,
                "layout": layout, "fps": fps,
                "merge_encoder": merge_encoder or "libx264", "merge_crf": merge_crf,
                "merge_preset": merge_preset, "keep_temp": keep_temp,
                "state": state, "merge_key": keys["merge"],
                "merge_parts": parts["merge"], "guard_config": guard_config,
            }
            if defer_merge:
                return handoff
            timeline.mark("merge-start", index)
            engine.finish_deferred_merge(handoff, guard=guard, log_cb=None)
            timeline.mark("merge-end", index)
            return output_path
        return fake

    def _watch_background_stages(self, timeline):
        """
        Timestamp the split and the merge that run on the background threads.

        Wrapped around the real functions rather than inferred from the log,
        because what has to be proved is an ordering in time - that chunk 2's
        split *finished* while chunk 1 still had the GPU - and a log line only
        proves it happened.
        """
        real_split, real_finish = self.real_split, self.real_finish

        def split(*args, **kwargs):
            lane = kwargs.get("lane")
            if lane != "split":
                return real_split(*args, **kwargs)
            # Which chunk this is, from the scratch directory it writes into.
            index = int(os.path.basename(args[1]).split("_")[-1])
            timeline.mark("split-bg-start", index)
            try:
                return real_split(*args, **kwargs)
            finally:
                timeline.mark("split-bg-end", index)

        def finish(handoff, **kwargs):
            index = int(os.path.basename(
                handoff["output_path"]).split("_")[-1].split(".")[0]) - 1
            self.merge_dirs.append(handoff["temp_dir"])
            timeline.mark("merge-bg-start", index)
            try:
                return real_finish(handoff, **kwargs)
            finally:
                timeline.mark("merge-bg-end", index)

        engine.split_video = split
        engine.finish_deferred_merge = finish

    def _run(self, overlap, timeline):
        engine.run_vr_pipeline = self._fake_pipeline(timeline)
        self._watch_background_stages(timeline)
        out_dir = tempfile.mkdtemp(prefix="vrdeep-out-", dir=self.work)
        output = os.path.join(out_dir, "joined.mp4")
        config = dict(engine.load_config(), chunk_seconds=CHUNK_SECONDS,
                      # No CUDA probe, no NVENC: this test is about the
                      # plumbing and has to run on any machine.
                      provider="cpu", encoder="libx264", threads=1,
                      temporal_smoothing=False, upscale_method=None,
                      interpolate_multi=1)
        guard_config = dict(guardrails.load_guard_config())
        guard_config["overlap_stages"] = overlap
        guard_config["watchdog_enabled"] = False
        estimate = guardrails.preflight(
            self.clip, self.temp_root, config, guard_config, is_vr=True,
            overlap=overlap)[2]
        prepared = (config, guard_config, estimate)
        lines = []
        result = engine.run_chunked_pipeline(
            True, "unused.png", self.clip, output, config,
            keep_temp=False, resume=False, log_cb=lines.append,
            prepared=prepared, layout="sbs", merge_encoder="libx264", merge_crf=20)
        return result, lines

    # -- the tests -------------------------------------------------------

    def test_overlap_hides_the_split_and_the_merge(self):
        overlapped_out, lines = self._run(True, self.timeline)
        log = "\n".join(lines)

        self.assertTrue(os.path.isfile(overlapped_out), log)

        # Chunk 1 splits in line - nothing has run yet to hide it under.
        self.assertTrue(self.timeline.happened("split-inline-start", 0), log)
        # Every chunk after that was handed a split that had already been done.
        for index in (1, 2):
            self.assertTrue(self.timeline.happened("presplit-used", index),
                            f"chunk {index + 1} split in line\n{log}")

        # The log says so in the words the plan asked for.
        self.assertIn("split of chunk 2 finished under chunk 1's swap", log)
        self.assertIn("merge of chunk 1 finished under chunk 2's swap", log)

        # And it really was underneath, in wall-clock terms: chunk 2's split
        # both started and FINISHED while chunk 1 still had the GPU, and chunk
        # 1's merge finished while chunk 2 did.
        # Chunk 2's split starts only once chunk 1's own split is FINISHED.
        # Before this was a callback the prefetch was submitted ahead of the
        # inner pipeline, so on the first chunk the two splits ran at the same
        # time and queued behind the card's single encoder.
        self.assertGreaterEqual(self.timeline.at("split-bg-start", 1),
                                self.timeline.at("split-inline-end", 0),
                                "chunk 2's split started while chunk 1 was "
                                "still splitting - they share one encoder")
        self.assertLess(self.timeline.at("split-bg-start", 1),
                        self.timeline.at("swap-end", 0),
                        "chunk 2's split did not start under chunk 1's swap")
        self.assertLess(self.timeline.at("split-bg-end", 1),
                        self.timeline.at("swap-end", 0),
                        "chunk 2's split did not finish under chunk 1's swap")
        self.assertLess(self.timeline.at("merge-bg-start", 0),
                        self.timeline.at("swap-end", 1),
                        "chunk 1's merge did not start under chunk 2's swap")
        self.assertLess(self.timeline.at("merge-bg-end", 0),
                        self.timeline.at("swap-end", 1),
                        "chunk 1's merge did not finish under chunk 2's swap")
        # Nothing is left in flight when the join happens: the last chunk's
        # merge is waited for, or the joined file would be built from a file
        # still being written.
        self.assertTrue(self.timeline.happened("merge-bg-end", 2), log)

        # Sequential run of the same clip, for the frame count.
        sequential_timeline = _Timeline()
        sequential_out, seq_lines = self._run(False, sequential_timeline)
        self.assertTrue(os.path.isfile(sequential_out), "\n".join(seq_lines))
        # With overlap off nothing is prefetched and every chunk splits in line.
        self.assertFalse(sequential_timeline.happened("presplit-used", 1),
                         "\n".join(seq_lines))
        self.assertNotIn("finished under chunk", "\n".join(seq_lines))

        overlapped_frames = engine.video_frame_count(overlapped_out, count_packets=True)
        sequential_frames = engine.video_frame_count(sequential_out, count_packets=True)
        self.assertGreater(sequential_frames, 0)
        self.assertEqual(overlapped_frames, sequential_frames,
                         "overlapping the stages must not change a single frame")
        self.assertAlmostEqual(engine.media_duration(overlapped_out),
                               engine.media_duration(sequential_out), delta=0.2)

    def test_each_chunk_gets_its_own_temp_directory(self):
        self._run(True, self.timeline)
        seen = self.merge_dirs
        self.assertEqual(len(seen), 3, seen)
        self.assertEqual(len(set(seen)), 3,
                         f"two chunks shared a scratch directory: {seen}")
        for path in seen:
            self.assertTrue(os.path.basename(path).startswith("chunk_"), path)
            # Swept once the chunk is finished.
            self.assertFalse(os.path.isdir(path), f"{path} was left behind")


class FrameCountCheckTest(unittest.TestCase):
    """The check that would catch a merge reading a half-written eye."""

    @classmethod
    def setUpClass(cls):
        if not have_ffmpeg():
            raise unittest.SkipTest("ffmpeg is not on PATH")
        cls.work = tempfile.mkdtemp(prefix="vrdeep-frames-")
        cls.long_eye = os.path.join(cls.work, "left.mp4")
        cls.short_eye = os.path.join(cls.work, "right_short.mp4")
        for path, seconds in ((cls.long_eye, 2.0), (cls.short_eye, 1.0)):
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=30:duration={seconds}",
                 "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", path],
                check=True, capture_output=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def test_equal_eyes_pass(self):
        merged = os.path.join(self.work, "merged_ok.mp4")
        engine.merge_videos(self.long_eye, self.long_eye, merged, "sbs",
                            encoder="libx264", crf=23)
        self.assertTrue(engine.check_merge_frame_counts(
            self.long_eye, self.long_eye, merged))

    def test_a_short_eye_is_caught(self):
        # hstack stops at the shorter input, so this produces a file that plays
        # perfectly and is half the length it should be. Duration alone would
        # not have told anyone which eye was wrong.
        merged = os.path.join(self.work, "merged_short.mp4")
        engine.merge_videos(self.long_eye, self.short_eye, merged, "sbs",
                            encoder="libx264", crf=23)
        with self.assertRaises(engine.PipelineError) as caught:
            engine.check_merge_frame_counts(self.long_eye, self.short_eye, merged)
        self.assertIn("does not hold the frames the eyes did", str(caught.exception))


class EyeShapeCheckTest(unittest.TestCase):
    """
    The check that catches a file that is not this clip's eye at all.

    Sep 10 2026: an older version of this file shared output/temp with the
    app and left its 15 s 320x240 stand-in as chunk 1's right eye. The merge
    failed ninety minutes later with an NVENC-looking error. These are the
    two checks that now name that file at once: before any merge, and when a
    checkpoint offers it for reuse.
    """

    @classmethod
    def setUpClass(cls):
        if not have_ffmpeg():
            raise unittest.SkipTest("ffmpeg is not on PATH")
        cls.work = tempfile.mkdtemp(prefix="vrdeep-eyes-")
        cls.eye = os.path.join(cls.work, "eye.mp4")            # 320x240, 4 s
        cls.other_eye = os.path.join(cls.work, "other.mp4")    # 320x240, 4 s
        cls.small_eye = os.path.join(cls.work, "small.mp4")    # 160x120, 4 s
        cls.short_eye = os.path.join(cls.work, "short.mp4")    # 320x240, 1 s
        cls.sbs = os.path.join(cls.work, "sbs.mp4")            # 640x240, 4 s
        for path, size, seconds in ((cls.eye, "320x240", 4.0), (cls.other_eye, "320x240", 4.0),
                                    (cls.small_eye, "160x120", 4.0), (cls.short_eye, "320x240", 1.0)):
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", f"testsrc=size={size}:rate=30:duration={seconds}",
                 "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", path],
                check=True, capture_output=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        make_sbs_clip(cls.sbs, seconds=4.0)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def test_matching_eyes_pass(self):
        self.assertTrue(engine.check_eyes_match(self.eye, self.other_eye))
        self.assertTrue(engine.check_eyes_match(self.eye, self.other_eye, expected_seconds=4.0))

    def test_a_different_shape_is_named(self):
        with self.assertRaises(engine.PipelineError) as caught:
            engine.check_eyes_match(self.eye, self.small_eye)
        message = str(caught.exception)
        self.assertIn("not the same shape", message)
        self.assertIn("320x240", message)
        self.assertIn("160x120", message)
        self.assertIn(self.small_eye, message)

    def test_a_short_eye_is_named(self):
        with self.assertRaises(engine.PipelineError) as caught:
            engine.check_eyes_match(self.eye, self.short_eye)
        self.assertIn("not the same length", str(caught.exception))
        # And against the range the chunk was meant to cover.
        with self.assertRaises(engine.PipelineError) as caught:
            engine.check_eyes_match(self.short_eye, self.short_eye, expected_seconds=4.0)
        self.assertIn("holds 1.0s where this range is 4.0s", str(caught.exception))

    def test_expected_geometry_comes_from_the_source(self):
        self.assertEqual(engine.expected_eye_geometry(self.sbs, "sbs")[:2], (320, 240))
        self.assertEqual(engine.expected_eye_geometry(self.sbs, "tb")[:2], (640, 120))
        self.assertAlmostEqual(engine.expected_eye_geometry(self.sbs, "sbs")[2], 4.0, delta=0.1)
        self.assertAlmostEqual(engine.expected_eye_geometry(self.sbs, "sbs", 0.5, 1.5)[2],
                               1.0, delta=0.01)

    def test_a_reused_eye_that_is_not_this_clips_is_refused(self):
        # The exact shape of the Sep 10 incident: the checkpoint is valid, the
        # file is a different clip's eye.
        self.assertIsNone(engine.describe_wrong_eye(self.eye, (320, 240), 4.0))
        self.assertIn("160x120 where this clip's eye is 320x240",
                      engine.describe_wrong_eye(self.small_eye, (320, 240), 4.0))
        self.assertIn("holds 1.0s where this range is 4.0s",
                      engine.describe_wrong_eye(self.short_eye, (320, 240), 4.0))
        # A file that is not there is the checkpoint's business, not this one's.
        self.assertIsNone(engine.describe_wrong_eye(
            os.path.join(self.work, "missing.mp4"), (320, 240), 4.0))
        # With the frame enhancer on a swapped eye may be an exact multiple.
        self.assertIsNone(engine.describe_wrong_eye(self.eye, (160, 120), 4.0, allow_scaled=True))
        self.assertIsNotNone(engine.describe_wrong_eye(self.eye, (160, 120), 4.0))

        have = {"split_left": True, "split_right": True, "swap_left": False, "swap_right": True}
        paths = {"split_left": self.eye, "split_right": self.small_eye,
                 "swap_left": self.small_eye, "swap_right": self.short_eye}
        wrong = engine._wrong_reused_eyes(have, paths, self.sbs, "sbs", None, None, {})
        self.assertEqual(sorted(wrong), ["split_right", "swap_right"], wrong)


if __name__ == "__main__":
    unittest.main(verbosity=2)
