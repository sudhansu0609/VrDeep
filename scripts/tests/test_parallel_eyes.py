"""
B10: the both-eyes-at-once gate, and the watchdog policy that backs it.

Three things are checked here and none of them needs a GPU or FaceFusion:

  1. parallel_eyes_verdict() against fixed machine readings. The gate is the
     only thing standing between "the swap stage is 30% faster" and "WDDM is
     paging GPU memory into host RAM and the desktop has stopped answering",
     so what matters is that it says NO on this card's real numbers at 4K with
     an enhancer, and YES when there is genuinely room.
  2. The plan the estimate produces: one combined swap stage when the eyes run
     together, two when they do not, and an overlapped stage that owns no part
     of the progress bar.
  3. The forced lane hold from B8/B10(c), driven through the real watchdog:
     with lane_pause_free_vram_mb set absurdly high the newest lane is frozen
     on the first sample and killed once lane_pause_max_sec passes, and the
     other lane never stops.

Run:  python scripts/tests/test_parallel_eyes.py
"""

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
from guardrails import ResourceGuard

GB = guardrails.GB

# The measured "Good" stack: hyperswap at pixel boost 256 with gpen_bfr_512,
# six execution threads. One 4096x4096 pass holds ~7 GB of VRAM.
GOOD = {
    "swapper_model": "hyperswap_1a_256",
    "pixel_boost": "256x256",
    "enhancer_model": "gpen_bfr_512",
    "frame_enhancer": None,
    "threads": 6,
}
EYE_4K = 4096 * 4096
EYE_2K = 2048 * 2048


class _Machine:
    """Fixed readings for gpu_memory_mb() and memory_status()."""

    def __init__(self, vram_total_mb, vram_used_mb, avail_ram_gb, avail_commit_gb):
        self.vram = (vram_total_mb, vram_used_mb)
        self.memory = (int(32 * GB), int(avail_ram_gb * GB), int(avail_commit_gb * GB))
        self._saved = None

    def __enter__(self):
        self._saved = (guardrails.gpu_memory_mb, guardrails.memory_status)
        guardrails.gpu_memory_mb = lambda: self.vram
        guardrails.memory_status = lambda: self.memory
        return self

    def __exit__(self, *_exc):
        guardrails.gpu_memory_mb, guardrails.memory_status = self._saved
        return False


class VerdictTest(unittest.TestCase):
    """Three machine readings, three answers."""

    def setUp(self):
        self.guard_config = dict(guardrails.default_guard_config())
        self.guard_config["parallel_eyes"] = "auto"

    def test_this_card_refuses_two_4k_lanes_with_an_enhancer(self):
        # 16.3 GB card, 1.4 GB of desktop on it, plenty of everything else.
        # One lane is ~7 GB, so two plus the 1.5 GB margin does not fit and the
        # honest answer is no.
        with _Machine(16311, 1400, avail_ram_gb=24, avail_commit_gb=60):
            ok, reason = guardrails.parallel_eyes_verdict(
                GOOD, self.guard_config, EYE_4K)
        self.assertFalse(ok, reason)
        self.assertIn("VRAM", reason)

    def test_a_lighter_stack_fits(self):
        # Same card, same machine, no face enhancer: ~5 GB a lane.
        with _Machine(16311, 1400, avail_ram_gb=24, avail_commit_gb=60):
            ok, reason = guardrails.parallel_eyes_verdict(
                dict(GOOD, enhancer_model=None), self.guard_config, EYE_2K)
        self.assertTrue(ok, reason)
        self.assertIn("per lane", reason)

    def test_commit_shortage_refuses_even_when_the_card_fits(self):
        # A 24 GB card would hold two lanes; a machine with 6 GB of commit left
        # would not survive them, and commit is what actually refused the Sep 10
        # job.
        with _Machine(24576, 800, avail_ram_gb=24, avail_commit_gb=6):
            ok, reason = guardrails.parallel_eyes_verdict(
                GOOD, self.guard_config, EYE_4K)
        self.assertFalse(ok, reason)
        self.assertIn("committed memory", reason)

    def test_ram_shortage_refuses(self):
        with _Machine(24576, 800, avail_ram_gb=6, avail_commit_gb=60):
            ok, reason = guardrails.parallel_eyes_verdict(
                GOOD, self.guard_config, EYE_4K)
        self.assertFalse(ok, reason)
        self.assertIn("RAM", reason)

    def test_off_never_tries(self):
        config = dict(self.guard_config, parallel_eyes="off")
        with _Machine(24576, 800, avail_ram_gb=24, avail_commit_gb=60):
            ok, reason = guardrails.parallel_eyes_verdict(GOOD, config, EYE_4K)
        self.assertFalse(ok)
        self.assertIn("turned off", reason)

    def test_on_skips_the_margins_but_not_the_card(self):
        config = dict(self.guard_config, parallel_eyes="on")
        # Short of commit and RAM, but the card has room: "on" means the user
        # has decided to accept the paging, and it runs.
        with _Machine(24576, 800, avail_ram_gb=5, avail_commit_gb=5):
            ok, reason = guardrails.parallel_eyes_verdict(GOOD, config, EYE_4K)
        self.assertTrue(ok, reason)
        # Short of VRAM: still refused, because that is a frozen machine and
        # not a slow run.
        with _Machine(16311, 1400, avail_ram_gb=24, avail_commit_gb=60):
            ok, reason = guardrails.parallel_eyes_verdict(GOOD, config, EYE_4K)
        self.assertFalse(ok, reason)

    def test_lane_threads_are_halved_and_never_below_two(self):
        self.assertEqual(guardrails.parallel_eyes_lane_threads({"threads": 6}), 3)
        self.assertEqual(guardrails.parallel_eyes_lane_threads({"threads": 3}), 2)
        self.assertEqual(guardrails.parallel_eyes_lane_threads({"threads": 1}), 2)


class RuntimePlanTest(unittest.TestCase):
    """What the estimate says the run will look like."""

    ESTIMATE = {"frames": 5400, "width": 8192, "height": 4096, "fps": 90.0,
                "duration": 60.0}

    def _plan(self, **kwargs):
        return guardrails.estimate_runtime(
            self.ESTIMATE, GOOD, is_vr=True, calibration={}, **kwargs)

    def test_sequential_plan_has_one_stage_per_eye(self):
        keys = [stage["key"] for stage in self._plan()["stages"]]
        self.assertIn("swap_left", keys)
        self.assertIn("swap_right", keys)
        self.assertNotIn("swap_both", keys)

    def test_parallel_plan_has_one_combined_stage(self):
        plan = self._plan(parallel_eyes=True)
        keys = [stage["key"] for stage in plan["stages"]]
        self.assertIn("swap_both", keys)
        self.assertNotIn("swap_left", keys)
        both = next(s for s in plan["stages"] if s["key"] == "swap_both")
        one = next(s for s in self._plan()["stages"] if s["key"] == "swap_left")
        # The wait is one eye plus the GPU-sharing penalty, not two eyes.
        self.assertAlmostEqual(both["seconds"],
                               one["seconds"] * guardrails.PARALLEL_EYES_PENALTY,
                               places=3)
        self.assertLess(plan["total_seconds"], self._plan()["total_seconds"])
        self.assertIn("both eyes together", both["label"])

    def test_an_overlapped_stage_costs_the_estimate_nothing(self):
        plan = self._plan(overlapped=("split", "merge"))
        split = next(s for s in plan["stages"] if s["key"] == "split")
        self.assertTrue(split.get("overlapped"))
        # Still costed, for the calibration - just not charged to the wait.
        self.assertGreater(split["seconds"], 0)
        sequential = self._plan()
        hidden = sum(s["seconds"] for s in sequential["stages"]
                     if s["key"] in ("split", "merge"))
        self.assertAlmostEqual(plan["total_seconds"],
                               sequential["total_seconds"] - hidden, places=3)

    def test_run_clock_gives_an_overlapped_stage_no_part_of_the_bar(self):
        clock = engine.RunClock(self._plan(overlapped=("split", "merge")))
        self.assertEqual(clock.range("split")[0], clock.range("split")[1])
        left_start, left_end = clock.range("swap_left")
        self.assertGreater(left_end, left_start)
        # The bar still ends where it always did.
        self.assertGreaterEqual(clock.range("merge")[1], 99)


class ForcedLaneHoldTest(unittest.TestCase):
    """
    B10(c) without the GPU: the watchdog holds a lane, then kills it.

    `lane_pause_free_vram_mb` is set above the whole card, so the VRAM trip
    point is crossed on every sample and cannot un-cross itself. That is
    exactly the state the real test induces by asking for 20 GB of free VRAM on
    a 16 GB card - and it is the state a second FaceFusion lane would put the
    machine in for real.
    """

    def setUp(self):
        if not guardrails.IS_WINDOWS:
            self.skipTest("lanes are a Job Object feature; Windows only")
        self.saved_gpu = guardrails.gpu_memory_mb
        # Fixed, so the test does not depend on what else is on the card.
        guardrails.gpu_memory_mb = lambda: (16311, 1500)

        config = dict(guardrails.default_guard_config())
        config.update({
            "watchdog_enabled": True,
            "watchdog_interval_sec": 0.5,
            # Everything the card has is less than this, so the lane trip is
            # permanently crossed.
            "lane_pause_free_vram_mb": 20000,
            "lane_pause_max_sec": 2.0,
            # The whole-job paths must not fire; only the lane policy is under
            # test, and this machine has plenty of RAM and commit.
            "auto_pause_enabled": False,
            "min_free_ram_gb": 0.05,
            "min_free_disk_gb": 0.01,
            "min_free_pagefile_gb": 0.05,
            "lane_pause_free_ram_gb": 0.05,
            "lane_pause_free_commit_gb": 0.05,
        })
        self.lines = []
        self.guard = ResourceGuard(os.environ.get("TEMP", "."), config,
                                   log_cb=self.lines.append).start()
        self.procs = {}
        for lane in ("left", "right"):
            self.procs[lane] = self.guard.spawn(
                [sys.executable, "-c", "import time; time.sleep(60)"], lane=lane)

    def tearDown(self):
        self.guard.close(wait_timeout=3.0)
        guardrails.gpu_memory_mb = self.saved_gpu

    def test_newest_lane_is_held_then_killed_and_the_other_survives(self):
        # Held: the lane registered last, because the older one is the closer
        # to finishing and finishing is what gives its memory back.
        deadline = time.time() + 10
        while time.time() < deadline and not self.guard.lane_paused("right"):
            time.sleep(0.2)
        self.assertTrue(self.guard.lane_paused("right"),
                        "the right lane should have been frozen\n"
                        + "\n".join(self.lines))
        self.assertFalse(self.guard.lane_paused("left"))
        self.assertIsNone(self.procs["left"].poll(),
                          "the left lane must keep running")

        # Killed, once the memory has not come back for lane_pause_max_sec.
        deadline = time.time() + 15
        while time.time() < deadline and "right" not in self.guard.lane_failed:
            time.sleep(0.2)
        self.assertIn("right", self.guard.lane_failed,
                      "the held lane should have been given up on\n"
                      + "\n".join(self.lines))
        deadline = time.time() + 5
        while self.procs["right"].poll() is None and time.time() < deadline:
            time.sleep(0.1)
        self.assertIsNotNone(self.procs["right"].poll())

        # The whole point: the run is not aborted for one lane's problem.
        self.assertIsNone(self.guard.abort_reason)
        self.assertIsNone(self.procs["left"].poll(),
                          "the left lane must still be running after the kill")
        self.assertTrue(any("holding the right lane" in line for line in self.lines),
                        "\n".join(self.lines))
        self.assertTrue(any("Killing that lane only" in line for line in self.lines),
                        "\n".join(self.lines))


class TwoLaneRunTest(unittest.TestCase):
    """
    run_vr_pipeline with both eyes in flight, with FaceFusion stubbed out.

    The swap itself is the one part of this that needs a GPU, so it is the one
    part replaced: everything else - the split, the two lanes, the shared
    progress bar, the checkpoints, the merge, the stereo tag - is the real
    code. What is being proved is that the two passes really do overlap in
    time, that each is given half the threads and its own lane name, and that
    one lane dying does not cost the run the eye that finished.
    """

    @classmethod
    def setUpClass(cls):
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=30,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError):
            raise unittest.SkipTest("ffmpeg is not on PATH")
        cls.work = tempfile.mkdtemp(prefix="vrdeep-lanes-")
        cls.clip = os.path.join(cls.work, "sbs.mp4")
        cls.face = os.path.join(cls.work, "face.png")
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=2",
             "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=2",
             "-filter_complex", "[0:v][1:v]hstack=2[v]", "-map", "[v]",
             "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", cls.clip],
            check=True, capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc=size=64x64:rate=1:duration=1",
             "-frames:v", "1", cls.face],
            check=True, capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    def setUp(self):
        self.real_facefusion = engine.run_facefusion
        self.calls = []
        self.temp_dir = tempfile.mkdtemp(prefix="lane-temp-", dir=self.work)
        self.output = os.path.join(self.temp_dir, "out.mp4")
        # The runtime re-check must agree with the forced verdict, or the run
        # falls back to sequential and this tests nothing.
        self.machine = _Machine(24576, 800, avail_ram_gb=24, avail_commit_gb=60)
        self.machine.__enter__()

    def tearDown(self):
        engine.run_facefusion = self.real_facefusion
        self.machine.__exit__(None, None, None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # A lane name no run will ever use, so the default stub fails nothing -
    # None is a real value here (a sequential pass has no lane).
    NEVER = "__no_such_lane__"

    def _stub(self, fail_lane=NEVER, seconds=0.8):
        def stub(source_path, target_path, output_path, temp_path, config,
                 log_cb=None, progress_cb=None, start_pct=10, end_pct=90,
                 stage_name="FaceFusion", guard=None, guard_config=None, lane=None):
            started = time.time()
            time.sleep(seconds)
            if lane == fail_lane and not any(
                    call["lane"] == fail_lane and call["ok"] for call in self.calls):
                self.calls.append({"lane": lane, "stage": stage_name, "start": started,
                                   "end": time.time(), "threads": config.get("threads"),
                                   "ok": False})
                raise engine.PipelineError(f"the {lane} lane ran out of VRAM (test)")
            shutil.copyfile(target_path, output_path)
            self.calls.append({"lane": lane, "stage": stage_name, "start": started,
                               "end": time.time(), "threads": config.get("threads"),
                               "ok": True})
            if progress_cb:
                progress_cb(end_pct, f"{stage_name}: done")
            return output_path
        return stub

    def _prepared(self):
        config = dict(engine.load_config(), provider="cpu", encoder="libx264",
                      threads=6, chunk_seconds=0, temporal_smoothing=False,
                      upscale_method=None, interpolate_multi=1)
        guard_config = dict(guardrails.load_guard_config())
        guard_config["watchdog_enabled"] = False
        estimate = guardrails.preflight(self.clip, self.temp_dir, config, guard_config,
                                        is_vr=True)[2]
        # Force the verdict on: this machine's real card would say no, and what
        # is under test is what happens when it says yes.
        estimate["parallel_eyes"] = {"ok": True, "reason": "forced on for the test",
                                     "line": "", "lane_threads": 3}
        estimate["runtime"] = guardrails.estimate_runtime(
            estimate, config, is_vr=True, calibration={}, parallel_eyes=True)
        return config, guard_config, estimate

    def _run(self, **kwargs):
        config, guard_config, estimate = self._prepared()
        return engine.run_vr_pipeline(
            self.face, self.clip, self.output, config, layout="sbs",
            merge_encoder="libx264", merge_crf=23, log_cb=lambda m: self.lines.append(m),
            guard=guardrails.NULL_GUARD, prepared=(config, guard_config, estimate),
            temp_dir=self.temp_dir, resume=False, **kwargs)

    def test_both_eyes_run_at_the_same_time(self):
        self.lines = []
        engine.run_facefusion = self._stub()
        # keep_temp so the checkpoint file survives: a run that finishes clears
        # it on purpose, and it is what the last assertion here needs to read.
        result = self._run(keep_temp=True)
        log = "\n".join(self.lines)

        self.assertTrue(os.path.isfile(result), log)
        self.assertEqual(len(self.calls), 2, self.calls)
        lanes = sorted(call["lane"] for call in self.calls)
        self.assertEqual(lanes, ["left", "right"],
                         "each eye must be spawned into its own guard lane")
        # Half the threads each: the two processes share one GPU.
        self.assertEqual([call["threads"] for call in self.calls], [3, 3], self.calls)
        # Genuinely at the same time, not one after the other.
        first, second = sorted(self.calls, key=lambda call: call["start"])
        self.assertLess(second["start"], first["end"],
                        f"the eyes did not overlap: {self.calls}")
        self.assertIn("Running BOTH eyes at once", log)
        # Both eyes are still checkpointed separately, so a later resume can
        # reuse whichever one finished.
        state = engine.RunState(self.temp_dir)
        self.assertIn("swap_left", state.stages, state.stages)
        self.assertIn("swap_right", state.stages, state.stages)

    def test_a_lane_that_dies_is_re_run_on_its_own(self):
        self.lines = []
        engine.run_facefusion = self._stub(fail_lane="right")
        result = self._run()
        log = "\n".join(self.lines)

        self.assertTrue(os.path.isfile(result), log)
        # Three passes: two lanes, one of which died, then the right eye again
        # on its own with the full thread count.
        self.assertEqual(len(self.calls), 3, self.calls)
        retry = self.calls[-1]
        self.assertEqual(retry["lane"], None,
                         "the retry must not run in a lane - there is nothing to "
                         "share the card with any more")
        self.assertEqual(retry["threads"], 6)
        self.assertIn("could not hold two lanes", log)
        self.assertIn("did not survive running alongside", log)

    def test_one_eye_left_to_do_falls_back_to_a_single_pass(self):
        """
        A resume with one eye already swapped must not open two lanes for it.

        The plan still says "both eyes together" - it was made before anyone
        knew an eye was already on disk - so this is the case where what the
        estimate expected and what the run has to do disagree. One pass, no
        lane, the full thread count, and the combined stage's slice of the bar.
        """
        self.lines = []
        engine.run_facefusion = self._stub()
        config, guard_config, estimate = self._prepared()
        engine.run_vr_pipeline(
            self.face, self.clip, self.output, config, layout="sbs",
            merge_encoder="libx264", merge_crf=23, log_cb=self.lines.append,
            guard=guardrails.NULL_GUARD, prepared=(config, guard_config, estimate),
            temp_dir=self.temp_dir, resume=False, keep_temp=True)
        self.assertEqual(len(self.calls), 2, self.calls)

        # Take the right eye away and leave the left one where it is. That is
        # exactly the state an abort during the right eye leaves behind.
        os.remove(os.path.join(self.temp_dir, "right_swapped.mp4"))
        os.remove(self.output)
        self.calls = []
        self.lines = []
        engine.run_facefusion = self._stub()
        result = engine.run_vr_pipeline(
            self.face, self.clip, self.output, config, layout="sbs",
            merge_encoder="libx264", merge_crf=23, log_cb=self.lines.append,
            guard=guardrails.NULL_GUARD, prepared=(config, guard_config, estimate),
            temp_dir=self.temp_dir, resume=True, keep_temp=True)
        log = "\n".join(self.lines)

        self.assertTrue(os.path.isfile(result), log)
        self.assertEqual(len(self.calls), 1,
                         f"only the right eye was left to do: {self.calls}\n{log}")
        call = self.calls[0]
        self.assertEqual(call["stage"], "Right Eye", log)
        self.assertIsNone(call["lane"],
                          "a single pass has nothing to share the card with")
        self.assertEqual(call["threads"], 6, "and so it gets all the threads")
        self.assertIn("Left Eye already swapped", log)


if __name__ == "__main__":
    unittest.main(verbosity=2)
