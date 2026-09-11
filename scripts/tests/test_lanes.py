"""
B8: per-lane freeze and thaw in ResourceGuard.

No GPU and no FaceFusion. Two sleeping python children are spawned into lanes
"a" and "b" under a real Job Object, and the test asks Windows itself - through
NtQueryInformationProcess, not through anything guardrails.py believes - which
of them is suspended.

That distinction is the point of the test. `_set_process_suspended` returning
True only says the call was made; a lane freeze that suspended the wrong
process, or froze the whole job, would return exactly the same thing. The
per-thread suspend counts in the kernel's own process snapshot are the only
answer that cannot be wrong.

Run:  python scripts/tests/test_lanes.py
"""

import ctypes
import os
import sys
import time
import unittest
from ctypes import wintypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import guardrails
from guardrails import ResourceGuard


# ---------------------------------------------------------------------------
# Asking Windows whether a process is frozen
#
# There is no "is this process suspended" call. A process is suspended when
# every one of its threads is, and the thread suspend counts live in
# SYSTEM_PROCESS_INFORMATION, which NtQuerySystemInformation returns for the
# whole machine in one buffer. That is what Process Explorer reads, and it is
# the only source here that guardrails.py does not also write.
# ---------------------------------------------------------------------------

SystemProcessInformation = 5
STATUS_INFO_LENGTH_MISMATCH = 0xC0000004
THREAD_STATE_WAIT = 5
WAIT_REASON_SUSPENDED = 5


class _UNICODE_STRING(ctypes.Structure):
    _fields_ = [("Length", wintypes.USHORT),
                ("MaximumLength", wintypes.USHORT),
                ("Buffer", ctypes.c_void_p)]


class _CLIENT_ID(ctypes.Structure):
    _fields_ = [("UniqueProcess", ctypes.c_void_p),
                ("UniqueThread", ctypes.c_void_p)]


class _SYSTEM_THREAD_INFORMATION(ctypes.Structure):
    _fields_ = [("KernelTime", ctypes.c_longlong),
                ("UserTime", ctypes.c_longlong),
                ("CreateTime", ctypes.c_longlong),
                ("WaitTime", wintypes.ULONG),
                ("StartAddress", ctypes.c_void_p),
                ("ClientId", _CLIENT_ID),
                ("Priority", ctypes.c_long),
                ("BasePriority", ctypes.c_long),
                ("ContextSwitches", wintypes.ULONG),
                ("ThreadState", wintypes.ULONG),
                ("WaitReason", wintypes.ULONG)]


class _SYSTEM_PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("NextEntryOffset", wintypes.ULONG),
                ("NumberOfThreads", wintypes.ULONG),
                ("WorkingSetPrivateSize", ctypes.c_longlong),
                ("HardFaultCount", wintypes.ULONG),
                ("NumberOfThreadsHighWatermark", wintypes.ULONG),
                ("CycleTime", ctypes.c_ulonglong),
                ("CreateTime", ctypes.c_longlong),
                ("UserTime", ctypes.c_longlong),
                ("KernelTime", ctypes.c_longlong),
                ("ImageName", _UNICODE_STRING),
                ("BasePriority", ctypes.c_long),
                ("UniqueProcessId", ctypes.c_void_p),
                ("InheritedFromUniqueProcessId", ctypes.c_void_p),
                ("HandleCount", wintypes.ULONG),
                ("SessionId", wintypes.ULONG),
                ("UniqueProcessKey", ctypes.c_void_p),
                ("PeakVirtualSize", ctypes.c_size_t),
                ("VirtualSize", ctypes.c_size_t),
                ("PageFaultCount", wintypes.ULONG),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivatePageCount", ctypes.c_size_t),
                ("ReadOperationCount", ctypes.c_longlong),
                ("WriteOperationCount", ctypes.c_longlong),
                ("OtherOperationCount", ctypes.c_longlong),
                ("ReadTransferCount", ctypes.c_longlong),
                ("WriteTransferCount", ctypes.c_longlong),
                ("OtherTransferCount", ctypes.c_longlong)]


def suspended_pids():
    """Every PID on the machine whose threads are all in the suspended wait."""
    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    size = 1 << 20
    while True:
        buf = ctypes.create_string_buffer(size)
        needed = wintypes.ULONG(0)
        status = ntdll.NtQuerySystemInformation(
            SystemProcessInformation, buf, size, ctypes.byref(needed))
        if status == 0:
            break
        if status & 0xFFFFFFFF != STATUS_INFO_LENGTH_MISMATCH or size > (1 << 28):
            raise OSError(f"NtQuerySystemInformation failed: {status:#x}")
        size = max(int(needed.value) * 2, size * 2)

    frozen = set()
    offset = 0
    while True:
        entry = _SYSTEM_PROCESS_INFORMATION.from_buffer(buf, offset)
        pid = int(entry.UniqueProcessId or 0)
        count = int(entry.NumberOfThreads)
        if pid and count:
            threads = (_SYSTEM_THREAD_INFORMATION * count).from_buffer(
                buf, offset + ctypes.sizeof(_SYSTEM_PROCESS_INFORMATION))
            if all(t.ThreadState == THREAD_STATE_WAIT
                   and t.WaitReason == WAIT_REASON_SUSPENDED for t in threads):
                frozen.add(pid)
        if not entry.NextEntryOffset:
            break
        offset += int(entry.NextEntryOffset)
    return frozen


SLEEPER = "import time; time.sleep(60)"


class LaneFreezeTest(unittest.TestCase):
    """Two sleeping children, one per lane, frozen and thawed independently."""

    def setUp(self):
        if not guardrails.IS_WINDOWS:
            self.skipTest("lanes are a Job Object feature; Windows only")
        config = dict(guardrails.default_guard_config())
        # Nothing here should be watched: the watchdog would see a machine that
        # is short of memory for its own reasons and freeze lanes underneath the
        # assertions. The lane API is what is under test, not the policy.
        config["watchdog_enabled"] = False
        self.guard = ResourceGuard(
            os.environ.get("TEMP", "."), config, log_cb=lambda m: None).start()
        self.procs = {}
        for lane in ("a", "b"):
            self.procs[lane] = self.guard.spawn(
                [sys.executable, "-c", SLEEPER], lane=lane)
        # Give the interpreters a moment to be more than a stub, so their
        # thread states are meaningful.
        time.sleep(1.0)

    def tearDown(self):
        self.guard.close(wait_timeout=3.0)

    def _pid(self, lane):
        return self.procs[lane].pid

    def assert_frozen(self, expected, message):
        frozen = suspended_pids()
        for lane in ("a", "b"):
            pid = self._pid(lane)
            if lane in expected:
                self.assertIn(pid, frozen, f"{message}: lane {lane} (pid {pid}) "
                                           f"should be suspended")
            else:
                self.assertNotIn(pid, frozen, f"{message}: lane {lane} (pid {pid}) "
                                              f"should be running")

    def test_lane_pids_are_separate_and_inside_the_job(self):
        job_pids = set(self.guard.job_pids())
        a_pids = set(self.guard.lane_pids("a"))
        b_pids = set(self.guard.lane_pids("b"))

        self.assertIn(self._pid("a"), a_pids)
        self.assertIn(self._pid("b"), b_pids)
        # Descendants count: python on this machine starts a child of its own
        # for a -c run, and FaceFusion starts three ffmpegs. A lane that only
        # knew its root would freeze the parent and leave the children running.
        self.assertSequenceEqual(sorted(a_pids | b_pids), sorted(job_pids))
        self.assertFalse(a_pids & b_pids, "a process cannot be in two lanes")
        self.assertTrue(a_pids <= job_pids and b_pids <= job_pids,
                        "a lane must never name a process outside the job")
        self.assertEqual(self.guard.live_lanes(), ["a", "b"])
        # A lane nobody spawned into has no processes, and asking for one must
        # not fall back to "everything".
        self.assertEqual(self.guard.lane_pids("nonexistent"), [])

    def test_pause_lane_freezes_only_that_lane(self):
        self.assert_frozen(set(), "before any freeze")

        expected = len(self.guard.lane_pids("b"))
        frozen = self.guard.pause_lane("b")
        self.assertEqual(frozen, expected)
        self.assertTrue(self.guard.lane_paused("b"))
        self.assertFalse(self.guard.lane_paused("a"))
        self.assert_frozen({"b"}, "after pause_lane('b')")

        # Freezing a lane that is already held is a no-op: NtSuspendProcess
        # nests, and a second freeze here would need a second thaw.
        self.assertEqual(self.guard.pause_lane("b"), 0)

        thawed = self.guard.resume_lane("b")
        self.assertEqual(thawed, expected)
        self.assertFalse(self.guard.lane_paused("b"))
        self.assert_frozen(set(), "after resume_lane('b')")

    def test_whole_job_pause_does_not_thaw_a_held_lane(self):
        self.guard.pause_lane("b")
        self.assert_frozen({"b"}, "lane b held")

        self.assertTrue(self.guard.pause())
        self.assert_frozen({"a", "b"}, "whole job paused")

        self.assertTrue(self.guard.resume())
        # a is running again; b is still held by the guard, which is the whole
        # point - a user Resume must not undo a watchdog freeze.
        self.assert_frozen({"b"}, "after whole-job resume")
        self.assertTrue(self.guard.lane_paused("b"))

        self.guard.resume_lane("b")
        self.assert_frozen(set(), "after releasing the lane")

    def test_run_clock_only_stops_when_every_lane_is_frozen(self):
        self.assertEqual(self.guard.paused_seconds(), 0.0)

        self.guard.pause_lane("b")
        time.sleep(0.4)
        # One lane held: the other is still working, so the run clock keeps
        # running and the estimate is charged for the time.
        self.assertEqual(self.guard.paused_seconds(), 0.0)

        self.guard.pause_lane("a")
        time.sleep(0.4)
        held_both = self.guard.paused_seconds()
        self.assertGreater(held_both, 0.2)

        self.guard.resume_lane("a")
        settled = self.guard.paused_seconds()
        time.sleep(0.4)
        # b alone is held again, so the clock has stopped accumulating.
        self.assertAlmostEqual(self.guard.paused_seconds(), settled, delta=0.05)
        self.guard.resume_lane("b")

    def test_pause_still_takes_when_every_lane_is_already_frozen(self):
        # The watchdog is holding both lanes; the user presses Pause. If that
        # did not take, the moment the watchdog released the lanes the run
        # would start again behind the user's back.
        self.guard.pause_lane("a")
        self.guard.pause_lane("b")
        self.assertTrue(self.guard.pause())
        self.assertTrue(self.guard.paused)
        self.assert_frozen({"a", "b"}, "user pause over two held lanes")

        # Releasing the lanes must not undo the user's pause.
        self.guard.resume_lane("a")
        self.guard.resume_lane("b")
        self.assert_frozen({"a", "b"}, "lanes released, but the user pause stands")
        self.assertTrue(self.guard.resume())
        self.assert_frozen(set(), "after the user resumes")

    def test_kill_lane_leaves_the_other_lane_running(self):
        expected = len(self.guard.lane_pids("b"))
        killed = self.guard.kill_lane("b", "test")
        self.assertEqual(killed, expected)
        self.assertIn("b", self.guard.lane_failed)
        deadline = time.time() + 5
        while self.procs["b"].poll() is None and time.time() < deadline:
            time.sleep(0.1)
        self.assertIsNotNone(self.procs["b"].poll(), "the b lane should be gone")
        self.assertIsNone(self.procs["a"].poll(), "the a lane should still be running")
        self.assertEqual(self.guard.live_lanes(), ["a"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
