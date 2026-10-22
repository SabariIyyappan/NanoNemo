"""Background GPU sampler: SM clock, temperature, power, and clock-event reasons over time.

One `nvidia-smi --loop-ms` subprocess streams samples, and a reader thread stamps each one
with time.monotonic(), so a benchmark can slice out the samples of each timed trial.
A separate process keeps the sampling off the training process's CPU time.
"""

import subprocess
import threading
import time
from statistics import median

FIELDS = ["clocks.sm", "clocks.mem", "temperature.gpu", "power.draw", "utilization.gpu", "clocks_event_reasons.active"]

# Bits of clocks_event_reasons that mean the GPU slowed itself down. SW power cap (0x4)
# is left out: under full load a laptop GPU sits at its power limit all the time, and
# that is just how boost works. A real slowdown then shows up as a clock drop.
SLOWDOWN_REASONS = {
    0x08: "hw_slowdown",
    0x20: "sw_thermal",
    0x40: "hw_thermal",
    0x80: "hw_power_brake",
}


def parse_line(line: str) -> dict | None:
    """One CSV line from nvidia-smi (noheader, nounits) -> sample dict; None if unparseable."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != len(FIELDS):
        return None
    try:
        return {
            "sm_mhz": int(parts[0]),
            "mem_mhz": int(parts[1]),
            "temp_c": int(parts[2]),
            "power_w": float(parts[3]),
            "util_pct": int(parts[4]),
            "reasons": int(parts[5], 16),
        }
    except ValueError:  # "[N/A]" on GPUs that don't report a field
        return None


class GpuMonitor:
    """Usage:  with GpuMonitor(interval_ms=200) as mon: ...;  mon.window(t0, t1) -> samples."""

    def __init__(self, interval_ms: int = 200, gpu_index: int = 0):
        self.cmd = [
            "nvidia-smi", f"--id={gpu_index}", f"--query-gpu={','.join(FIELDS)}",
            "--format=csv,noheader,nounits", f"--loop-ms={interval_ms}",
        ]
        self.samples: list[dict] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self):
        try:
            self._proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:  # no nvidia-smi (CPU-only machine): monitor records nothing
            return self
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self):
        for line in self._proc.stdout:
            sample = parse_line(line)
            if sample is not None:
                sample["t"] = time.monotonic()
                self.samples.append(sample)

    def __exit__(self, *exc):
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait()
            self._thread.join(timeout=2)

    def window(self, t0: float, t1: float) -> list[dict]:
        return [s for s in self.samples if t0 <= s["t"] <= t1]


def summarize(samples: list[dict]) -> dict:
    """Clock/temperature/power statistics of one window of samples."""
    if not samples:
        return {"n_samples": 0}
    sm = [s["sm_mhz"] for s in samples]
    reasons = 0
    for s in samples:
        reasons |= s["reasons"]
    return {
        "n_samples": len(samples),
        "sm_mhz": {"median": median(sm), "min": min(sm), "max": max(sm)},
        "mem_mhz_median": median(s["mem_mhz"] for s in samples),
        "temp_c": {"median": median(s["temp_c"] for s in samples), "max": max(s["temp_c"] for s in samples)},
        "power_w_median": median(s["power_w"] for s in samples),
        "util_pct_median": median(s["util_pct"] for s in samples),
        "slowdown_reasons": [name for bit, name in SLOWDOWN_REASONS.items() if reasons & bit],
    }


def throttle_flags(trials: list[dict], tol: float = 0.10) -> list[str]:
    """Reasons to distrust a set of trials (summaries from `summarize`); empty means clean.

    Flags a trial whose lowest SM clock fell more than `tol` below the median clock
    across all trials, and any trial that reported a thermal/hardware slowdown.
    """
    sampled = [t for t in trials if t["n_samples"]]
    if not sampled:
        return ["no GPU samples"]
    ref = median(t["sm_mhz"]["median"] for t in sampled)
    flags = []
    for i, t in enumerate(trials):
        if not t["n_samples"]:
            flags.append(f"trial {i}: no GPU samples")
            continue
        if t["sm_mhz"]["min"] < (1 - tol) * ref:
            flags.append(f"trial {i}: SM clock fell to {t['sm_mhz']['min']} MHz (median of trials {ref:.0f})")
        if t["slowdown_reasons"]:
            flags.append(f"trial {i}: {', '.join(t['slowdown_reasons'])}")
    return flags
