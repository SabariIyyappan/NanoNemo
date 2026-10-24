import time

import pytest

from nanonemo.utils.gpumon import GpuMonitor, parse_line, summarize, throttle_flags


def _sample(sm, temp=70, reasons=0x4):
    return {"sm_mhz": sm, "mem_mhz": 6801, "temp_c": temp, "power_w": 110.0, "util_pct": 100, "reasons": reasons}


def test_parse_line():
    s = parse_line("1755, 6801, 71, 112.43, 100, 0x0000000000000004")
    assert s == {"sm_mhz": 1755, "mem_mhz": 6801, "temp_c": 71, "power_w": 112.43, "util_pct": 100, "reasons": 4}
    assert parse_line("1755, [N/A], 71, 112.43, 100, 0x4") is None
    assert parse_line("garbage") is None


def test_steady_clocks_are_not_flagged():
    trials = [summarize([_sample(sm) for sm in (1740, 1755, 1770)]) for _ in range(3)]
    assert trials[0]["sm_mhz"] == {"median": 1755, "min": 1740, "max": 1770}
    assert trials[0]["slowdown_reasons"] == []  # SW power cap alone is normal under load
    assert throttle_flags(trials) == []


def test_brief_dips_are_not_flagged():
    # Power-limited boost: the clock swings sample to sample, the median holds.
    swinging = summarize([_sample(sm) for sm in (1530, 1335, 1785, 1500, 1520)])
    assert throttle_flags([swinging] * 3) == []


def test_sustained_clock_drop_and_thermal_slowdown_are_flagged():
    steady = summarize([_sample(1755)] * 5)
    dropped = summarize([_sample(1755)] * 2 + [_sample(1200)] * 3)
    hot = summarize([_sample(1755, temp=87, reasons=0x20 | 0x4)] * 5)
    flags = throttle_flags([steady, steady, dropped, hot])
    assert flags == ["trial 2: median SM clock 1200 MHz (median of trials 1755)", "trial 3: sw_thermal"]


def test_clock_held_below_rated_boost_is_flagged():
    # A cap over the whole run: every trial agrees, so only the absolute floor catches it.
    capped = summarize([_sample(1140)] * 5)
    assert throttle_flags([capped] * 3) == []
    flags = throttle_flags([capped] * 3, min_mhz=1380)
    assert flags == [f"trial {i}: median SM clock 1140 MHz is below the rated boost 1380" for i in range(3)]


def test_no_samples_is_flagged():
    assert throttle_flags([summarize([])]) == ["no GPU samples"]


@pytest.mark.gpu
def test_monitor_collects_samples():
    with GpuMonitor(interval_ms=100) as mon:
        t0 = time.monotonic()
        time.sleep(1.0)
        t1 = time.monotonic()
    window = mon.window(t0, t1)
    assert len(window) >= 5
    assert all(s["sm_mhz"] > 0 for s in window)
