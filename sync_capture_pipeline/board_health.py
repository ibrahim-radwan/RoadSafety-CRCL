"""
board_health.py — Jetson temperatures and fan, read from sysfs (no root).

Used by capture_session.py (logged every few seconds into session.log and
summarised in session.json) and by the web control page (live Board card).

tj-therm is the hottest junction on the SoC: the number that matters. The
Orin throttles its clocks near ~99 C and shuts down to protect itself near
~105 C; a throttled or shut-down board loses data, so the thresholds below
warn well before that.
"""
from __future__ import annotations

from pathlib import Path

WARN_C = 80.0          # getting hot: check airflow / shade
HOT_C = 90.0           # close to throttling: stop or cool down


def temperatures() -> dict:
    """{zone type: degrees C} for every zone that reports a real value."""
    out = {}
    for z in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            name = (z / "type").read_text().strip()
            value = int((z / "temp").read_text().strip()) / 1000.0
        except (OSError, ValueError):
            continue
        if value > -40:                     # unpowered zones report -256
            out[name] = value
    return out


def fan_percent():
    """Fan PWM as a percentage, or None if there is no PWM fan."""
    for hw in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (hw / "name").read_text().strip() == "pwmfan":
                return round(int((hw / "pwm1").read_text().strip()) * 100 / 255)
        except (OSError, ValueError):
            continue
    return None


def snapshot() -> dict:
    t = temperatures()
    tj = t.get("tj-therm", max(t.values()) if t else None)
    level = "unknown" if tj is None else ("hot" if tj >= HOT_C else "warm" if tj >= WARN_C else "ok")
    return {"tj_c": tj, "cpu_c": t.get("CPU-therm"), "gpu_c": t.get("GPU-therm"),
            "board_c": t.get("Tboard_tegra"), "fan_pct": fan_percent(), "level": level}


def log_line(s: dict) -> str:
    f = lambda v: "-" if v is None else f"{v:.1f}"
    fan = "-" if s["fan_pct"] is None else f"{s['fan_pct']}%"
    return (f"[system] tj {f(s['tj_c'])} C | cpu {f(s['cpu_c'])} C | gpu {f(s['gpu_c'])} C | "
            f"board {f(s['board_c'])} C | fan {fan} | {s['level']}")


if __name__ == "__main__":
    print(log_line(snapshot()))
