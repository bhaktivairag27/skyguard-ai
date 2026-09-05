"""
features.py

Shared temporal feature engineering for the anomaly detector — used by BOTH
train_and_evaluate.py (training/evaluation) and main.py (live inference), so
the model always sees features computed the exact same way it was trained on.
This avoids "train/serve skew" — a common bug where a model trained on one
feature definition quietly behaves differently in production because live
features are computed slightly differently.

v3 fix: the cross_sensor physics rule originally compared current temperature
against the mean of its OWN short rolling window (WINDOW=6). That
self-contaminates: the synthetic cross_sensor fault holds humidity near
saturation for up to several hours, so within ~30 minutes the rolling window
fills entirely with the fault's own elevated readings, and "deviation from
recent mean" collapses toward zero right when the fault is still ongoing.
Measured effect: this let only ~7% of true cross_sensor faults be caught by
the rule (22/305 on a 10-day/5-station synthetic run), with the ML model
catching the rest (or missing them too).

Fix: maintain a SEPARATE temperature baseline that only accumulates readings
taken while humidity looks normal (<= CROSS_SENSOR_HUMIDITY_THRESHOLD). Since
the fault forces humidity high for its entire duration, none of the fault's
own readings can ever pollute this baseline — it always reflects genuine
pre-fault conditions, however long the fault lasts.
"""

from collections import deque
from typing import Dict

import numpy as np

RAW_FEATURE_NAMES = ["temperature_c", "pressure_hpa", "humidity_pct"]

FEATURE_NAMES = [
    "temperature_c", "pressure_hpa", "humidity_pct",
    "delta_temperature_c", "delta_pressure_hpa", "delta_humidity_pct",
    "rolling_std_temperature_c", "rolling_std_pressure_hpa", "rolling_std_humidity_pct",
    "max_stale_streak",
    "long_dev_temperature_c", "long_dev_pressure_hpa", "long_dev_humidity_pct",
]

WINDOW = 6  # ~30 min of history at 5-min intervals — catches sudden faults
LONG_WINDOW = 60  # ~5 hours — long enough to see gradual drift (which unfolds over 3-10 hrs)
STALE_EPSILON = 1e-6  # treat values closer than this as "unchanged" (float precision safety)

# Physics rule thresholds for cross_sensor detection — tuned against the
# generator's own fault definition (humidity forced to ~95% while temp rises
# 4-8C), kept a bit looser so it generalizes beyond the exact synthetic values.
CROSS_SENSOR_HUMIDITY_THRESHOLD = 88.0
CROSS_SENSOR_TEMP_DEVIATION_THRESHOLD = 1.0
BASELINE_WINDOW = 30  # readings kept for the humidity-gated temperature baseline


class StationFeatureBuilder:
    """
    Maintains rolling per-station history and computes a temporal feature
    vector for each new reading.

    One instance should be kept alive per station — in training, one is
    created per station and fed its readings in chronological order; in
    live inference, main.py keeps one instance per station_id across
    requests so it always reflects that station's real recent history.
    """

    def __init__(self, window: int = WINDOW, long_window: int = LONG_WINDOW):
        self.window = window
        self.long_window = long_window
        self._history: Dict[str, deque] = {
            name: deque(maxlen=window) for name in RAW_FEATURE_NAMES
        }
        self._long_history: Dict[str, deque] = {
            name: deque(maxlen=long_window) for name in RAW_FEATURE_NAMES
        }
        self._stale_streak: Dict[str, int] = {name: 0 for name in RAW_FEATURE_NAMES}
        self._last_value: Dict[str, float] = {name: None for name in RAW_FEATURE_NAMES}
        # Temperature baseline gated on humidity looking normal — see the v3
        # fix note at the top of this file. Only ever contains readings taken
        # while humidity was <= CROSS_SENSOR_HUMIDITY_THRESHOLD, so a
        # sustained high-humidity fault can never pull its own baseline up
        # to match itself.
        self._temp_baseline: deque = deque(maxlen=BASELINE_WINDOW)

    def update_and_build(self, temperature_c: float, pressure_hpa: float, humidity_pct: float) -> np.ndarray:
        """Feed in a new reading, update internal state, and return its feature vector."""
        values = {
            "temperature_c": temperature_c,
            "pressure_hpa": pressure_hpa,
            "humidity_pct": humidity_pct,
        }
        deltas = {}
        rolling_stds = {}
        long_devs = {}

        for name, val in values.items():
            last = self._last_value[name]
            deltas[name] = 0.0 if last is None else val - last

            if last is not None and abs(val - last) < STALE_EPSILON:
                self._stale_streak[name] += 1
            else:
                self._stale_streak[name] = 0

            self._history[name].append(val)
            rolling_stds[name] = float(np.std(self._history[name])) if len(self._history[name]) > 1 else 0.0

            # Long-term deviation: compare current value to the mean of
            # everything seen so far in the long window, BEFORE adding the
            # current value in — this is what makes gradual drift visible,
            # since a slowly diverging value will pull away from its own
            # longer-run average even when each individual step is small.
            long_hist = self._long_history[name]
            long_devs[name] = 0.0 if len(long_hist) == 0 else float(val - np.mean(long_hist))
            long_hist.append(val)

            self._last_value[name] = val

        # Only feed the humidity-gated temperature baseline while humidity
        # looks normal — this must happen with THIS reading's own humidity,
        # so a fault's elevated humidity readings are excluded from the
        # moment the fault starts, not one step late.
        if humidity_pct <= CROSS_SENSOR_HUMIDITY_THRESHOLD:
            self._temp_baseline.append(temperature_c)

        max_stale_streak = float(max(self._stale_streak.values()))

        return np.array([
            values["temperature_c"], values["pressure_hpa"], values["humidity_pct"],
            deltas["temperature_c"], deltas["pressure_hpa"], deltas["humidity_pct"],
            rolling_stds["temperature_c"], rolling_stds["pressure_hpa"], rolling_stds["humidity_pct"],
            max_stale_streak,
            long_devs["temperature_c"], long_devs["pressure_hpa"], long_devs["humidity_pct"],
        ])

    def check_cross_sensor_rule(self) -> bool:
        """
        Physics-informed check, evaluated AFTER update_and_build has been
        called for the current reading (so self._history and
        self._temp_baseline reflect it).

        Temperature and humidity normally move inversely — humidity drops as
        temperature rises. If humidity is pinned near saturation WHILE
        temperature is simultaneously above its own PRE-FAULT baseline (not
        its live, potentially fault-contaminated window), that combination is
        physically implausible and is a strong, direct signal of a
        cross-sensor fault — more reliable here than relying on the ML model
        to rediscover this relationship among many other features.

        Uses self._temp_baseline (humidity-gated) rather than the live
        self._history window, since the live window fills with the fault's
        own elevated readings within ~30 minutes on a sustained fault,
        making a live-window comparison blind to the fault it's supposed to
        catch for all but its first few readings.
        """
        humidity_hist = self._history["humidity_pct"]
        if len(humidity_hist) == 0 or len(self._temp_baseline) < 3:
            # Not enough pre-fault history yet to have a trustworthy baseline
            # (e.g. right at startup, or humidity has been elevated since the
            # very first reading for this station) — don't guess.
            return False

        current_temp = self._history["temperature_c"][-1]
        current_humidity = humidity_hist[-1]
        baseline_temp_mean = float(np.mean(self._temp_baseline))

        return (
            current_humidity > CROSS_SENSOR_HUMIDITY_THRESHOLD
            and current_temp > baseline_temp_mean + CROSS_SENSOR_TEMP_DEVIATION_THRESHOLD
        )