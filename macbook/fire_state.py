"""FIRE-mode arming logic.

A drone-like detection isn't enough to start sweeping the laser — we
need a SUSTAINED high-confidence lock so we don't fire at false
positives. This module implements that logic as a small state machine
with three states:

  TRACKING (default)
      We see the drone but haven't held a high-confidence lock long
      enough to trust the target. Laser is OFF.

  ARMED
      Confidence has been above `arm_conf` continuously for at least
      `arm_dwell_s`. Laser is ON, sweeping the aim zone.

  COOLDOWN
      Confidence dropped below `disarm_conf` (or the lock was lost
      entirely). Laser is OFF for at least `cooldown_s` to prevent
      flicker between ARMED <-> TRACKING.

The state machine is consulted every frame with the current predictor
confidence. Returns whether the laser should be active.

Manual override modes:
  "auto"       - state machine decides (default)
  "force_on"   - laser always on while we have ANY detection
  "force_off"  - laser always off (safe / setup mode)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FireOverride = Literal["auto", "force_on", "force_off"]
FireState = Literal["tracking", "armed", "cooldown"]


@dataclass
class FireDecision:
    laser_on: bool
    state: FireState
    dwell_s: float       # how long the current state has been held
    conf: float          # the confidence value used for the decision


class FireController:
    def __init__(
        self,
        arm_conf: float = 0.55,
        disarm_conf: float = 0.30,
        arm_dwell_s: float = 0.8,
        cooldown_s: float = 0.5,
        clock=None,
    ) -> None:
        if disarm_conf >= arm_conf:
            raise ValueError("disarm_conf must be < arm_conf (hysteresis)")
        if arm_dwell_s < 0 or cooldown_s < 0:
            raise ValueError("dwell/cooldown must be >= 0")
        import time
        self.arm_conf = float(arm_conf)
        self.disarm_conf = float(disarm_conf)
        self.arm_dwell_s = float(arm_dwell_s)
        self.cooldown_s = float(cooldown_s)
        self._clock = clock or time.perf_counter

        self._state: FireState = "tracking"
        self._state_entered_at: float = self._clock()
        self._above_arm_since: float | None = None

    @property
    def state(self) -> FireState:
        return self._state

    def update(self, conf: float | None, override: FireOverride = "auto") -> FireDecision:
        """Feed one frame of evidence and get the laser on/off decision.

        `conf` is the predictor's confidence (or None if no detection).
        `override` lets the user force the decision regardless of state.
        """
        now = self._clock()
        c = 0.0 if conf is None else max(0.0, min(1.0, conf))

        # Manual overrides bypass the state machine but we still update
        # internal tracking so transitioning back to "auto" is sane.
        if override == "force_off":
            self._state = "tracking"
            self._state_entered_at = now
            self._above_arm_since = None
            return FireDecision(laser_on=False, state="tracking",
                                dwell_s=0.0, conf=c)
        if override == "force_on":
            return FireDecision(laser_on=True, state="armed",
                                dwell_s=now - self._state_entered_at, conf=c)

        # Track how long confidence has been above arm threshold.
        if c >= self.arm_conf:
            if self._above_arm_since is None:
                self._above_arm_since = now
        else:
            self._above_arm_since = None

        # State transitions
        if self._state == "tracking":
            if (self._above_arm_since is not None
                    and (now - self._above_arm_since) >= self.arm_dwell_s):
                self._state = "armed"
                self._state_entered_at = now
        elif self._state == "armed":
            if c < self.disarm_conf:
                self._state = "cooldown"
                self._state_entered_at = now
                self._above_arm_since = None
        elif self._state == "cooldown":
            if (now - self._state_entered_at) >= self.cooldown_s:
                self._state = "tracking"
                self._state_entered_at = now

        return FireDecision(
            laser_on=(self._state == "armed"),
            state=self._state,
            dwell_s=now - self._state_entered_at,
            conf=c,
        )

    def reset(self) -> None:
        """Force back to TRACKING. Use when target is lost."""
        self._state = "tracking"
        self._state_entered_at = self._clock()
        self._above_arm_since = None
