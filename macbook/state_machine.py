"""Engagement state machine.

States flow:

    SEARCHING -> TARGET_ACQUIRED -> CLASSIFYING -> FOG_CONFIRMED
              -> ENGAGING -> FIBER_COMPROMISED -> TARGET_NEUTRALIZED

At any point we may bail to TARGET_LOST or STAND_DOWN.

Used by the tracker to drive the on-screen overlay text and to gate
side effects (e.g. "only fire the laser when state == ENGAGING").
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum


class EngagementState(str, Enum):
    SEARCHING = "SEARCHING"
    TARGET_ACQUIRED = "TARGET ACQUIRED"
    CLASSIFYING = "CLASSIFYING"
    FOG_CONFIRMED = "FOG CONFIRMED"
    ENGAGING = "ENGAGING"
    FIBER_COMPROMISED = "FIBER COMPROMISED"
    TARGET_NEUTRALIZED = "TARGET NEUTRALIZED"
    TARGET_LOST = "TARGET LOST"
    STAND_DOWN = "STAND DOWN"


_ALLOWED: dict[EngagementState, set[EngagementState]] = {
    EngagementState.SEARCHING: {
        EngagementState.TARGET_ACQUIRED,
        EngagementState.STAND_DOWN,
    },
    EngagementState.TARGET_ACQUIRED: {
        EngagementState.CLASSIFYING,
        EngagementState.TARGET_LOST,
        EngagementState.STAND_DOWN,
    },
    EngagementState.CLASSIFYING: {
        EngagementState.FOG_CONFIRMED,
        EngagementState.TARGET_LOST,
        EngagementState.STAND_DOWN,
    },
    EngagementState.FOG_CONFIRMED: {
        EngagementState.ENGAGING,
        EngagementState.TARGET_LOST,
        EngagementState.STAND_DOWN,
    },
    EngagementState.ENGAGING: {
        EngagementState.FIBER_COMPROMISED,
        EngagementState.TARGET_LOST,
        EngagementState.STAND_DOWN,
    },
    EngagementState.FIBER_COMPROMISED: {
        EngagementState.TARGET_NEUTRALIZED,
        EngagementState.TARGET_LOST,
        EngagementState.STAND_DOWN,
    },
    EngagementState.TARGET_NEUTRALIZED: {EngagementState.SEARCHING},
    EngagementState.TARGET_LOST: {EngagementState.SEARCHING},
    EngagementState.STAND_DOWN: {EngagementState.SEARCHING},
}


_COLOR_BGR: dict[EngagementState, tuple[int, int, int]] = {
    EngagementState.SEARCHING: (180, 180, 180),
    EngagementState.TARGET_ACQUIRED: (0, 255, 255),
    EngagementState.CLASSIFYING: (0, 255, 255),
    EngagementState.FOG_CONFIRMED: (0, 200, 255),
    EngagementState.ENGAGING: (0, 100, 255),
    EngagementState.FIBER_COMPROMISED: (0, 50, 255),
    EngagementState.TARGET_NEUTRALIZED: (0, 200, 0),
    EngagementState.TARGET_LOST: (80, 80, 200),
    EngagementState.STAND_DOWN: (50, 50, 50),
}


class EngagementStateMachine:
    """Tracks current state, validates transitions, fires callbacks."""

    def __init__(self, initial: EngagementState = EngagementState.SEARCHING) -> None:
        self._state = initial
        self._listeners: list[Callable[[EngagementState, EngagementState], None]] = []

    @property
    def state(self) -> EngagementState:
        return self._state

    def color(self) -> tuple[int, int, int]:
        return _COLOR_BGR[self._state]

    def on_change(self, callback: Callable[[EngagementState, EngagementState], None]) -> None:
        self._listeners.append(callback)

    def can_transition(self, target: EngagementState) -> bool:
        return target in _ALLOWED.get(self._state, set())

    def transition(self, target: EngagementState) -> bool:
        if target == self._state:
            return True
        if not self.can_transition(target):
            return False
        prev = self._state
        self._state = target
        for cb in self._listeners:
            try:
                cb(prev, target)
            except Exception:
                pass
        return True

    def reset(self) -> None:
        self.transition(EngagementState.SEARCHING)
