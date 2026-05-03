"""Tests for the engagement state machine."""

from __future__ import annotations

import pytest

from macbook.state_machine import EngagementState, EngagementStateMachine


class TestEngagementStateMachine:
    def test_default_state_is_searching(self):
        sm = EngagementStateMachine()
        assert sm.state == EngagementState.SEARCHING

    def test_happy_path_full_engagement(self):
        sm = EngagementStateMachine()
        path = [
            EngagementState.TARGET_ACQUIRED,
            EngagementState.CLASSIFYING,
            EngagementState.FOG_CONFIRMED,
            EngagementState.ENGAGING,
            EngagementState.FIBER_COMPROMISED,
            EngagementState.TARGET_NEUTRALIZED,
        ]
        for tgt in path:
            assert sm.transition(tgt) is True
            assert sm.state == tgt

    def test_illegal_transition_blocked(self):
        sm = EngagementStateMachine()
        assert sm.transition(EngagementState.ENGAGING) is False
        assert sm.state == EngagementState.SEARCHING

    def test_self_transition_is_noop_success(self):
        sm = EngagementStateMachine()
        assert sm.transition(EngagementState.SEARCHING) is True

    def test_listener_fires_on_change(self):
        sm = EngagementStateMachine()
        events: list[tuple[EngagementState, EngagementState]] = []
        sm.on_change(lambda old, new: events.append((old, new)))
        sm.transition(EngagementState.TARGET_ACQUIRED)
        sm.transition(EngagementState.CLASSIFYING)
        assert events == [
            (EngagementState.SEARCHING, EngagementState.TARGET_ACQUIRED),
            (EngagementState.TARGET_ACQUIRED, EngagementState.CLASSIFYING),
        ]

    def test_listener_exception_does_not_break_transition(self):
        sm = EngagementStateMachine()
        sm.on_change(lambda old, new: 1 / 0)
        assert sm.transition(EngagementState.TARGET_ACQUIRED) is True
        assert sm.state == EngagementState.TARGET_ACQUIRED

    def test_stand_down_reachable_from_any_active_state(self):
        active = [
            EngagementState.SEARCHING,
            EngagementState.TARGET_ACQUIRED,
            EngagementState.CLASSIFYING,
            EngagementState.FOG_CONFIRMED,
            EngagementState.ENGAGING,
            EngagementState.FIBER_COMPROMISED,
        ]
        for s in active:
            sm = EngagementStateMachine(initial=s)
            assert sm.transition(EngagementState.STAND_DOWN) is True

    def test_terminal_states_return_to_searching(self):
        for term in (
            EngagementState.TARGET_NEUTRALIZED,
            EngagementState.TARGET_LOST,
            EngagementState.STAND_DOWN,
        ):
            sm = EngagementStateMachine(initial=term)
            assert sm.transition(EngagementState.SEARCHING) is True

    def test_reset_returns_to_searching(self):
        sm = EngagementStateMachine(initial=EngagementState.ENGAGING)
        sm.transition(EngagementState.STAND_DOWN)
        sm.reset()
        assert sm.state == EngagementState.SEARCHING

    @pytest.mark.parametrize("state", list(EngagementState))
    def test_every_state_has_a_color(self, state: EngagementState):
        sm = EngagementStateMachine(initial=state)
        color = sm.color()
        assert isinstance(color, tuple) and len(color) == 3
        assert all(0 <= c <= 255 for c in color)

    def test_can_transition_query_does_not_mutate(self):
        sm = EngagementStateMachine()
        assert sm.can_transition(EngagementState.TARGET_ACQUIRED) is True
        assert sm.can_transition(EngagementState.ENGAGING) is False
        assert sm.state == EngagementState.SEARCHING
