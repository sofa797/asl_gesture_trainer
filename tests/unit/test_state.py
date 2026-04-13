import pytest


@pytest.mark.unit
def test_initial_state(state):
    assert state.current_index == 0
    assert state.current_target == "A"


@pytest.mark.unit
def test_next_letter(state):
    old = state.current_target
    state.next_letter()
    assert state.current_target != old
    assert state.last_verdict is None
