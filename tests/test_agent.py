import pytest
import numpy as np
from gridworld.agent import TabularQAgent

@pytest.fixture
def basic_tabularq_agent_greedy():
    return TabularQAgent(alpha=0.1, gamma=0.9, epsilon=0)

@pytest.fixture
def basic_actions():
    return {"stay": (0,0), "right": (0,1), "left": (0,-1), "up": (-1,0), "down": (1,0)}

@pytest.fixture
def basic_state():
    state = np.zeros((5, 5))
    state[0, 0] = 1
    return state

def test_greedy_tabularq_select_action(basic_state, basic_actions, basic_tabularq_agent_greedy):
    state_key = tuple(basic_state.flatten())
    basic_tabularq_agent_greedy.q_table[(state_key, 'right')] = 10.0 #manually set one value of q(s,a) to be highest
    assert basic_tabularq_agent_greedy.select_action(basic_state, basic_actions) == 'right'

def test_update(basic_state, basic_actions, basic_tabularq_agent_greedy):
    next_state = np.zeros((5, 5))
    next_state[0, 1] = 1 #set agent in next state, move right

    current_state_key = tuple(basic_state.flatten())
    next_state_key = tuple(next_state.flatten())

    basic_tabularq_agent_greedy.update(basic_state, 'right', 1, next_state, basic_actions)

    #0.0 + 0.1 * (1.0 + (0.9 * 0.0) - 0.0) = 0.1
    assert basic_tabularq_agent_greedy.q_table[(current_state_key, 'right')] == pytest.approx(0.1)



