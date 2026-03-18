import pytest
import numpy as np
from gridworld.environment import Environment, Entity
from gridworld.agent import TabularQAgent

@pytest.fixture
def basic_environment():
    actions = {"stay": (0,0), "right": (0,1), "left": (0,-1), "up": (-1,0), "down": (1,0)}
    entities = [Entity((0,2), 1.0)]
    return Environment((5,5), (0,0), actions, 3, -0.1, entities)

def test_is_valid_position(basic_environment):

    invalid_position_one = (-1, -1)
    invalid_position_two = (5, 5)
    invalid_position_three = (1, 7)

    valid_position_one = (0, 0)
    valid_position_two = (4, 4)
    
    assert not basic_environment.is_valid_position(invalid_position_one)
    assert not basic_environment.is_valid_position(invalid_position_two)
    assert not basic_environment.is_valid_position(invalid_position_three)

    assert basic_environment.is_valid_position(valid_position_one)
    assert basic_environment.is_valid_position(valid_position_two)

def test_invalid_placement():
    actions = {"stay": (0,0)}

    with pytest.raises(AssertionError):
        Environment((5,5), (10,10), actions, 3, -0.1)

    with pytest.raises(AssertionError):
        Environment((5,5), (0, 0), actions, 3, -0.1, [Entity((10, 10), 1.0)])

def test_step(basic_environment):

    step_one_done, step_one_reward, step_one_state = basic_environment.step('up')
    assert step_one_done == False
    assert step_one_reward == -0.1
    assert basic_environment.get_agent_position() == (0, 0)

    basic_environment.step('right')

    step_three_done, step_three_reward, step_three_state = basic_environment.step('right')
    assert step_three_done == True
    assert step_three_reward == pytest.approx(1.0 + (-0.1)) #entity reward + step penalty
    assert len(basic_environment.get_current_entities()) == 0

def test_get_state(basic_environment):
    
    expected_start_state = np.zeros((5, 5))
    expected_start_state[0, 0] = 1 #agent position
    expected_start_state[0, 2] = 2 #entity position
    assert np.array_equal(basic_environment.get_state(), expected_start_state)

    basic_environment.step('right')
    basic_environment.step('right')

    expected_step_three_state = np.zeros((5, 5))
    expected_step_three_state[0, 2] = 1 #agent position after 2 right steps, entity removed
    assert np.array_equal(basic_environment.get_state(), expected_step_three_state)

def test_reset(basic_environment):

    basic_environment.step('right')
    basic_environment.step('right') #agent encounters entity, entity removed

    basic_environment.reset()

    assert basic_environment.get_agent_position() == (0, 0)
    assert basic_environment.get_step_num() == 0
    assert len(basic_environment.get_current_entities()) == 1
    assert basic_environment.get_current_entities()[0].get_position() == (0, 2)
