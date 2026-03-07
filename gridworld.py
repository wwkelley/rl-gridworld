import numpy as np
import random

class Environment:
    def __init__(self, dims: tuple, agent: Agent, max_steps: int, step_penalty: float, entities: list = None):
        #Time
        self.step_num = 0
        self.max_steps = max_steps
        self.step_penalty = step_penalty

        #Space
        self.dims = dims
        self.agent = agent
        self.entities = entities if entities is not None else []
        self.init_entities = list(self.entities)


    def get_step_num(self) -> int:
        return self.step_num

    def get_agent(self) -> Agent:
        return self.agent

    def get_current_entities(self) -> list:
        return self.entities

    def add_entity(self, new_entity) -> None:
        self.entities.append(new_entity)

    def remove_entity(self, current_entity) -> None:
        self.entities.remove(current_entity)

    def is_valid_position(self, position: tuple) -> bool:
        if (
            position[0] >= 0 and position[0] < self.dims[0] and
            position[1] >= 0 and position[1] < self.dims[1]):
                return True
        else:
                return False

    def get_state(self) -> np.array:

        #Note: Assumes a 3x3 state centered on the agent.
        state = np.zeros((3, 3))
        agent_row, agent_col = self.agent.get_position()
        state[1, 1] = 1

        #Check if state includes boundary, set to -1 if so
        for row in range (agent_row - 1, agent_row + 2):
            for col in range(agent_col - 1, agent_col + 2):
                if(
                    row >= self.dims[0] or row < 0 or
                    col >= self.dims[1] or col < 0):

                        #Offset coordinates to center at [1, 1]
                        state_row = 1 + (row - agent_row)
                        state_col = 1 + (col - agent_col)
                        state[state_row, state_col] = -1

        #Check if state includes entities, set to 2 if so
        for entity in self.entities:
            entity_row, entity_col = entity.get_position()
            if( 
                entity_row >= agent_row - 1 and entity_row < agent_row + 2 and
                entity_col >= agent_col - 1 and entity_col < agent_col + 2):

                    #Offset coordinates to center at [1, 1]
                    state_row = 1 + (entity_row - agent_row)
                    state_col = 1 + (entity_col - agent_col)
                    state[state_row, state_col] = 2

        return state

    def step(self, move: str) -> tuple:

        self.step_num += 1
        done = self.step_num >= self.max_steps

        #Request new position for agent
        current_row, current_col = self.agent.get_position()
        requested_move_row, requested_move_col = self.agent.moves[move]
        requested_position = (current_row + requested_move_row, current_col + requested_move_col)

        #If requested position is valid, move agent
        if self.is_valid_position(requested_position):
            self.agent.set_position(requested_position)

            #If entity in new position, add reward/penalty to cumulative
            for entity in self.entities:
                if entity.get_position() == requested_position:
                    self.remove_entity(entity)
                    return (done, entity.get_reward())

        return (done, self.step_penalty)

    def reset(self) -> None:
        self.step_num = 0
        self.agent.reset()
        self.entities = list(self.init_entities)
        for entity in self.entities:
            entity.reset()
        

    def render(self) -> None:
        grid = np.zeros(self.dims)

        #Render Agent location
        agent_row, agent_col = self.agent.get_position()
        grid[agent_row, agent_col] = 1

        #Render Entity location
        for entity in self.entities:
            entity_row, entity_col = entity.get_position()
            grid[entity_row, entity_col] = 2
            
        print(grid)


class Agent:
    def __init__(self, position: tuple, moves: dict = None):
        self.position = position
        self.init_position = position
        self.moves = moves if moves is not None else {
            "stay": (0, 0), 
            "up":   (1, 0),
            "up_left": (1, -1),
            "up_right": (1, 1),
            "down": (-1, 0),
            "down_left": (-1, -1),
            "down_right": (-1, 1),
            "right": (0, 1),
            "left": (0, -1)
        }

    def get_position(self) -> tuple:
        return self.position

    def set_position(self, position: tuple) -> None:
        self.position = position

    def get_moves(self) -> dict:
        return self.moves

    def reset(self) -> None:
        self.position = self.init_position


class Entity:
    def __init__(self, position: tuple, reward: float):
        self.position = position
        self.init_position = position
        self.reward = reward

    def get_position(self) -> tuple:
        return self.position

    def set_position(self, position: tuple) -> None:
        self.position = position

    def set_reward(self, reward: float) -> None:
        self.reward = reward

    def get_reward(self) -> float:
        return self.reward

    def reset(self) -> None:
        self.position = self.init_position


def main():
    agent = Agent((3, 4))
    resource_one = Entity((4, 5), 1)
    resource_two = Entity((3, 5), 1.5)
    entities = [resource_one, resource_two]

    environment = Environment((10, 10), agent, 20, -0.1, entities)
    environment.render()
    print(environment.step('right'))
    environment.render()
    print(environment.step('up'))
    environment.render()

    print(environment.get_agent().get_position())

    environment.reset()
    environment.render()
    print(environment.get_state())

    environment.get_agent().set_position((9, 9))
    print(environment.get_state())

if __name__=="__main__":
    main()

