import numpy as np
import random


# ========================
# ENVIRONMENT CLASSES
# ========================

class Environment:
    def __init__(self, dims: tuple, agent_position: tuple, actions: dict, max_steps: int, step_penalty: float, entities: list = None):
        #Time
        self.step_num = 0
        self.max_steps = max_steps
        self.step_penalty = step_penalty

        #Space
        self.dims = dims
        self.agent_position = agent_position
        self.init_agent_position = agent_position
        self.entities = entities if entities is not None else []
        self.init_entities = list(entities) if entities is not None else []

        #Interaction
        self.actions = actions

    ###Getter and Setter Methods
    def get_step_num(self) -> int:
        return self.step_num

    def get_agent_position(self) -> tuple:
        return self.agent_position

    def get_current_entities(self) -> list:
        return self.entities

    def add_entity(self, new_entity) -> None:
        self.entities.append(new_entity)

    def remove_entity(self, current_entity) -> None:
        self.entities.remove(current_entity)

    def get_actions(self) -> dict:
        return self.actions

    ###Environmental Update Methods
    def is_valid_position(self, position: tuple) -> bool:
        if (
            position[0] >= 0 and position[0] < self.dims[0] and
            position[1] >= 0 and position[1] < self.dims[1]):
                return True
        else:
                return False

    def get_state(self) -> np.array:

        state = np.zeros(self.dims)

        #Add agent position
        agent_row, agent_col = self.agent_position
        state[agent_row, agent_col] = 1

        #Add entity positions
        for entity in self.entities:
            entity_row, entity_col = entity.get_position()
            state[entity_row, entity_col] = 2

        return state

    def step(self, action: str) -> tuple:

        self.step_num += 1
        done = self.step_num >= self.max_steps

        #Request new position for agent
        current_row, current_col = self.agent_position
        requested_move_row, requested_move_col = self.actions[action]
        requested_position = (current_row + requested_move_row, current_col + requested_move_col)

        #If requested position is valid, move agent
        if self.is_valid_position(requested_position):
            self.agent_position = requested_position

            #If entity in new position, add reward/penalty to cumulative
            for entity in self.entities:
                if entity.get_position() == requested_position:
                    self.remove_entity(entity)
                    return (done, entity.get_reward(), self.get_state())

        return (done, self.step_penalty, self.get_state())

    def reset(self) -> None:
        self.step_num = 0
        self.agent_position = self.init_agent_position
        self.entities = list(self.init_entities)
        for entity in self.entities:
            entity.reset()
        
    ###Visualization Methods
    def render(self) -> None:
        grid = np.zeros(self.dims)

        #Render Agent location
        agent_row, agent_col = self.agent_position
        grid[agent_row, agent_col] = 1

        #Render Entity location
        for entity in self.entities:
            entity_row, entity_col = entity.get_position()
            grid[entity_row, entity_col] = 2
            
        print(grid)

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


# ========================
# AGENT CLASSES
# ========================

class Agent:
    #TO-DO: Add Q-learning functionality!
    def __init__(self, position: tuple, moves: dict = None):
        self.foo = "foo"

def main():
    actions = {
        "stay": (0, 0), 
        "up": (-1, 0),
        "down": (1, 0),
        "right": (0, 1),
        "left": (0, -1)
    }
    resource_one = Entity((4, 5), 1)
    resource_two = Entity((3, 5), 1.5)
    entities = [resource_one, resource_two]

    environment = Environment((10, 10), (3, 4), actions, 20, -0.1, entities)
    
    print("Initial state:")
    environment.render()

    print("\nStep right:")
    print(environment.step('right'))
    environment.render()

    print("\nStep up (onto resource):")
    print(environment.step('up'))
    environment.render()

    print("\nStep right (onto resource):")
    print(environment.step('right'))
    environment.render()

    print("\nReset:")
    environment.reset()
    environment.render()
    print("Step num after reset:", environment.get_step_num())
    print("Agent position after reset:", environment.get_agent_position())

if __name__=="__main__":
    main()

