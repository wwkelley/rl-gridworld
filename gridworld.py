import numpy as np
import random

###Objects in the GridWorld
class Environment:
    def __init__(self, dims: tuple, entities: list = None):
        self.dims = dims
        self.entities = entities if entities is not None else []

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

    def step(self, action: str) -> None:
        for entity in self.entities:
            if isinstance(entity, Predator):
                current_row, current_col = entity.get_position()
                valid_move = False
                while valid_move == False:
                    suggested_row_move, suggested_col_move = entity.random_move()
                    suggested_position = (current_row + suggested_row_move, current_col + suggested_col_move)
                    if self.is_valid_position(suggested_position):
                        valid_move = True
                        new_position = suggested_position
                entity.set_position(new_position)
            #TO-DO: Agent action
            #elif isinstance(entity, Agent):

    def render(self) -> None:
        grid = np.zeros(self.dims)
        for entity in self.entities:
            row, col = entity.get_position()
            if isinstance(entity, Agent):
                grid[row, col] = 1
            elif isinstance(entity, Predator):
                grid[row, col] = 2
            elif isinstance(entity, Resource):
                grid[row, col] = 4
        print(grid)


class Agent:
    def __init__(self, position: tuple, actions: dict = None):
        self.position = position
        self.actions = actions if actions is not None else {
            "stay": (0, 0), 
            "up":   (1, 0),
            "down": (-1, 0),
            "right": (0, 1),
            "left": (0, -1)
        }

    def get_position(self) -> tuple:
        return self.position

    def set_position(self, position: tuple) -> None:
        self.position = position

    def get_actions(self) -> dict:
        return self.actions

class Predator:
    def __init__(self, position: tuple):
        self.position = position

    def get_position(self) -> tuple:
        return self.position

    def set_position(self, position: tuple) -> None:
        self.position = position

    def random_move(self) -> tuple:
        
        random_row_move = random.choice([-1, 0 , 1])
        random_col_move = random.choice([-1, 0 , 1])
        return (random_row_move, random_col_move)


class Resource:
    def __init__(self, position: tuple):
        self.position = position

    def get_position(self) -> tuple:
        return self.position


def main():
    start_agent = Agent((3, 4))
    start_predator = Predator((5, 5))
    start_resource_one = Resource((1, 8))
    start_resource_two = Resource((7, 7))
    entities = [start_agent, start_predator, start_resource_one, start_resource_two]

    start_environment = Environment((10, 10), entities)
    print(start_environment.is_valid_position((2, 2)))
    print(start_environment.is_valid_position((12, 4)))
    print(start_environment.is_valid_position((9, 9)))

if __name__=="__main__":
    main()

