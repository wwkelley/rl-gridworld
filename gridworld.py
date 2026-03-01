import numpy as np

###Objects in the GridWorld
class Environment:
    def __init__(self, dims: tuple, entities: list = None):
        self.dims = dims
        self.entities = entities if entities is not None else []

    def get_current_entities(self) -> list:
        return self.entities

    def add_entity(self, new_entity) -> None:
        self.entities.append(new_entity)

    def remove_entity(self, cur_entity) -> None:
        self.entities.remove(cur_entity)

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
    def __init__(self, position: tuple):
        self.position = position

    def get_position(self) -> tuple:
        return self.position

class Predator:
    def __init__(self, position: tuple):
        self.position = position

    def get_position(self) -> tuple:
        return self.position

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
    start_environment.render()

if __name__=="__main__":
    main()

