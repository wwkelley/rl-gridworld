import numpy as np

###Objects in the GridWorld
class Environment:
    def __init__(self, state_matrix):
        self.state = state_matrix

    def get_current_state(self):
        return self.state

    def set_state(self, new_state):
        self.state = new_state

    def render(self):
        print(self.state)

class Agent:
    def __init__(self, position):
        self.position = position

    def get_position(self):
        return self.position

class Predator:
    def __init__(self, position):
        self.position = position

    def get_position(self):
        return self.position

class Resources:
    def __init__(self, position):
        self.position = position

    def get_position(self):
        return self.position


def main():
    start_environment = Environment(np.zeros((10, 10)))
    start_environment.render()

if __name__=="__main__":
    main()

