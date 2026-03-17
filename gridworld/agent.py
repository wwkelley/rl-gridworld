"""
agent.py
Agent classes for gridworld RL implementation. Will expand to include other learning algorithms.
Classes: TabularQAgent
"""
import numpy as np
import random
from collections import defaultdict

class TabularQAgent:
    """
    Agent implementing tabular Q learning for RL experiments.
    """
    def __init__(self, alpha: float, gamma: float, epsilon: float):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(float)

    def select_action(self, state: np.array, actions: dict) -> str:
        """
        Selects an action using epsilon-greedy strategy: with probability
        epsilon, selects a random action (exploration), otherwise selects action
        with the highest Q-value (exploitation).
        """
        
        #Flatten numpy array to use as key for q-table.
        state_key = tuple(state.flatten())

        #Epsilon-greedy strategy for action selection
        if random.random() < self.epsilon:
            return random.choice(list(actions.keys()))
        else:
            return max(actions.keys(), key=lambda a: self.q_table[(state_key, a)])

    def update(self, state: np.array, action: str, reward: float, next_state: np.array, actions: dict) -> None:
        """
        Updates Q-table using the Bellman equation:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
        """

        #Flatten numpy array to use as key for q-table.
        current_state_key = tuple(state.flatten())
        next_state_key = tuple(next_state.flatten())

        current_qsa = self.q_table[(current_state_key, action)]
        max_q_splus1_a = max(self.q_table[(next_state_key, a)] for a in actions.keys())

        #Update qsa according to Bellman equation
        updated_qsa = current_qsa + (self.alpha * (reward + (self.gamma * max_q_splus1_a) - current_qsa))

        #Add updated qsa to q-table
        self.q_table[(current_state_key, action)] = updated_qsa