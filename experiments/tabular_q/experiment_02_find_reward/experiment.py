"""
Experimental Setup: A gridworld with an Agent placed at (2, 2) and reward points with value 2 at (2, 3) and (2, 4).
Parameters:
    Step Penalty: -0.1
    Epsilon: 0.5, decays at 0.005 per episode.
    Gamma: 0.9
    Maximum Steps: 3
    Maximum Episodes: 500

Hypothesis: After training, the agent will consistently acquire 3.9 reward per episode, and Q-table will show higher values
for 'right' actions in states where the agent hasn't collected both resources. In states where the agent has, expected reward 
will be similar for all actions.

Results: Agent consistently acquires 3.9 reward, Q-table shows higher value for 'right' actions when both resources haven't been
collected, and reward estimates that converge to the step penalty for all actions in states where both resources have been collected.
However, actions other than 'right' consistently valued higher than expected.
TO-DO: Move to README, shorten.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from gridworld import Environment, Entity, TabularQAgent


###Environment Setup
actions = {
        "stay": (0, 0), 
        "up": (-1, 0),
        "down": (1, 0),
        "right": (0, 1),
        "left": (0, -1)
    }

resource_one = Entity((2, 3), 2, 2)
resource_two = Entity((2, 4), 2, 2)
entities = [resource_one, resource_two]

max_steps = 3

environment = Environment((5, 5), (2, 2), actions, max_steps, -0.1, entities)
agent = TabularQAgent(alpha=0.1, gamma=0.9, epsilon=0.5)


###Training Loop

episode_num = 500
epsilon_decay = 0.995

episode_reward = []

for episode in range(episode_num):

        environment.reset()
        state = environment.get_state()
        total_reward = 0

        #Epsilon decay
        agent.epsilon = max(0.01, agent.epsilon * epsilon_decay)

        for step in range(max_steps):
            action = agent.select_action(state, actions)
            done, reward, next_state = environment.step(action)
            agent.update(state, action, reward, next_state, actions)
            total_reward += reward
            state = next_state
            if done:
                break

        episode_reward.append(total_reward)

        #Every 100 episodes, print reward
        if episode % 100 == 0:
            print(f"Episode {episode + 1}, total reward: {total_reward:.2f}")


###Results

#Print q for state with both resources present
env_with_both_resources = Environment((5, 5), (2, 2), actions, 3, -0.1, [Entity((2,3), 2, 2), Entity((2,4), 2, 2)])
state_with_both_resources = tuple(env_with_both_resources.get_state().flatten())

print("Q-values with both resources present")
for action in actions.keys():
    print(f"    {action}: {agent.q_table[(state_with_both_resources, action)]:.4f}")


#Print q for state with one resource present
env_with_one_resource = Environment((5, 5), (2, 3), actions, 3, -0.1, [Entity((2,4), 2, 2)])
state_with_one_resource = tuple(env_with_one_resource.get_state().flatten())

print("Q-values with one resource present")
for action in actions.keys():
    print(f"    {action}: {agent.q_table[(state_with_one_resource, action)]:.4f}")


#Print q for state with no resource present
env_with_no_resource = Environment((5, 5), (2, 4), actions, 3, -0.1, [])
state_with_no_resource = tuple(env_with_no_resource.get_state().flatten())

print("Q-values with no resource present")
for action in actions.keys():
    print(f"    {action}: {agent.q_table[(state_with_no_resource, action)]:.4f}")


###Plotting

#Smoothing method for cumulative reward plot
def smooth(rewards, window=10):
    return np.convolve(rewards, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(episode_reward, alpha=0.3, color='blue', label='Raw')
plt.plot(smooth(episode_reward), color='blue', label='Smoothed')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Experiment 01: Move Right')
plt.legend()
plt.tight_layout()

#Saving
results_figures_dir = os.path.join(os.path.dirname(__file__), 'results_figures')
os.makedirs(results_figures_dir, exist_ok=True)
plt.savefig(os.path.join(results_figures_dir, 'reward_curve.png'))