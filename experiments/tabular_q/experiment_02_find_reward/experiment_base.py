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

max_steps = 10

random_agent_start = np.random.randint(5, size=2)

environment = Environment((5, 5), random_agent_start, actions, max_steps, -0.1, entities)
agent = TabularQAgent(alpha=0.1, gamma=0.9, epsilon=0.5)


###Training Loop

episode_num = 500
epsilon_decay = 0.995

episode_reward = []

for episode in range(episode_num):

        random_agent_start = np.random.randint(5, size=2)

        environment.reset(random_agent_start)
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

        #Every 50 episodes, print reward
        if episode % 50 == 0:
            print(f"Episode {episode + 1}, total reward: {total_reward:.2f}")


###Results

print(f"Q-table entries: {len(agent.q_table)}")
print(f"Distinct states visited: {len(agent.q_table) // len(actions)}")

#Print q for state with both resources present
#env_with_both_resources = Environment((5, 5), (2, 2), actions, 3, -0.1, [Entity((2,3), 2, 2), Entity((2,4), 2, 2)])
#state_with_both_resources = tuple(env_with_both_resources.get_state().flatten())

#print("Q-values with both resources present")
#for action in actions.keys():
#    print(f"    {action}: {agent.q_table[(state_with_both_resources, action)]:.4f}")


###Plotting

#Smoothing method for cumulative reward plot
def smooth(rewards, window=10):
    return np.convolve(rewards, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(episode_reward, alpha=0.3, color='blue', label='Raw')
plt.plot(smooth(episode_reward), color='blue', label='Smoothed')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Experiment 02: Find Reward')
plt.legend()
plt.tight_layout()

#Saving
results_figures_dir = os.path.join(os.path.dirname(__file__), 'results_figures')
os.makedirs(results_figures_dir, exist_ok=True)
plt.savefig(os.path.join(results_figures_dir, 'reward_curve_base.png'))