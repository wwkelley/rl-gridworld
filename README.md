# RL Gridworld

A lightweight gridworld environment for reinforcement learning experiments. Built from scratch for my own self-education. Currently exploring Tabular Q and Monte Carlo methods, but my ultimate goal is to implement RL models from cognitive and behavioral modeling research. 

## Setup
```bash
git clone https://github.com/wwkelley/rl-gridworld.git
pip install -r requirements.txt
pip install -e .
```

## Project Structure
  ### gridworld/
  - Core environment and agent classes
  ### experiments/
  - Files for individual RL experiments. See experimental directories for write-ups and results; reward curves in 'results_figures' directory.
  ###  tests/
  - Unit tests for gridworld classes

## Assumptions
- The state is the entirety of the environment grid.
- All 'entities', with positive or negative reward, are represented in the same way (to be changed). 
- Agent actions are limited to movements in the grid.
- When an agent reaches an entity's location, the entity is removed from the grid.

## Experiments
| Experiment | Description | Algorithm | Location |
|------------|-------------|-----------|----------|
| exp01_move_right | Agent learns to collect two adjacent resources | Tabular Q-Learning | experiments/tabular_q/experiment_01_move_right |

## Roadmap
- Variable encoding representation for entities
- Monte Carlo agent
- DQN agent
- Variable state representations
- Trapdoors 



