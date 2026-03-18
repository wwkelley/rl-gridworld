# RL Gridworld

A lightweight gridworld environment for reinforcement learning experiments. Built from scratch for my own self-education. I am interested in cognitive science and would like to implement RL models from cognitive and behavioral modeling research.

## Setup
```bash
git clone https://github.com/wwkelley/rl-gridworld.git
pip install -r requirements.txt
pip install -e .
```

## Project Structure
  # gridworld/
  - Core environment and agent classes
  # experiments/
  - RL experiments with visualizations and writeups
  #  test/
  - Unit tests

## Experiments
| Experiment | Description | Algorithm | Location |
|------------|-------------|-----------|----------|
| exp01_move_right | Agent learns to collect two adjacent resources | Tabular Q-Learning | experiments/tabular_q/experiment_01_move_right |


## Roadmap
- Penalty entities
- Monte Carlo agent
- Variable state representations
- DQN

