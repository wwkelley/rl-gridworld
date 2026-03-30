# Experiment 02: Find Reward

## Setup
A gridworld with an agent placed at a random grid location, with reward points with value 2 placed at (2, 3) and (2, 4). Parameters:
- Grid size: 5x5
- Step Penalty: -0.1
- Epsilon: 0.5, decays at 0.005 per episode.
- Gamma: 0.9
- Maximum Steps: 10
- Maximum Episodes: 500

10 steps are allowed to give the agent time to find the resource entities.

## Hypothesis
- After training, the agent will consistently find both reward entities; this will converge toward acquiring 3.0 reward per episode: 
(2 * entity reward) - (step penalty * max steps) = (2 * 2) - (10 * 0.1) = 3
- The convergence speed will be slower with this experimental setup than with experiment 01, as the agent will learn Q values for
a greater number of states given its random initial placement.

## Results

### Base Experiment
- After training, agent consistently finds both reward entities, with total reward converging to 3.0.
- Contrary to the hypothesis, the results of the base experiment show convergence speed faster than in experiment 01, with the raw reward
curve plateauing entirely by episode 210. In experiment 01, only the raw reward curve does not plateau at all. Addtionally, random initial
placement did not lead the agent to learn Q values for a greater number of states in comparison to experiment 01.

### Slower Epsilon Decay Experiment
- With slower epsilon decay, the raw reward curve converges more slowly, around episode 400. Agent learned about the same number of states.

### Lower Step Count Experiment
- When the step count is lowered from 10 to 5, agent does not converge to new total expected reward (3.5) within 500 episodes. At 6 steps, agent
also does not converge to new total expected reward (3.4) within 500 episodes. At 8 steps, agent converges to new total expected reward (3.2) around
episode 280. With step count 6 and below, agent learns Q values for fewer and fewer states.

## Discussion
- The rate of epsilon decay, and more than anything else the total step count, determined the speed at which the agent converged to total expected
reward. 
- Given the worst initial position (0, 0), the agent requires 6 steps to find both reward entities in the grid, so it makes sense that agent did not
converge with only 5 steps. That the agent learns Q values for around the same number of states until the step count decreases to the point where reward cannot
be reached also makes sense; once reward is found, agent generally discovers ways to chart paths between itself and reward.
- Random initial agent position did not increase the number of states visited, contrary to expectations. Once epsilon decays, the agent converges on efficeint paths
to the reward entities regardless of starting position, visiting fewer states than the agent in experiment 01.