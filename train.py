from unityagents import UnityEnvironment
import numpy as np
import torch
from dqn_agent import Agent
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

if(len(sys.argv)==1):
    print('Please inform the path of the Banana Enviroment: python train.py PATH')
    sys.exit()
else:
    BANANA_PATH = sys.argv[1]  # path to unit environment

# Training Parameters
# -----------------------------------------------------------------------------------------------------------
N_EPISODES = 1000                       # maximum number of training episodes
MAX_T = 1000                            # maximum number of timesteps per episode
EPS_START = 1.0                         # starting value of epsilon, for epsilon-greedy action selection
EPS_END = 0.01                          # minimum value of epsilon
EPS_DECAY = 0.995                       # multiplicative factor (per episode) for decreasing epsilon
SAVE_AGENT = True                       # whether to save trained agent
AGENT_PATH = 'output/checkpoint.pth'   # path to save agent
SAVE_PLOT = True                        # whether to save plot scores
PLOT_PATH = 'output/plot_scores.png'   # path to save plot scores
# -----------------------------------------------------------------------------------------------------------

# initializing environment
env = UnityEnvironment(file_name=BANANA_PATH)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of actions
action_size = brain.vector_action_space_size

# examine the state space 
state = env_info.vector_observations[0]
state_size = len(state)

# dqn agent
agent = Agent(state_size=37, action_size=4, seed=0)

# Start Training
print()                                     # jump one line
scores = []                                 # list containing scores from each episode
scores_window = deque(maxlen=100)           # last 100 scores
scores_window_ = np.zeros((N_EPISODES,))    # list containing average of scores_window after each 100 episodes
eps = EPS_START                             # initialize epsilon
for i_episode in range(1, N_EPISODES+1):
    
    env_info = env.reset(train_mode=True)[brain_name]   # reset the environment
    state = env_info.vector_observations[0]             # get the current state
    score = 0
    for t in range(MAX_T):
        action = agent.act(state, eps)
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break 
    scores_window.append(score)       # save most recent score
    scores.append(score)              # save most recent score
    eps = max(EPS_END, EPS_DECAY*eps) # decrease epsilon
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        scores_window_[i_episode-100:i_episode] = np.mean(scores_window)

# End Training

env.close()

if SAVE_AGENT:
    if np.mean(scores_window)>=13:
        torch.save(agent.qnetwork_local.state_dict(), AGENT_PATH)
    else:
        print('Problem not solved, did not save the model')

if SAVE_PLOT:

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores, label='Score per episode')
    plt.plot(np.arange(len(scores)), scores_window_, 'r-', label='Average Score per 100 episodes')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.savefig(PLOT_PATH)