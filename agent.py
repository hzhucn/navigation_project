from unityagents import UnityEnvironment
import numpy as np
import torch
from dqn_agent import Agent
import sys


if(len(sys.argv)==1):
    print('Please inform the path of the Banana Enviroment: python agent.py PATH')
    sys.exit()
else:
    BANANA_PATH = sys.argv[1]  # path to unit environment


AGENT_PATH = 'output/checkpoint.pth'   # path to trained agent

# initializing environment
env = UnityEnvironment(file_name=BANANA_PATH)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=False)[brain_name]

# number of actions
action_size = brain.vector_action_space_size

# examine the state space 
state = env_info.vector_observations[0]
state_size = len(state)

# dqn agent
agent = Agent(state_size=state_size, action_size=action_size, seed=0)

# loading trained agent
agent.qnetwork_local.load_state_dict(torch.load(AGENT_PATH))

# Trained agent actions
score = 0
while True:
    action = agent.act(state)
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished

    state = next_state
    score += reward
    if done:
        break 
print('Agent score: %d' % score)

env.close()