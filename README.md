# Navigation Project

![Env Image](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif)

## Project Details
This project consist of training an agent to navigate and collect bananas in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- `0` - move forward.
- `1` - move backward.
- `2` - turn left.
- `3` - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Results
![Score](scores.png =100x100)
## Instructions

First of all, you need to clone this repository: `git clone https://github.com/jardsantos/navigation_project`

### Getting Started

- In order to use this repository you must have [Anaconda(or just python 3)](https://www.anaconda.com/distribution/) and [Pytorch](https://pytorch.org/get-started/locally/) installed.

- To set up the environment follow the instructions `1`to `3` of [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies).


### How to train the agent

If you want to train the agent using different hyperparameters you have to run the code on terminal:

`python train.py`

The training hyper-params are:
- aa
- bb
- cc

### Using the trained agent to see the results