import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make('BipedalWalker-v2')

print(env.observation_space.shape)
print(env.action_space.shape)
print(env.action_space.high)
print(env.action_space.low)