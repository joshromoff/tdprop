import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init
from backpack import backpack, extend


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, num_actions, base_kwargs=None, extra_kwargs=None):
        super(Policy, self).__init__()
        self.use_backpack = extra_kwargs['use_backpack']
        self.recurrent_hidden_state_size = 1 
        num_outputs = num_actions
        hidden_size = 512 
        conv_init_ = lambda m: init(m, nn.init.orthogonal_, 
                lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        lin_init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.model = nn.Sequential(conv_init_(nn.Conv2d(obs_shape[0], 32, 8, stride=4)), nn.ReLU(),
                conv_init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(), 
                conv_init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
                conv_init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU(),
                lin_init_(nn.Linear(hidden_size, num_outputs)))
        if self.use_backpack:
            extend(self.model)
        
        self.model.train()
    
    def forward(self, inputs):
        qs = self.model(inputs / 255.) 
        return qs
