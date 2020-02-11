#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 roshan <roshan@roshan-ThinkPad-T440p>
#
# Distributed under terms of the MIT license.

"""

    SpaceInvaders-v0

    Maximize your score in the Atari 2600 game SpaceInvaders. 
    In this environment, the observation is an RGB image of the screen, which is an array of shape (210, 160, 3) 
    Each action is repeatedly performed for a duration of kkk frames, where kkk is uniformly sampled from {2,3,4}\{2, 3, 4\}{2,3,4}.


    Deep Q Network
    
    Model Architecture details 
    3 convolution layer
    2 FC layer -->
                    In --> 128*19*8 
                    Out ---> 512,6(action in space invaders)

"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    def __init__(self,Alpha):
        super(DQN,self).__init__()
        self.conv_one = nn.Conv2d(1, 32, 8, stride=4, padding=1)
        self.conv_two = nn.Conv2d(32,64,4,stride=2)
        self.conv_three = nn.Conv2d(64,128,3)
        self.fullyconnected_one = nn.Linear(128*19*8,512)
        self.fullyconnected_two = nn.Linear(512,6)        
        
        self.optimizer = optim.RMSprop(self.parameters(), lr=Alpha)
        self.loss      = nn.MSELoss()
        self.device    = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self,observation):
        observation = T.Tensor(observation).to(self.device)
        observation = observation.view(-1, 1, 185, 95)
        observation = F.relu(self.conv_one(observation))
        observation = F.relu(self.conv_two(observation))
        observation = F.relu(self.conv_three(observation))
        observation = observation.view(-1, 128*19*8)
        observation = F.relu(self.fullyconnected_one(observation))

        actions = self.fullyconnected_two(observation)

        return actions



































