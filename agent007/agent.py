#/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 roshan <roshan@roshan-ThinkPad-T440p>
#
# Distributed under terms of the MIT license.

"""
            The Agent 
                        
"""


class Agent(object):
    def __init__(self,gamma,epsilon,alpha,maxMemSize,episodeEnd=0.05,
                 replace=10000, actionSpace=[0,1,2,3,4,5]):

    self.GAMMA  = gamma
    self.EPSILON = epsilon
    self.EPS_END = eps_end
    self.actionSpace = actionSpace
    self.memSize = maxMemSize
    self.steps = 0
    self.learn_step_counter = 0
    self.memory = []
    self.memCount = 0
    self.replace_tar_cnt = replace
    self.Q_eval = DQN(alpha)
    self.Q_next = DQN(alpha)


    def storeTransition(self,state,action,reward,state_):
        if self.memCount < self.memSize:
            self.memory.append([state,action,reward,state_])
        else:
            self.memory[self.memCount%self.memSize] = [state,action,reward,state_]
            self.memCount +=1

    def chooseAction(self, observation):
        rand = np.random.random()
        actions = self.Q_eval.forward(observation)
        if rand < 1 - self.EPSILON:
            action = T.argmax(actions[1]).item()
        else:
            action = np.random.choice(self.actionSpace)
        self.steps += 1

        return action

    def learn(self,batch_size):
        self.Q_eval.optimizer.zero_gradient()
        if self.replace_tar_cnt is not None and \
           self.learn_step_counter % replace_tar_cnt == 0:
               self.Q_next.load_state_dict(self.Q_eval.state_dict())


        if memCount + batch_size < memSize:
            mem_start = int(np.random.choice(range(self.memCount)))
        else:
            mem_start = int(np.random.choice(range(self.memCount - batch_size -1)))

        minibatch = self.memory[mem_start:mem_start + batch_size]
        memory = np.array(minibatch)
        
        Qpred = self.Q_eval.forward(list(memory[:,0][:])).self.Q_eval.to(device)
        Qnext = self.Q_next.forward(list(memory[:,3][:])).self.Q_eval.to(device)

        maxA = T.argmax(Qnext,dim=1).to(self.Q_eval.device) 
        rewards = T.Tensors(list(memory[:,2])).to(self.Q_eval.device)
        Qtarget = Qpred
        Qtarget[:,maxA] = rewards + self.GAMMA*T.max(Qnext[1])

        if self.steps > 500:
            if self.EPSILON - 1e-4 > self.EPS_END:
                self.EPSILON -= 1e-4
            else:
                self.EPSILON = self.EPS_END

        loss = self.Q_eval.loss(Qtarget,Qpred).to(self.Q_eval.device)
        loss.backward()

        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1













         





    






            



    























