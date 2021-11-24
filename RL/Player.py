from configuration import *
from baseline.baseAgent import baseAgent
from multiprocessing import Process
from collections import deque
from copy import deepcopy

from Simulator.simulator import Simulator
from typing import List
import numpy as np
import _pickle as pickle
import torch
import redis

import torch.nn as nn
from torch.distributions import Categorical


class Player():

    def __init__(self, train_mode: bool=True, end_time: str=None):
        # super(Player, self).__init__()
        if USE_FLOAT64:
            torch.set_default_dtype(torch.float64)
        if end_time is None:
            end_time = STARTDAY
        self.sim = Simulator(to=end_time, duration=DURATION)

        self.device = torch.device(DEVICE)
        self.build_model()

        self.to()
        self.train_mode = train_mode
        self.obsDeque = []
        self.connect = redis.StrictRedis(host=REDIS_SERVER, port=6379)
        self.prev_embedding = [None, None, None, None]
        self.count = 0

    def build_model(self):
        self.prev_models = [baseAgent(PREV_AGENT_INFO) for i in range(4)]
        self.model = baseAgent(AGENT_INFO)
        self.model.evalMode()
        for m in self.prev_models:
            m.evalMode()

    def to(self):
        self.model.to(self.device)
        for m in self.prev_models:
            m.to(self.device)
    
    def forward(self, state:List[np.ndarray], step=0):
        # 4 * [time, midpoint, delta, acc_volume, coin]
        cell_state = None
        with torch.no_grad():
            cell_state = [m.getCellState() for m in self.prev_models]
            if USE_FLOAT64:
                state = [torch.unsqueeze(torch.tensor(i[2:].astype(np.float64)).to(self.device), 0) for i in state]
            else:
                state = [torch.unsqueeze(torch.tensor(i[2:].astype(np.float32)).float().to(self.device), 0) for i in state]
            output = []
            if step % 60 == 0:
                k = 4
            elif step % 15 == 0:
                k = 3 
            elif step % 5 == 0:
                k = 2
            else:
                k = 1
            
            prev_embedding = [self.prev_models[i].forward([torch.unsqueeze(state[i], 0)])[0] for i in range(k)]
            self.prev_embedding[:k] = prev_embedding
            output = self.model.forward(self.prev_embedding)[0]

            action_logit = output[:, 1:]
            action_prob = torch.softmax(action_logit, dim=-1)
            sampler = Categorical(probs=action_prob)
            action = sampler.sample()
        return action.numpy()[0], cell_state, action_prob.numpy()

    def reset_cell_state(self):

        for m in self.prev_models:
            m.zeroCellState()

    def process_traj(self, traj=None):
        if traj is None:
            traj = self.obsDeque
        traj = np.array(traj)
        state_idx = [3 * i for i in range(UNROLL_STEP+1)]
        action_idx = [3 * i + 1 for i in range(UNROLL_STEP+1)]
        reward_idx = [3 * i + 2 for i in range(UNROLL_STEP+1)]

        state_tmp = traj[state_idx]
        state = []
        policy = []
        for s in state_tmp:
            tmp = s[:4]
            tmp_ = []
            for t in tmp:
                tmp_.append(t[2:].astype(np.float64))
            tmp = np.array(tmp_)
            state.append(tmp)
            policy.append(s[4])
        s0 = state_tmp[0][5:]
        state = np.stack(state, axis=0)
        policy = np.stack(policy, axis=0)
        state = (state, s0, policy)
        action = traj[action_idx]
        reward = traj[reward_idx]
        done = traj[-1]
        return (state, action, reward, done)

    def pull_param(self):
        param = self.connect.get("param")
        count = self.connect.get("count")
        if count is not None:
            count = pickle.loads(count)
            if count > self.count:
                # print(count)
                param = pickle.loads(param)
                self.count = count

                self.model.load_state_dict(
                    param[0]
                )
                i = 1
                for m in self.prev_models:
                    m.load_state_dict(
                        param[i]
                    )
                    i += 1

    def run(self):

        while 1:
            self.sim.init_random()
            print(self.sim.to)
            prev_state = self.sim.reset()
            self.reset_cell_state()
            done = False
            step = 0
            cumulative_reward = 0
            traj = []
            while not done:
                self.pull_param()
                action, cell_state, policy = self.forward(prev_state, step)
                state, reward, done, info = self.sim.step(action, unit=RUN_UNIT_TIME)
                prev_state = deepcopy(state)
                cumulative_reward += reward
                if cumulative_reward < -1:
                    done = True
                if not done:
                    state += [policy[0]]
                    state += cell_state
                    step += 1
                    self.obsDeque += [state, action, reward]
                
                if step % 100 == 0:
                    self.pull_param()
                    # print(cumulative_reward)
                
                if done:
                    number_of_steps = UNROLL_STEP - int(len(self.obsDeque) / 3) + 1
                    prev_traj = self.prevDeque[-3*number_of_steps:]
                    traj = prev_traj + self.obsDeque
                    traj += [done]
                    traj = self.process_traj(traj)
                    self.connect.rpush("trajectory", pickle.dumps(traj))
                    self.obsDeque.clear()
                    print(cumulative_reward)
                    break

                if len(self.obsDeque) == 3 * (UNROLL_STEP + 1):
                    self.prevDeque = deepcopy(self.obsDeque)
                    self.obsDeque += [done]
                    x = self.process_traj()
                    self.connect.rpush("trajectory", pickle.dumps(x))
                    self.obsDeque.clear()
    
    def get_sample(self):

        dd = []
        self.sim.init_random()
        prev_state = self.sim.reset()
        self.reset_cell_state()
        done = False
        step = 0
        cumulative_reward = 0
        traj = []
        while not done:
            action, cell_state, policy = self.forward(prev_state, step)
            state, reward, _, info = self.sim.step(action, unit=RUN_UNIT_TIME)
            prev_state = deepcopy(state)
            state += [policy[0]]
            state += cell_state
            step += 1
            cumulative_reward += reward
            if cumulative_reward < -1:
                done = True
                reward = -1
            else:
                done = False

            if step % 100 == 0:
                self.pull_param()
                print(cumulative_reward)
            self.obsDeque += [state, action, reward]
            
            if done:
                number_of_steps = UNROLL_STEP - int(len(self.obsDeque) / 3) + 1
                prev_traj = self.prevDeque[-3*number_of_steps:]
                traj = prev_traj + self.obsDeque
                traj += [done]
                traj = self.process_traj(traj)
                dd.append(deepcopy(traj))
                self.obsDeque.clear()

            if len(self.obsDeque) == 3 * (UNROLL_STEP + 1):
                self.prevDeque = deepcopy(self.obsDeque)
                self.obsDeque += [done]
                x = self.process_traj()
                dd.append(deepcopy(x))
                self.obsDeque.clear()
            
            if len(dd) == 32:
                xx = pickle.dumps(dd)
                file = open("./sample.bin", 'wb')
                file.write(xx)
                file.close()
                break
