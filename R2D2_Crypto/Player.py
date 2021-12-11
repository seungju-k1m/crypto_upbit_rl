from threading import local
from numpy.core.fromnumeric import mean
from numpy.core.shape_base import stack
from numpy.random.mtrand import rand
from configuration import *
from baseline.baseAgent import baseAgent
from multiprocessing import Process
from collections import deque
from copy import deepcopy
from itertools import count

from typing import List
import numpy as np
import _pickle as pickle
import torch
import redis

import torch.nn as nn
from torch.distributions import Categorical
from Simulator.simulator_v1 import LEVERAGE, Simulator

import time
import random
from PIL import Image as im



class LocalBuffer:

    def __init__(self):
        self.storage = []
        self.hidden_state = []
    
    def push(self, s0, s1, a, r):
        s_ = deepcopy(s0)
        s1_ = deepcopy(s1)
        a_ = deepcopy(a)
        r_ = deepcopy(r)
        self.storage += [s_, s1_, a_, r_]
    
    def push_hidden_state(self, h):
        h_ = deepcopy(h)
        self.hidden_state.append(h_)
    
    def __len__(self):
        return int(len(self.storage) / 4)
    
    def get_traj(self, done=False):
        if done:
            traj_ = []
            traj_.append(
                self.hidden_state[-FIXED_TRAJECTORY]
            )
            traj_ += deepcopy(
                self.storage[-4*FIXED_TRAJECTORY:]
            )
            traj_.append(done)
            self.storage.clear()
            self.hidden_state.clear()
        else:
            traj_ = []
            traj_.append(deepcopy(
                self.hidden_state[0]
            ))
            traj_ += deepcopy(
                self.storage[:4*FIXED_TRAJECTORY]
            )
            traj_.append(done)

            # kk = np.random.choice([i+1 for i in range(UNROLL_STEP)], 1)[0]
            del self.storage[:4*int(FIXED_TRAJECTORY/2)]
            del self.hidden_state[:int(FIXED_TRAJECTORY/2)]
        return np.array(traj_)
    
    def clear(self):
        self.storage.clear()
        self.hidden_state.clear()


class Player():

    def __init__(self, idx=0):
        # super(Player, self).__init__()
        self.idx = idx
        self.sim = Simulator(unit_step=15, size=48, day=7)
        self.device = torch.device(DEVICE)
        self.build_model()
        self.target_epsilon =  0.4 **(1 + 7 * self.idx / (N-1))

        self.to()

        self.connect = redis.StrictRedis(host=REDIS_SERVER, port=6379)

        self.count = 0
        self.target_model_version = -1

    def build_model(self):
        info = DATA["model"]
        self.model = baseAgent(info)
        self.target_model = baseAgent(info)

    def to(self):
        self.model.to(self.device)
        self.target_model.to(self.device)
    
    def pull_param(self):
       
        count = self.connect.get("count")
        if count is not None:
            count = pickle.loads(count)
            target_version = int(count / TARGET_FREQUENCY)
            t_param = self.connect.get("target_state_dict")
            if t_param is None:
                return
            t_param = pickle.loads(t_param)
            self.target_model_version = target_version
            self.target_model.load_state_dict(t_param)
            param = self.connect.get("state_dict")
            if param is None:
                return
            param = pickle.loads(param)
            self.count = count

            self.model.load_state_dict(
                param
            )

    def load_data(self):
        path='./weight/r2d2_cpt/weight.pth'

        self.model.load_state_dict(
            torch.load(path)
        )

    def eval(self):
        mean_cumulative_reward = 0
        mean_yield = 0
        per_episode = 1
        step = 0
        total_step = 0
        self.target_epsilon = 0
        # self.pull_param()
        self.load_data()
        action_count = np.array([0 for i in range(ACTION_SIZE)])

        def preprocess_obs(obs):
            chart_info, account_info = obs
            image, candle_info = chart_info

            value = account_info['Current_Value']
            KRW_value = account_info['KRW_Balance']
            # coin_value = value - KRW_value
            ratio = KRW_value / value
            return (image, ratio)
        raw_yield = 0
        for t in count():
            cumulative_reward = 0   
            done = False
            mean_yield = 0
            step = 0
            num_idle = 0
            # try:
            #     port = deepcopy(self.sim.portfolio)
            #     obs = self.sim.reset(True, port=port)
            # except:
            obs = self.sim.reset(True)
            self.model.zeroCellState()
            # self.sim.print()
            obs = preprocess_obs(obs)
            # obs
                # char info, account info
                # char info
                    # image, 

            action, _, __ = self.forward(obs)
            
            print('--------------')
            self.sim.portfolio.print()

            while done is False:
                next_obs, reward, done, idle = self.sim.step(action)
                # info 현재 수익률 
                # reward -> 100 * log(current_value/prev_value)
                action_count[action] += 1
                next_obs = preprocess_obs(next_obs)
                # self.sim.render()
                # reward = max(-1.0, min(reward, 1.0))
                step += 1
                total_step += 1

                cumulative_reward += reward
                if idle:
                    raw_yield += reward / (1 - FEE)
                    num_idle += 1
                else:
                    raw_yield += reward
                action, _, epsilon = self.forward(next_obs)
                if step% 240 == 0:
                    self.sim.portfolio.print()
                
            mean_yield += (math.exp(raw_yield/LEVERAGE) - 1)
            self.sim.print()

            print('--------------------')

            if (t+1) % per_episode == 0:
                print("""
                EPISODE:{} // YIELD:{:.3f} // EPSILON:{:.3f} // COUNT:{} // T_Version:{}
                """.format(t+1, mean_yield / per_episode, epsilon, self.count, self.target_model_version))
                mean_yield = 0
                print(action_count)
                print(num_idle)
                num_idle = 0
                action_count = np.zeros(ACTION_SIZE)
            
            if(t+1) == 25:
                break

    def forward(self, state:np.ndarray, no_epsilon=False) -> int:
        # image, (ratio), shape01, shape02
        
        epsilon = self.target_epsilon

        state, ratio = state

        self.model.detachCellState()
        hidden_state = self.model.getCellState()
        with torch.no_grad():
            state = np.expand_dims(state, axis=0)
            state = torch.tensor(state).float()
            state = state * (1/255.)

            ratio = torch.tensor(ratio).float()
            ratio = ratio.view(-1, 1)
            
            shape = torch.tensor([1, 1, -1])
            action_value = self.model.forward([state, ratio, shape])[0]

        if no_epsilon:
            epsilon = 0
        
        if random.random() < epsilon:
            action = random.choice([i for i in range(ACTION_SIZE)])
        else:
            action = int(action_value.argmax(dim=-1).numpy())
                    # print(action)
        return action, hidden_state, epsilon

    def calculate_priority(self, traj):
        # traj, hidden_state, (s, a, r) * 80, done
        prev_hidden_state = self.model.getCellState()
        with torch.no_grad():
            hidden_state = traj[0]
            done = traj[-1]
            done = float(not done)
            state_idx =[1 + 4 * i for i in range(FIXED_TRAJECTORY)]
            ratio_idx = [2 + 4 * i for i in range(FIXED_TRAJECTORY)]
            action_idx = [3 + 4 * i for i in range(FIXED_TRAJECTORY)]
            reward_idx = [4 + 4 * i for i in range(FIXED_TRAJECTORY)]

            state = traj[state_idx]
            state = [np.uint8(s) for s in state]
            state = np.stack(state, 0)

            ratio = traj[ratio_idx]
            ratio = ratio.astype(np.float32)

            action = traj[action_idx]
            action = action.astype(np.int32)
            reward = traj[reward_idx]
            reward = reward.astype(np.float32)

            # 80개중 실제로 훈련에 쓰이는 것은 80개 중 79개.
            
            state = torch.tensor(state).float()
            state = state / 255.
            state = state.to(self.device)

            ratio = torch.tensor(ratio).float()
            ratio = ratio.view(-1, 1)

            self.model.setCellState(hidden_state)
            self.target_model.setCellState(hidden_state)

            shape = torch.tensor([FIXED_TRAJECTORY, 1, -1])
            online_action_value = self.model.forward([state, ratio, shape])[0]
            target_action_value = self.target_model.forward([state, ratio, shape])[0].view(-1)

            action_max = online_action_value.argmax(-1).numpy()
            action_max_idx = [ACTION_SIZE * i + j for i, j in enumerate(action_max)]

            target_max_action_value = target_action_value[action_max_idx]
            action_value = online_action_value.view(-1)[action]

            bootstrap = float(target_max_action_value[-1].numpy())

            target_value = target_max_action_value[UNROLL_STEP+1:]

            # 80 - 5 - 1 = 74
            rewards = np.zeros((FIXED_TRAJECTORY - UNROLL_STEP - 1))

            # 5
            remainder = [bootstrap * done]

            for i in range(UNROLL_STEP):
                # i -> 4
                rewards += GAMMA ** i * reward[i:FIXED_TRAJECTORY - UNROLL_STEP-1 + i]
                remainder.append(
                    reward[-(i+1)] + GAMMA * remainder[i]
                )
            rewards = torch.tensor(rewards).float().to(self.device)
            remainder = remainder[::-1]
            remainder.pop()
            remainder = torch.tensor(remainder).float().to(self.device)
            target = rewards + GAMMA * UNROLL_STEP * target_value
            target = torch.cat((target, remainder), 0)

            td_error = abs((target - action_value[:-1]))

            weight = td_error.max() * 0.9 + 0.1 * td_error.mean()
            weight = weight ** ALPHA
        
        self.model.setCellState(prev_hidden_state)
            
        return float(weight.cpu().numpy())
        
    def run(self):
        
        mean_cumulative_reward = 0
        mean_yield = 0
        per_episode = 2
        step = 0
        local_buffer = LocalBuffer()
        total_step = 0

        def preprocess_obs(obs):
            chart_info, account_info = obs
            image, _= chart_info

            value = account_info['Current_Value']
            KRW_value = account_info['KRW_Balance']
            # coin_value = value - KRW_value
            ratio = KRW_value / value
            return (image, ratio)

        for t in count():
            cumulative_reward = 0   
            done = False
            experience = []
            local_buffer.clear()
            step = 0

            obs = self.sim.reset()
            state = preprocess_obs(obs)

            self.model.zeroCellState()
            self.target_model.zeroCellState()

            action, hidden_state, _ = self.forward(state)
            raw_yield = 0

            while done is False:

                next_obs, reward, done, raw_reward = self.sim.step(action)

                next_state = preprocess_obs(next_obs)

                step += 1
                total_step += 1

                cumulative_reward += reward
                raw_yield += raw_reward
                local_buffer.push(state[0], state[1], action, reward)
                local_buffer.push_hidden_state(hidden_state)

                action, next_hidden_state, epsilon = self.forward(next_state)
                state = next_state
                hidden_state = next_hidden_state

                if done:
                    local_buffer.push(state[0], state[1], 0, 0)
                    local_buffer.push_hidden_state(hidden_state)

                if len(local_buffer) == int(FIXED_TRAJECTORY * 1.5) or done:
                    experience = local_buffer.get_traj(done)

                    priority = self.calculate_priority(experience)
                    experience = np.append(experience, priority)

                    self.connect.rpush(
                        "experience",
                        pickle.dumps(experience)
                    )

                if total_step %  400 == 0:
                    self.pull_param()
            mean_cumulative_reward += cumulative_reward

            mean_yield += (math.exp(raw_yield/100) - 1)

            if (t+1) % per_episode == 0:
                print("""
                EPISODE:{} // YIELD:{:.3f} // EPSILON:{:.3f} // COUNT:{} // T_Version:{}
                """.format(t+1, mean_cumulative_reward / per_episode, epsilon, self.count, self.target_model_version))
                if self.target_epsilon < 0.05:
                    self.connect.rpush(
                        "reward", pickle.dumps(
                            mean_yield / per_episode
                        )
                    )
                mean_cumulative_reward = 0
                mean_yield = 0

         