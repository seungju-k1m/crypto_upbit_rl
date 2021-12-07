from numpy.random.mtrand import rand
from configuration import *
from baseline.baseAgent import baseAgent
from collections import deque
from copy import deepcopy
from itertools import count

from Simulator.simulator_v1 import Simulator
import numpy as np
import _pickle as pickle
import torch
import redis

import torch.nn as nn
import random
import math
from PIL import Image as im



class LocalBuffer:

    def __init__(self):
        self.storage = []
    
    def push(self, s0, s1, a, r):
        s0_ = deepcopy(s0)
        s1_ = deepcopy(s1)
        a_ = deepcopy(a)
        r_ = deepcopy(r)
        self.storage += [s0_, s1_, a_, r_]
    
    def __len__(self):
        return int(len(self.storage) / 4)
    
    def get_traj(self, done=False):
        if done:
            traj = [self.storage[-4*UNROLL_STEP]]
            traj.append(self.storage[-4*UNROLL_STEP + 1])

            traj.append(self.storage[-4*UNROLL_STEP + 2])
            r = 0
            for i in range(UNROLL_STEP):
                r += (GAMMA ** i) * self.storage[-4*UNROLL_STEP + i*4 + 3]
            traj.append(r)
            traj.append(self.storage[-4])
            traj.append(self.storage[-3])
            traj += [done]
            traj_ = deepcopy(traj)
            self.storage.clear()
        else:
            traj = [self.storage[0]]
            traj.append(self.storage[1])
            traj.append(self.storage[2])
            r = 0
            for i in range(UNROLL_STEP):
                r += (GAMMA ** i) * self.storage[i*4 + 3]
            traj.append(r)
            traj.append(self.storage[4*UNROLL_STEP])
            traj.append(self.storage[4*UNROLL_STEP+1])
            traj += [done]
            traj_ = deepcopy(traj)
            # kk = np.random.choice([i+1 for i in range(UNROLL_STEP)], 1)[0]
            del self.storage[:4*UNROLL_STEP]
        return traj_
    
    def clear(self):
        self.storage.clear()


class Player():

    def __init__(self, idx=0):
        # super(Player, self).__init__()
        self.idx = idx

        self.sim = Simulator()

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
    
    def forward(self, state:list) -> int:
        
        # if step < TOTAL_TRAINING_STEP:
        #     epsilon = 1 - step * (1 - self.target_epsilon) / phase_01_random_step
        
        # elif step < phase_02_random_step:
        #     epsilon = 0.1 - (step - phase_01_random_step) / phase_02_random_step
        # else:
        #     epsilon = self.target_epsilon
        state, ratio = state
        epsilon = self.target_epsilon
        
        if random.random() < epsilon:
            action = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8])
            # action = random.choice([0, 1, 2])
        else:
            with torch.no_grad():
                state = np.expand_dims(state, axis=0)
                state = torch.tensor(state).float()
                state = state * (1/255.)

                ratio = torch.tensor(ratio).float()
                # ratio = torch.unsqueeze(ratio, dim=0)
                ratio = ratio.view(1, 1)
                
                # val, adv = self.model.forward([state])
                # action_value = val + adv - torch.mean(adv, dim=-1, keepdim=True)
                
                action_value = self.model.forward([state, ratio])[0]

                action = int(action_value.argmax(dim=-1).numpy())
                # print(action)
        return action, epsilon

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

    def calculate_priority(self, traj):
        with torch.no_grad():
            s0, s1, a, r, ns, ns1, d = traj

            s = torch.tensor([s0]).float().to(self.device)
            s = s/255.

            info_s = torch.tensor(s1).float().view(1, 1)


            s_ = torch.tensor([ns]).float().to(self.device)
            s_ = s_/255.

            next_info_s = torch.tensor([ns1]).float().to(self.device).view(1, 1)
            
            state_value = self.model.forward([s, info_s])[0][0]
            # val, adv = self.model.forward([s])
            # state_value = val + adv - torch.mean(adv, dim=-1, keepdim=True)
            current_state_value = float(state_value[a].detach().cpu().numpy())
            next_state_value = self.target_model.forward([s_, next_info_s])[0][0]
            # t_val, t_adv = self.target_model.forward([s_])
            # next_state_value = t_val + t_adv - torch.mean(t_adv, dim=-1, keepdim=True)
            d = float(not d)
            action = int(state_value.argmax().cpu().detach().numpy())
            max_next_state_value = float(next_state_value[action].cpu().detach().numpy()) * d
            td_error = r + (GAMMA)**UNROLL_STEP * max_next_state_value - current_state_value
            td_error = min(1, max(td_error, -1))
            x = (abs(td_error) + 1e-7) ** ALPHA
            
            return x

    def run(self):
        mean_cumulative_reward = 0
        per_episode = 2
        step = 0
        local_buffer = LocalBuffer()
        total_step = 0

        def preprocess_obs(obs):
            chart_info, account_info = obs
            image, candle_info = chart_info

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
            # self.sim.print()
            obs = preprocess_obs(obs)
            # obs
                # char info, account info
                # char info
                    # image, 

            action, _ = self.forward(obs)

            while done is False:

                next_obs, reward, done, info = self.sim.step(action)
                next_obs = preprocess_obs(next_obs)
                # self.sim.render()
                # reward = max(-1.0, min(reward, 1.0))
                step += 1
                total_step += 1

                cumulative_reward += reward
                local_buffer.push(obs[0], obs[1], action, reward)
                action, epsilon = self.forward(next_obs)
                obs = next_obs

                if done:
                    local_buffer.push(obs[0], obs[1], 0, 0)

                if len(local_buffer) == 2 * UNROLL_STEP or done:
                    experience = local_buffer.get_traj(done)

                    priority = self.calculate_priority(experience)
                    experience.append(priority)

                    self.connect.rpush(
                        "experience",
                        pickle.dumps(experience)
                    )

                if step %  20 == 0:
                    self.pull_param()
            mean_cumulative_reward += (math.exp(cumulative_reward/100) - 1)
            # self.sim.print()

            if (t+1) % per_episode == 0:
                print("""
                EPISODE:{} // YIELD:{:.3f} // EPSILON:{:.3f} // COUNT:{} // T_Version:{}
                """.format(t+1, mean_cumulative_reward / per_episode, epsilon, self.count, self.target_model_version))
                self.connect.rpush(
                    "reward", pickle.dumps(
                        mean_cumulative_reward / per_episode
                    )
                )
                mean_cumulative_reward = 0

    def eval(self):
        obsDeque = deque(maxlen=4)
        mean_cumulative_reward = 0
        per_episode = 10
        step = 0
        keys = ['ale.lives', 'lives']
        key = "ale.lives"
        self.pull_param()
        # key = "lives"
        def rgb_to_gray(img, W=84, H=84):
            grayImage = im.fromarray(img, mode="RGB").convert("L")

            # grayImage = np.expand_dims(Avg, -1)
            grayImage = grayImage.resize((84, 84), im.NEAREST)
            return grayImage
        
        def stack_obs(img):
            gray_img = rgb_to_gray(img)
            obsDeque.append(gray_img)
            state = []
            if len(obsDeque) > 3:
                for i in range(4):
                    state.append(obsDeque[i])
                state = np.stack(state, axis=0)
                return state
            else:
                return None
        
        for t in count():
            cumulative_reward = 0   
            done = False
            live = -1
            experience = []
            step = 0

            obs = self.sim.reset()
            self.obsDeque.clear()
            for i in range(4):
                stack_obs(obs)
            
            state = stack_obs(obs)

            action, _ = self.forward(state)
            
            while done is False:
                reward = 0
                for i in range(3):
                    _, __, r, _ = self.sim.step(action)
                    reward += r
                next_obs, r, done, info = self.sim.step(action)
                reward += r
                reward_ = reward
                reward = max(-1.0, min(reward, 1.0))
                step += 1
                # print(step)
                # self.sim.render()

                if live == -1:
                    try:
                        live = info[key]
                    except:
                        key = keys[1 - keys.index(key)]
                        live = info[key]
                
                if info[key] != 0:
                    _done = live != info[key]
                    if _done:
                        live = info[key]
                else:
                    _done = reward != 0

                state = stack_obs(next_obs)
                action, epsilon = self.forward(state, t, no_epsilon=True)
                cumulative_reward += reward_
            mean_cumulative_reward += cumulative_reward
            if (t+1) % per_episode == 0:
                print("""
                EPISODE:{} // REWARD:{:.3f}
                """.format(t+1, mean_cumulative_reward / per_episode))
                mean_cumulative_reward = 0
                break