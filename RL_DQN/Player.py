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

from Simulator.simulator import Simulator
from typing import List
import numpy as np
import _pickle as pickle
import torch
import redis

import torch.nn as nn
from torch.distributions import Categorical
import gym
import random
from PIL import Image as im



class LocalBuffer:

    def __init__(self):
        self.storage = []
    
    def push(self, s, a, r):
        self.storage += [s, a, r]
    
    def __len__(self):
        return int(len(self.storage) / 3)
    
    def get_traj(self, done=False):
        traj = [self.storage[-3*UNROLL_STEP]]
        traj.append(self.storage[-3*UNROLL_STEP+1])
        r = 0
        for i in range(UNROLL_STEP):
            r += (GAMMA ** i) * self.storage[-3*UNROLL_STEP + i*3 + 2]
        traj.append(r)
        traj.append(self.storage[-3])
        traj += [done]
        traj_ = deepcopy(traj)
        kk = np.random.choice([i+1 for i in range(UNROLL_STEP)], 1)[0]
        del self.storage[:3*kk]
        return traj_
    
    def clear(self):
        self.storage.clear()


class Player():

    def __init__(self, idx=0, train_mode: bool=True, end_time: str=None):
        # super(Player, self).__init__()
        self.idx = idx
        if end_time is None:
            end_time = STARTDAY
        if USE_GYM:
            self.sim = gym.make("PongNoFrameskip-v4")
            zz = np.random.choice([i for i in range(idx*21+13)], 1)[0]
            self.sim.seed(int(zz))
            # self.sim = gym.make('SpaceInvadersNoFrameskip-v4')
        else:
            self.sim = Simulator(to=end_time, duration=DURATION)

        self.device = torch.device(DEVICE)
        self.build_model()
        self.target_epsilon =  0.4 **(1 + 7 * self.idx / (N-1))

        self.to()
        self.train_mode = train_mode
        self.obsDeque = []
        self.connect = redis.StrictRedis(host=REDIS_SERVER, port=6379)
        self.prev_embedding = [None, None, None, None]
        self.count = 0
        self.target_model_version = -1
        if USE_GYM:
            self.forward = self.gym_forward
            self.run = self.gym_run
        else:
            self.forward = self.bit_forward
            self.run = self.bit_run

    def build_model(self):
        info = DATA["model"]
        self.model = baseAgent(info)
        self.target_model = baseAgent(info)

    def to(self):
        self.model.to(self.device)
        self.target_model.to(self.device)
    
    def bit_forward(self, state:List[np.ndarray], step=0):
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

    def gym_forward(self, state:np.ndarray, step=0, no_epsilon=False, random_action=False) -> int:
        
        phase_01_random_step = TOTAL_TRAINING_STEP
        phase_02_random_step = 1e7
        step = step
        # if step < TOTAL_TRAINING_STEP:
        #     epsilon = 1 - step * (1 - self.target_epsilon) / phase_01_random_step
        
        # elif step < phase_02_random_step:
        #     epsilon = 0.1 - (step - phase_01_random_step) / phase_02_random_step
        # else:
        #     epsilon = self.target_epsilon
        
        epsilon = self.target_epsilon
        if random_action:
            epsilon = 1
        
        if random.random() < epsilon:
            action = random.choice([0, 1, 2, 3, 4, 5])
        else:
            with torch.no_grad():
                state = np.expand_dims(state, axis=0)
                state = torch.tensor(state).float()
                state = state * (1/255.)
                
                val, adv = self.model.forward([state])
                action_value = val + adv - torch.mean(adv, dim=-1, keepdim=True)
                action = int(action_value.argmax(dim=-1).numpy())
                # print(action)
        return action, epsilon

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
       
        count = self.connect.get("count")
        if count is not None:
            count = pickle.loads(count)
            target_version = int(count / TARGET_FREQUENCY)
            if target_version > self.target_model_version:
                t_param = self.connect.get("target_state_dict")
                if t_param is None:
                    return
                t_param = pickle.loads(t_param)
                self.target_model_version = target_version
                self.target_model.load_state_dict(t_param)
                
            if count > self.count:
                param = self.connect.get("state_dict")
                if param is None:
                    return
                param = pickle.loads(param)
                self.count = count

                self.model.load_state_dict(
                    param
                )

    def bit_run(self):

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
                if self.sim.coin_account < 0.1:
                    if self.sim.krw_account < 1000000:
                        done = True
                if not done:
                    state += [policy[0]]
                    state += cell_state
                    step += 1
                    self.obsDeque += [state, action, reward]
                
                if step % 100 == 0:
                    self.pull_param()
                    # print(cumulative_reward)
                if step % 1440 == 0:
                    print(self.sim.coin_account)
                    print(self.sim.krw_account)

                
                if done:
                    number_of_steps = UNROLL_STEP - int(len(self.obsDeque) / 3) + 1
                    prev_traj = self.prevDeque[-3*number_of_steps:]
                    traj = prev_traj + self.obsDeque
                    traj += [False]
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

    def calculate_priority(self, traj):
        with torch.no_grad():
            s, a, r, s_, d = traj
            s = torch.tensor([s]).float().to(self.device)
            s = s/255.
            s_ = torch.tensor([s_]).float().to(self.device)
            s_ = s_/255.

            # state_value = self.model.forward([s])[0][0]
            val, adv = self.model.forward([s])
            state_value = val + adv - torch.mean(adv, dim=-1, keepdim=True)
            
            current_state_value = float(state_value[0].detach().cpu().numpy()[a])
            # next_state_value = self.target_model.forward([s_])[0][0]
            t_val, t_adv = self.target_model.forward([s_])
            next_state_value = t_val + t_adv - torch.mean(t_adv, dim=-1, keepdim=True)

            action = int(state_value.argmax().cpu().detach().numpy())
            max_next_state_value = float(next_state_value[0][action].cpu().detach().numpy())
            td_error = r + (GAMMA)**UNROLL_STEP * max_next_state_value - current_state_value
            td_error = min(1, max(td_error, -1))
            x = (abs(td_error) + 1e-4) ** ALPHA
            
            return x
        
    def gym_run(self):
        obsDeque = deque(maxlen=4)
        mean_cumulative_reward = 0
        per_episode = 2
        step = 0
        local_buffer = LocalBuffer()
        keys = ['ale.lives', 'lives']
        key = "ale.lives"
        random_action=False
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
        
        total_step = 0

        for t in count():
            cumulative_reward = 0   
            done = False
            live = -1
            experience = []
            local_buffer.clear()
            step = 0

            obs = self.sim.reset()
            self.obsDeque.clear()
            for i in range(4):
                stack_obs(obs)
            
            state = stack_obs(obs)

            action, _ = self.forward(state, random_action=random_action)
            
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
                total_step += 1
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
                
                next_state = stack_obs(next_obs)
                cumulative_reward += reward_

                # experience += deepcopy([state, action, reward, next_state])

                local_buffer.push(state, action, reward)
                action, epsilon = self.forward(next_state, total_step, random_action=random_action)
                state = next_state

                if random_action:
                    check_learning_start = self.connect.get("Start")
                    if check_learning_start is not None:
                        aa = pickle.loads(check_learning_start)
                        if aa:
                            random_action = False
                    # self.connect.delete("Start")

                if len(local_buffer) ==  2 * UNROLL_STEP or done:
                    experience = local_buffer.get_traj(done)

                    priority = self.calculate_priority(experience)
                    experience.append(priority)

                    self.connect.rpush(
                        "experience",
                        pickle.dumps(experience)
                    )

                if step %  100 == 0:
                    self.pull_param()
            mean_cumulative_reward += cumulative_reward

            if (t+1) % per_episode == 0:
                print("""
                EPISODE:{} // REWARD:{:.3f} // EPSILON:{:.3f} // COUNT:{} // T_Version:{}
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