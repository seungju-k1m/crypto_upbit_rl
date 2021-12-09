from RL_R2D2.Player import FIXED_TRAJECTORY
from configuration import *
from baseline.baseAgent import baseAgent
from baseline.utils import getOptim, writeTrainInfo
from RL_DQN.ReplayMemory import Replay, Replay_Server

from torch.utils.tensorboard import SummaryWriter
from itertools import count

import numpy as np
import torch.nn as nn
import torch
import time
import gc

import redis
import _pickle as pickle

import cProfile


class Learner:
    def __init__(self):
        self.device = torch.device(LEARNER_DEVICE)
        self.build_model()
        self.build_optim()
        self.connect = redis.StrictRedis(host=REDIS_SERVER, port=6379)

        if USE_REDIS_SERVER:
            self.memory = Replay_Server()
        else:
            self.memory = Replay()

        # self.memory = Replay()s
        self.memory.start()
        
        self.writer = SummaryWriter(
            LOG_PATH
        )
        
        names = self.connect.scan()
        if len(names[-1]) > 0:
            self.connect.delete(*names[-1])

    def build_model(self):
        info = DATA["model"]
        self.model = baseAgent(info)
        self.model.to(self.device)
        self.target_model = baseAgent(info)
        self.target_model.to(self.device)
    
    def build_optim(self):
        self.optim = getOptim(OPTIM_INFO, self.model)
        
    def train(self, transition, t=0) -> dict:
        new_priority = None

        hidden_state, state, action, reward, done, weight, idx = transition
        # state -> SEQ * BATCH
        weight = torch.tensor(weight).float().to(self.device)

        hidden_state_0 = hidden_state[0].to(self.device)
        hidden_state_1 = hidden_state[1].to(self.device)

        self.model.setCellState((hidden_state_0, hidden_state_1))
        self.target_model.setCellState((hidden_state_0, hidden_state_1))

        state = torch.tensor(state).to(self.device).float()
        state = state / 255.
        # state = state.to(self.device)

        shape = torch.tensor([80, BATCHSIZE, -1])

        # action = torch.tensor(action).long().to(self.device)
        action = np.transpose(action, (1, 0))
        action = action[:-1]
        action = action.reshape(-1)

        action = [6 * i + a for i, a in enumerate(action)]

        reward= reward.astype(np.float32)
        reward = np.transpose(reward, (1, 0))

        action_value = self.model.forward([state, shape])[0].view(-1)
        # 320 * 6
        selected_action_value = action_value[action]

        detach_action_value = action_value.detach()
        detach_action_value = detach_action_value.view(-1, 6)
        # val, adv = self.model.forward([state])
        # action_value = val + adv - torch.mean(adv, dim=-1, keepdim=True)
        

        with torch.no_grad():
            
            target_action_value = self.target_model.forward([state, shape])[0]
            target_action_value = target_action_value.view(-1)
            action_max = detach_action_value.argmax(-1)
            action_idx = [6 * i + j for i, j in enumerate(action_max)]
            target_action_max_value = target_action_value[action_idx]

            next_max_value =  target_action_max_value
            next_max_value = next_max_value.view(80, BATCHSIZE)

            target_value = next_max_value[UNROLL_STEP+1:].contiguous()

            rewards = np.zeros((80 - UNROLL_STEP - 1, BATCHSIZE))
            bootstrap = next_max_value[-1].detach().cpu().numpy()

            remainder = [bootstrap * done]

            for i in range(UNROLL_STEP):
                rewards += GAMMA ** i * reward[i:80 - UNROLL_STEP-1+i]
                remainder.append(
                    reward[-(i+1)] + GAMMA * remainder[i]
                )
            
            rewards = torch.tensor(rewards).float().to(self.device)
            remainder = remainder[::-1]
            remainder.pop()
            remainder = torch.tensor(remainder).float().to(self.device)
            # remainder = torch.cat(remainder)

            target = rewards + GAMMA * UNROLL_STEP * target_value
            target = torch.cat((target, remainder), 0)
            target = target.view(-1)


            # next_max_value, _ = next_action_value.max(dim=-1) 
            # next_max_value = next_max_value * done
            

        td_error_ = target - selected_action_value

        td_error = torch.clamp(td_error_, -1, 1)


        td_error_for_prior = td_error.detach().cpu().numpy()

        # td_error_for_prior = (np.abs(td_error_for_prior) + 1e-7) ** ALPHA
        new_priority = td_error_for_prior

        td_error_view = td_error.view(79, -1)
        td_error_truncated = td_error_view[20:].contiguous()

        td_error_truncated = td_error_truncated.permute(1, 0)
        weight = weight.view(-1, 1)

        loss = torch.mean(
            weight * (td_error_truncated ** 2)
        ) * 0.5

        loss.backward()

        info = self.step()

        info['mean_value'] = float(target.mean().detach().cpu().numpy())           
        return info, new_priority, idx

    def step(self):
        p_norm = 0
        pp = []
        with torch.no_grad():
            pp += self.model.getParameters()
            for p in pp:
                p_norm += p.grad.data.norm(2)
            p_norm = p_norm ** .5
        # torch.nn.utils.clip_grad_norm_(pp, 40)
        # for optim in self.optim:
        #     optim.step()
        self.optim.step()
        self.optim.zero_grad()
        info = {}
        info['p_norm'] = p_norm.cpu().numpy()
        return info

    def run(self):
        def wait_memory():
            while True:
                if USE_REDIS_SERVER:
                    cond = self.connect.get("FLAG_BATCH")
                    if cond is not None:
                        cond = pickle.loads(cond)
                        if cond:
                            break
                else:
                    if len(self.memory.memory) > 50000:
                        break
                    else:
                        print(len(self.memory.memory))
                time.sleep(1)
        wait_memory()
        state_dict = pickle.dumps(self.state_dict)
        step_bin = pickle.dumps(1)
        target_state_dict = pickle.dumps(self.target_state_dict)
        self.connect.set("state_dict", state_dict)
        self.connect.set("count", step_bin)
        self.connect.set("target_state_dict", target_state_dict)
        self.connect.set("Start", pickle.dumps(True))
        print("Learning is Started !!")
        step, norm, mean_value = 0, 0, 0
        amount_sample_time, amount_train_tim, amount_update_time = 0, 0, 0
        init_time = time.time()
        mm = 25
        mean_weight = 0
        for t in count():
            time_sample = time.time()

            experience = self.memory.sample()

            if experience is False:
                time.sleep(0.002)
                continue

            amount_sample_time += (time.time() - time_sample)
            # -----------------

            # ------train---------
            tt = time.time()
            step += 1
            if step == 1:
                profile = cProfile.Profile()
                profile.runctx('self.train(experience)', globals(), locals())
                profile.print_stats()
            info, priority, idx  = self.train(experience)
            amount_train_tim += (time.time() - tt)
            mean_weight += 0
            # -----------------

            # ------Update------
            tt = time.time()
            
            if (step % 500) == 0:
                
                if USE_REDIS_SERVER:
                    self.connect.set("FLAG_REMOVE", pickle.dumps(True))
                else:
                    self.memory.lock = True
            
            if USE_PER:
                if USE_REDIS_SERVER:
                    self.memory.update(
                        list(idx), priority
                    )
                else:
                    if self.memory.lock is False:
                        self.memory.update(
                            list(idx), priority
                        )

            norm += info['p_norm']
            mean_value += info['mean_value']

            # target network updqt
            # soft
            # self.target_model.updateParameter(self.model, 0.005)
            # hard

            if step % TARGET_FREQUENCY == 0:
                self.target_model.updateParameter(self.model, 1)
                target_state_dict = pickle.dumps(self.target_state_dict)
                self.connect.set("target_state_dict", target_state_dict)

            if step % 25 == 0:
                state_dict = pickle.dumps(self.state_dict)
                step_bin = pickle.dumps(step-50)
                self.connect.set("state_dict", state_dict)
                self.connect.set("count", step_bin)
            amount_update_time += (time.time() - tt)
            
            if step % mm == 0:
                pipe = self.connect.pipeline()
                pipe.lrange("reward", 0, -1)
                pipe.ltrim("reward", -1, 0)
                data = pipe.execute()[0]
                self.connect.delete("reward")
                cumulative_reward = 0
                if len(data) > 0:
                    for d in data:
                        cumulative_reward += pickle.loads(d)
                    cumulative_reward /= len(data)
                else:
                    cumulative_reward = -21
                amount_sample_time /= mm
                amount_train_tim /= mm
                amount_update_time /= mm
                tt = time.time() - init_time
                init_time = time.time()
                if USE_REDIS_SERVER:
                    print(
                        """step:{} // mean_value:{:.3f} // norm:{:.3f} // REWARd:{:.3f}
                        TIME:{:.3f} // TRAIN_TIME:{:.3f} // SAMPLE_TIME:{:.3f} // UPDATE_TIME:{:.3f}""".format(
                            step, mean_value / mm, norm / mm, cumulative_reward, tt/mm, amount_train_tim, amount_sample_time, amount_update_time
                        )
                    )
                else:
                    print(
                        """step:{} // mean_value:{:.3f} // norm: {:.3f} // REWARD:{:.3f} // NUM_MEMORY:{} 
        Mean_Weight:{:.3f}  // MAX_WEIGHT:{:.3f}  // TIME:{:.3f} // TRAIN_TIME:{:.3f} // SAMPLE_TIME:{:.3f} // UPDATE_TIME:{:.3f}""".format(
                            step, mean_value / mm, norm / mm, cumulative_reward, len(self.memory.memory), mean_weight / mm, self.memory.memory.max_weight,tt / mm, amount_train_tim, amount_sample_time, amount_update_time)
                    )
                amount_sample_time, amount_train_tim, amount_update_time = 0, 0, 0
                if len(data) > 0:
                    self.writer.add_scalar(
                        "Reward", cumulative_reward, step
                    )
                self.writer.add_scalar(
                    "value", mean_value / mm, step
                )
                self.writer.add_scalar(
                    "norm", norm/ mm, step
                )
                mean_value, norm = 0, 0
                mean_weight = 0
                torch.save(self.state_dict, './weight/dqn/weight.pth')
    
    @property
    def state_dict(self):
        state_dict = {k:v.cpu() for k, v in self.model.state_dict().items()}
        return state_dict
    
    @property
    def target_state_dict(self):
        target_state_dict = {k:v.cpu() for k, v in self.target_model.state_dict().items()}
        return target_state_dict
    