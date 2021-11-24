from configuration import *
from baseline.baseAgent import baseAgent
from baseline.utils import getOptim, writeTrainInfo
from RL.ReplayMemory import Replay

from itertools import count

import numpy as np
import torch.nn as nn
import torch
import time
import gc

import redis
import _pickle as pickle


class Learner:
    def __init__(self):
        self.device = torch.device(LEARNER_DEVICE)
        self.build_model()
        self.build_optim()
        self.connect = redis.StrictRedis(host=REDIS_SERVER, port=6379)
        self.memory = Replay()
        self.memory.start()
        
        names = self.connect.scan()
        if len(names[-1]) > 0:
            self.connect.delete(*names[-1])

    def build_model(self):
        self.prev_models = [baseAgent(LEARNER_AGENT_INFO) for i in range(4)]
        self.model = baseAgent(AGENT_INFO)
        self.model.to(self.device)
        self.init_parameters = []
        for m in self.prev_models:
            m.to(self.device)
    
    def build_optim(self):
        optim_info = None
        self.optim = []
        self.optim.append(
            getOptim(OPTIM_INFO, self.model)
        )
        for m in self.prev_models:
            self.optim.append(
                getOptim(OPTIM_INFO, m)
            )
        
    def train(self, transition, t=0):
        state_, action, reward, done = transition
        done = torch.tensor([float(not d) for d in done]).to(self.device)
        # BATCH * UNROLL_STEP * 2
        ACTIONSIZE = 2

        reward = torch.tensor(reward).to(self.device)
        action_idx = np.concatenate([(UNROLL_STEP+1) * i +  np.arange(0, UNROLL_STEP+1) for i in range(32)], 0) * 2
        action_idx += action.reshape(-1).astype(action_idx.dtype)
        state, cell_state, old_policy = state_
        old_policy = old_policy.view(-1)[action_idx]
        # m1_s, m5_s, m15_s, m60_s = state
        # m1_c, m5_c, m15_c, m60_c = cell_state
        idx = [1, 5, 15, 60]
        embedding = []
        SEQ, BATCH, _ = state[0].shape
        for c, s, m, i in zip(cell_state, state, self.prev_models, idx):
            c = c.detach()
            m.setCellState(c)
            k = m.forward([s])[0]
            k = k.view(-1, 32, 16)
            if USE_BPTT:
                if i == 1:
                    k_split = [10 * (i+1) for i in range(17)]
                    k[k_split] = k[k_split].detach()
                if i == 5:
                    k_split = [6 * (i+1) for i in range(5)]
                    k[k_split] = k[k_split].detach()
            # -1, 16
            # k = k.view(-1, 32)
            # SEQ, BATCH, 32
            if i == 1:
                k = k.view(-1, 16)
                embedding.append(k)
            else:
                kk = torch.cat([k for i in range(i)], -1)
                kk = kk.view(-1, 16)
                embedding.append(kk)
        
        output = self.model.forward(embedding)[0].view(SEQ, BATCH, -1).permute(1, 0, 2).contiguous()

        value = output[:, :, 0]
        policy_logit = output[:, :, 1:].contiguous()
        policy = torch.softmax(policy_logit, dim=-1)
        view_policy = policy.view(-1, 2)
        entropy = - torch.sum(torch.sum(view_policy * torch.log(view_policy), 1))
        selected_policy = policy.view(-1)[action_idx]
        selected_log_policy = torch.log(selected_policy)

        detach_policy_logit = policy_logit.detach()
        detach_value = value.detach()
        DISCOUNT = 0.99
        RHO_C, RHO_P = 1.0, 1.0 

        RHO_C = torch.tensor(RHO_C).to(self.device)
        RHO_P = torch.tensor(RHO_P).to(self.device)

        with torch.no_grad():
            detach_policy = torch.softmax(detach_policy_logit, dim=-1)
            detach_selected_policy = detach_policy.view(-1)[action_idx]
            log_ratio = torch.log(detach_selected_policy) - torch.log(old_policy)
            ratio = torch.exp(log_ratio)
            cliped_c = torch.min(RHO_C, ratio).view(BATCH, UNROLL_STEP+1)
            cliped_p = torch.min(RHO_P, ratio).view(BATCH, UNROLL_STEP+1)
            mean_value = detach_value.mean().numpy()
            detach_value[:, -1] *= done
            bootstrap_value = detach_value[:, -1]
            delta_target_list = [torch.zeros(BATCH).to(self.device)]
            delta_target_list.append(
                reward[:, UNROLL_STEP-1] + DISCOUNT * bootstrap_value - detach_value[:, -2]
            )
            j = 1
            for i in reversed(range(UNROLL_STEP-1)):
                cs_action = cliped_c[:, i]
                td_error = reward[:, i] + DISCOUNT *detach_value[:, i+1] -  detach_value[:, i]
                td_error *= cliped_p[:, i]

                delta_target_list.append(
                    td_error + DISCOUNT * cs_action *delta_target_list[j]
                )
                j += 1
            
            delta_target = torch.stack(delta_target_list[::-1], dim=1)
            target_value = detach_value + delta_target
            action_target = reward[:, :-1] + DISCOUNT * target_value[:, 1:]
            prev_target_value = target_value[:, :-1].contiguous()

            advantage = action_target - prev_target_value
        selected_log_policy = selected_log_policy.view(BATCH, UNROLL_STEP+1)
        selected_log_policy *= cliped_p
        selected_log_policy = selected_log_policy[:, :-1].contiguous()
        value = value[:, :-1].contiguous()
        critic_loss = torch.sum((value - prev_target_value) ** 2)
        actor_loss = torch.sum(
            - selected_log_policy * advantage
        )
        obj = critic_loss * 0.5 + actor_loss - ENTROPY_COEFFI * entropy
        obj.backward()
        info = self.step(t)
        mean_entropy = (entropy / (BATCH * UNROLL_STEP)).detach().cpu().numpy()
        info['entropy'] = float(mean_entropy)
        info['mean_value'] = float(mean_value)
        return info

    def step(self, t):
        norm = 40
        c_norm = 0
        p_norm = 0
        pp = []
        with torch.no_grad():
            pp += self.model.getParameters()
            for p in pp:
                p_norm += p.grad.data.norm(2)
            
            for m in self.prev_models:
                pp += m.getParameters()
            for p in pp:
                c_norm += p.grad.data.norm(2)
            c_norm = c_norm ** .5
            p_norm = p_norm ** .5
        torch.nn.utils.clip_grad_norm_(pp, norm)
        for optim in self.optim:
            optim.step()
        for optim in self.optim:
            optim.zero_grad()
        c_lr = self._decay_lr()
        info = {}
        info['norm'] = c_norm.numpy()
        info['p_norm'] = p_norm.numpy()
        info['lr'] = c_lr
        return info
    
    def _decay_lr(self):
        delta = (0.0006 - 0.0001) / 1e6
        for optim in self.optim:
            for g in optim.param_groups:
                if g['lr'] > 0.0001:
                    g['lr'] -= delta
        return g['lr']
        
    def state_dict(self):
        state_dict = {}
        state_dict[0] = {k:v.cpu() for k, v in self.model.state_dict().items()}
        for i, m in enumerate(self.prev_models):
            state_dict[i+1] = {k:v.cpu() for k, v in m.state_dict().items()}
        return state_dict

    def receive_log_from_actor(self, step=0):
        pipe = self.connect.pipeline()
        pipe.lrange("metric", 0, -1)
        pipe.ltrim("metric", -1, 0)
        info_ = pipe.execute()[0]
        self.connect.delete("metric")
        if info_ is not None:
            for info in info_:
                metric = pickle.loads(info)
                day, income = metric
                print(day, income)
                if TRAIN_LOG_MODE:
                    TRAIN_LOGGER.info("{},{}".format(day, income))

    def run(self):
        def wait_memory():
            while True:
                if len(self.memory.memory) > (REPLAY_MEMORY_LEN - 1):
                    break
                else:
                    print(len(self.memory.memory))
                time.sleep(1)

        wait_memory()
        print("Learning is Start !!")
        step = 0
        norm, mean_value, entropy = 0, 0, 0
        p_norm = 0

        for t in count():
            transition = self.memory.sample()
            if transition is False:
                time.sleep(0.2)
                continue
            step += 1
            info = self.train(transition, step)
            # print(info)

            norm += info['norm']
            mean_value += info['mean_value']
            entropy += info['entropy']
            p_norm += info['p_norm']
            lr = info['lr']
            
            state_dict = self.state_dict()
            self.connect.set(
                "param", pickle.dumps(
                    state_dict
                )
            )
            self.connect.set(
                "count", pickle.dumps(step)
            )
            gc.collect()

            if step % 100 == 0:
                norm /= 100
                mean_value /= 100
                entropy /= 100
                p_norm /= 100

                print("STEP:{} // NORM:{:.3f} // P_NORM:{:.3f} // VALUE:{:.3f} // ENTROPY:{:.3f} // LR:{:.5f}".format(
                    step, norm, p_norm, mean_value, entropy, lr
                ))
                if TRAIN_LOG_MODE:
                    TRAIN_LOGGER.info(
                        '{},{},{},{}'.format(step, norm, p_norm, mean_value, entropy)
                    )
                
                norm, mean_value, entropy =0, 0, 0
                p_norm = 0


