from configuration import *
from baseline.baseAgent import baseAgent
from baseline.utils import getOptim, writeTrainInfo
from RL_DQN.ReplayMemory import Replay

from torch.utils.tensorboard import SummaryWriter
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


        self.writer = SummaryWriter(
            os.path.join(
                LOG_PATH, 'DQN'
            )
        )
        
        names = self.connect.scan()
        if len(names[-1]) > 0:
            self.connect.delete(*names[-1])
        
        if USE_GYM:
            self.run = self.gym_run
        else:
            self.run = self.bit_run


    def build_model(self):
        info = DATA["model"]
        self.model = baseAgent(info)
        self.model.to(self.device)
        self.target_model = baseAgent(info)
        self.target_model.to(self.device)
    
    def build_optim(self):
        self.optim = getOptim(OPTIM_INFO, self.model)
        
    def train(self, transition, t=0) -> dict:
        state, action, reward, next_state, done = transition

        state = torch.tensor(state).float()
        state = state / 255.
        state = state.to(self.device)

        next_state = torch.tensor(next_state).float()
        next_state = next_state / 255.
        next_state = next_state.to(self.device)

        # action = torch.tensor(action).long().to(self.device)
        action = [6 * i + a for i, a in enumerate(action)]
        reward = torch.tensor(reward).float().to(self.device)

        done = [float(not d) for d in done]
        done = torch.tensor(done).float().to(self.device)
        action_value = self.model.forward([state])[0]

        with torch.no_grad():
            action_ddqn =  action_value.argmax(dim=-1).detach().cpu().numpy()
            action_ddqn = [6*i + a for i, a in enumerate(action_ddqn)]
            next_action_value = self.target_model.forward([next_state])[0]
            next_action_value = next_action_value.view(-1)
            next_action_value = next_action_value[action_ddqn]
            # next_max_value, _ = next_action_value.max(dim=-1)
            next_max_value =  next_action_value * done
        
        action_value = action_value.view(-1)
        selected_action_value = action_value[action].view(-1, 1)

        target = reward + 0.99 * next_max_value

        loss = torch.mean((target - selected_action_value) ** 2)

        loss.backward()
        info = self.step()

        info['mean_value'] = float(selected_action_value.mean().detach().cpu().numpy())           
        return info

    def step(self):
        p_norm = 0
        pp = []
        with torch.no_grad():
            pp += self.model.getParameters()
            for p in pp:
                p_norm += p.grad.data.norm(2)
            p_norm = p_norm ** .5
        # for optim in self.optim:
        #     optim.step()
        self.optim.step()
        self.optim.zero_grad()
        info = {}
        info['p_norm'] = p_norm.cpu().numpy()
        return info
    
    def state_dict(self):
        state_dict = {k:v.cpu() for k, v in self.model.state_dict().items()}
        return state_dict

    def bit_run(self):
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
        obj = 0

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
            obj += info['obj']
            
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
                obj /= 100

                print("STEP:{} // NORM:{:.3f} // P_NORM:{:.3f} // VALUE:{:.3f} // ENTROPY:{:.3f} // LR:{:.5f} // OBJ:{:.3f}".format(
                    step, norm, p_norm, mean_value, entropy, lr, obj
                ))
                if TRAIN_LOG_MODE:
                    TRAIN_LOGGER.info(
                        '{},{},{},{}'.format(step, norm, p_norm, mean_value, entropy)
                    )
                
                norm, mean_value, entropy =0, 0, 0
                p_norm = 0
                obj = 0

    def gym_run(self):
        def wait_memory():
            while True:
                if len(self.memory.deque) > 1:
                    break
                else:
                    print(len(self.memory.memory))
                time.sleep(1)
        wait_memory()
        print("Learning is Started !!")

        step, norm, mean_value = 0, 0, 0

        for t in count():
            experience = self.memory.sample()
            if experience is False:
                time.sleep(0.2)
                continue
            step += 1
            info = self.train(experience)

            norm += info['p_norm']
            mean_value += info['mean_value']

            # target network updqt
            # soft
            # self.target_model.updateParameter(self.model, 0.005)
            # hard
            if step % 500 == 0:
                self.target_model.updateParameter(self.model, 1)

            state_dict = pickle.dumps(self.state_dict())
            step_bin = pickle.dumps(step)
            self.connect.set("state_dict", state_dict)
            self.connect.set("count", step_bin)
            if step % 1000 == 0:
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

                print(
                    "step:{} // mean_value:{:.3f} // norm: {:.3f} // REWARD:{:.3f} // NUM_MEMORY:{}".format(
                        step, mean_value, norm, cumulative_reward, len(self.memory.memory))
                )

                if len(data) > 0:
                    self.writer.add_scalar(
                        "Reward", cumulative_reward, step
                    )
                self.writer.add_scalar(
                    "value", mean_value / 1000, step
                )
                self.writer.add_scalar(
                    "norm", norm/ 1000, step
                )
                mean_value, norm = 0, 0
                torch.save(self.state_dict(), './weight/dqn/weight.pth')

                

