import gc
import time
import torch
import _pickle as pickle
import threading
import redis
import cProfile
from pstats import Stats

import itertools
import numpy as np

from copy import deepcopy
from collections import deque
from baseline.utils import loads, ReplayMemory
from baseline.PER import PER
from random import choices

from configuration import *


# class Replay:
class Replay(threading.Thread):

    def __init__(self):
        super(Replay, self).__init__()
        self.setDaemon(True)
        self.use_PER = USE_PER
        self.memory = PER(
            maxlen=REPLAY_MEMORY_LEN,
            max_value=1.0,
            beta=BETA)

        self.cond = False
        # self.connect = redis.StrictRedis(host=REDIS_SERVER, port=6379)
        self.connect = redis.StrictRedis()
        self._lock = threading.Lock()
        self.deque = []
        self.update_list = []
        self.device = torch.device(LEARNER_DEVICE)
        self.total_frame = 0
        self.lock = False

        self.idx = []
        self.vals = []
    
    def update(self, idx:list, vals:np.ndarray):
        self.idx += idx
        self.vals.append(vals)
    
    def buffer(self):
        m = 16

        experiences, prob, idx = self.memory.sample(
            BATCHSIZE * m
        )
        n = len(self.memory.memory)
        weight = (1 / (n * prob)) ** BETA
        weight /= self.memory.max_weight

        experiences = [pickle.loads(bin) for bin in experiences]


        state_idx = [1 + i * 4 for i in range(FIXED_TRAJECTORY)]
        ratio_idx = [2 + i * 4 for i in range(FIXED_TRAJECTORY)]
        action_idx = [3 + i * 4 for i in range(FIXED_TRAJECTORY)]
        reward_idx = [4 + i * 4 for i in range(FIXED_TRAJECTORY)]

        state = [np.stack(exp[state_idx], 0) for exp in experiences]
        state = np.stack(state, 0)

        ratio = np.array([exp[ratio_idx].astype(np.float32) for exp in experiences])

        action = np.array([exp[action_idx].astype(np.int32) for exp in experiences])
        # action = np.transpose(action, (1, 0))
        reward = np.array([exp[reward_idx].astype(np.float32) for exp in experiences])
        # reward = np.transpose(reward, (1, 0))
        done = np.array([float(not exp[-2]) for exp in experiences])
        hidden_state_0 = torch.cat([exp[0][0] for exp in experiences], 1)
        hidden_state_1 = torch.cat([exp[0][1] for exp in experiences], 1)

        hidden_states_0 = torch.split(hidden_state_0, BATCHSIZE, dim=1)
        hidden_states_1 = torch.split(hidden_state_1, BATCHSIZE, dim=1)

        states = np.vsplit(state, m)

        ratios = np.split(ratio, m)

        actions = np.split(action, m)

        rewards = np.split(reward, m)

        dones = np.split(done, m)

        weights = weight.split(BATCHSIZE)
        idices = idx.split(BATCHSIZE)
        

        for s0, s1, a, r, h0, h1, d, w, i in zip(
            states, ratios, actions, rewards, hidden_states_0, hidden_states_1, dones, weights, idices
        ):

            self.deque.append(
                [(h0, h1), s0, s1, a, r, d, w, i]
            )

    def _update(self):
        with self._lock:
            vals = np.concatenate(deepcopy(self.vals), axis=0)
            if len(self.idx) == 0:
                return 
            if len(vals) != len(self.idx):
                print(len(vals))
                print(len(self.idx))
                print("!!")
                return
            try:
                self.memory.update(self.idx, vals)
            except:
                print("Update fails, if it happens")
            self.vals.clear()
            self.idx.clear()

    def run(self):
        t = 0
        data = []
        m = 1000
        while True:
            
            pipe = self.connect.pipeline()
            pipe.lrange("experience", 0, -1)
            pipe.ltrim("experience", -1, 0)
            data += pipe.execute()[0]
            data: list
            self.connect.delete("experience")
            if len(data) > 0:
                self.memory.push(data)
                self.total_frame += len(data)
                data.clear()
                if len(self.memory.priority.prior_torch) > m:
                    if len(self.deque) < 12:
                        self.buffer()
                        t += 1
                        if t == 1:
                            print("Data Batch Start!!!")
                if len(self.idx) > 1000:
                    self._update()
                    self.idx.clear()
                    self.vals.clear()
                if self.lock:
                    if len(self.memory) < REPLAY_MEMORY_LEN:
                        pass
                    else:
                        self.deque.clear()
                        self.memory.remove_to_fit()
                        self.idx.clear()
                        self.vals.clear()
                        self.buffer()
                    self.lock = False
            gc.collect()
        
    def sample(self):
        if len(self.deque) > 0:
            return self.deque.pop(0)
        else:
            return False
