import gc
import time
import torch
import _pickle as pickle
import threading
import redis

import numpy as np

from copy import deepcopy
from collections import deque
from baseline.utils import loads, ReplayMemory
from baseline.PER import PER

from configuration import *

# class Replay:
class Replay(threading.Thread):

    def __init__(self):
        super(Replay, self).__init__()
        self.setDaemon(True)
        self.use_PER = USE_PER
        # self.memory = ReplayMemory(REPLAY_MEMORY_LEN)
        self.lock = False
        if self.use_PER:
            self.memory = PER(
                REPLAY_MEMORY_LEN,
                max_value=1.0)
        else:
            self.memory = ReplayMemory(REPLAY_MEMORY_LEN)
        # PER 구현해보자
        self.cond = False
        self.connect = redis.StrictRedis(host=REDIS_SERVER, port=6379)
        self._lock = threading.Lock()
        self.deque = []
        self.device = torch.device(LEARNER_DEVICE)
    
    def buffer(self):
        if self.use_PER:
            experiences, prob, idx = self.memory.sample(32)
            n = len(self.memory)
            weight = (1 / (n * prob)) ** BETA / self.memory.max_weight
        else:
            experiences = deepcopy(self.memory.sample(32))
        experiences = np.array([pickle.loads(bin) for bin in experiences])
        # S, A, R, S_
        # experiences = np.array([list(map(pickle.loads, experiences))])
        # BATCH, 4
        state = np.stack(experiences[:, 0], 0)
        action = list(experiences[:, 1])
        reward = list(experiences[:, 2])
        next_state = np.stack(experiences[:, 3], 0)
        done = list(experiences[:, -1])
        if self.use_PER:
            return (state, action, reward, next_state, done, weight), idx
        else:
            return (state, action, reward, next_state, done)
    
    def run(self):
        t = 0
        data = []
        while True:
            if len(self.memory) > REPLAY_MEMORY_LEN * 0.05:
            # if len(self.memory) > 1000:
                self.cond = True
            t += 1
            if not self.lock:
                pipe = self.connect.pipeline()
                pipe.lrange("experience", 0, -1)
                pipe.ltrim("experience", -1, 0)
                data += pipe.execute()[0]
                self.connect.delete("experience")
                # for i in range(len(data)):
                #     if not self.lock:
                #         self.memory.push(data.pop(0))
                #     else:
                #         break
                if len(data) > 0:
                    # print(len(data))
                    if self.lock:
                        while True:
                            if not self.lock:
                                break
                        self.memory.push(data)
                        data.clear()
            time.sleep(0.01)
            gc.collect()
        
    def sample(self):
        if self.cond:
            return self.buffer()
        else:
            # print(len(self.memory))
            return False