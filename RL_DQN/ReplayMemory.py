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

from configuration import *

# class Replay:
class Replay(threading.Thread):

    def __init__(self):
        super(Replay, self).__init__()
        self.setDaemon(True)
        self.memory = ReplayMemory(REPLAY_MEMORY_LEN)
        # PER 구현해보자
        self.connect = redis.StrictRedis(host=REDIS_SERVER, port=6379)
        self._lock = threading.Lock()
        self.deque = []
        self.device = torch.device(LEARNER_DEVICE)
    
    def buffer(self):
        experiences = self.memory.sample(32)
        experiences = np.array([pickle.loads(bin) for bin in experiences])
        # S, A, R, S_
        # experiences = np.array([list(map(pickle.loads, experiences))])
        # BATCH, 4
        state = np.stack(experiences[:, 0], 0)
        action = list(experiences[:, 1])
        reward = list(experiences[:, 2])
        next_state = np.stack(experiences[:, 3], 0)
        done = list(experiences[:, -1])
        return (state, action, reward, next_state, done)
    
    def run(self):
        t = 0
        while True:
            if len(self.memory) > REPLAY_MEMORY_LEN * 0.1:
                if len(self.deque) < 5:
                    self.deque.append(self.buffer())
            t += 1
            pipe = self.connect.pipeline()
            pipe.lrange("experience", 0, -1)
            pipe.ltrim("experience", -1, 0)
            data = pipe.execute()[0]
            self.connect.delete("trajectory")
            if data is not None:
                with self._lock:
                    for d in data:
                        self.memory.push(d)
            time.sleep(0.01)
            gc.collect()
        
    def sample(self):
        if len(self.deque) > 0:
            d = self.deque.pop(0)
            return d
        else:
            # print(len(self.memory))
            return False