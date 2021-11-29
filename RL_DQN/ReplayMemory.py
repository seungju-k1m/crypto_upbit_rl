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

from configuration import *


# class Replay:
class Replay(threading.Thread):

    def __init__(self):
        super(Replay, self).__init__()
        self.setDaemon(True)
        self.use_PER = USE_PER
        # self.memory = ReplayMemory(REPLAY_MEMORY_LEN)
        if self.use_PER:
            self.memory = PER(
                maxlen=REPLAY_MEMORY_LEN,
                max_value=1.0,
                beta=BETA)
        else:
            self.memory = ReplayMemory(REPLAY_MEMORY_LEN)
        # PER 구현해보자
        self.cond = False
        self.connect = redis.StrictRedis(host=REDIS_SERVER, port=6379)
        self._lock = threading.Lock()
        self.deque = []
        self.update_list = []
        self.device = torch.device(LEARNER_DEVICE)
        self.total_frame = 0
        self.lock = False
    
    def update(self, idx:list, vals:np.ndarray):
        # self.update_list.append((idx, vals))
        self.memory.update(idx, vals)

    def buffer(self, print_f=False):
        sample_time = time.time()
        if self.use_PER:
            experiences, prob, idx = self.memory.sample(BATCHSIZE)
            n = len(self.memory)
            weight = (1 / (n * prob)) ** BETA
            weight /= max(weight)
        else:
            experiences = deepcopy(self.memory.sample(BATCHSIZE))
        
        if print_f:
            print("---Sample Time:{:.3f}".format(time.time() - sample_time))

        preprocess_time = time.time()

        experiences = np.array([pickle.loads(bin) for bin in experiences])

        # state = np.stack([(bin), dtype=np.uint8) for bin in experiences[:, 0]], 0)
        state = np.stack(experiences[:, 0], 0)
        if print_f:
            print("-----PP_01:{:.3f}".format(preprocess_time - time.time()))
        # S, A, R, S_
        # experiences = np.array([list(map(pickle.loads, experiences))])
        # BATCH, 4
        # state = np.stack(experiences[:, 0], 0)
        if print_f:
            print("-----PP_02:{:.3f}".format(preprocess_time - time.time()))

        action = list(experiences[:, 1])
        # action = [int(i) for i in experiences[:, 1]]
        reward = list(experiences[:, 2])
        # reward = [float(i) for i in experiences[:, 2]]
        next_state = np.stack(experiences[:, 3], 0)
        # next_state = np.stack([np.frombuffer(base64.b64decode(bin), dtype=np.uint8) for bin in experiences[:, 3]], 0)
        done = list(experiences[:, 4])
        # done = [bool(i) for i in experiences[:, 4]]
        if print_f:
            print("-----PP_03:{:.3f}".format(preprocess_time - time.time()))

        if self.use_PER:
            return (state, action, reward, next_state, done, weight, idx)
        else:
            return (state, action, reward, next_state, done)
    
    def run(self):
        t = 0
        data = []
        while True:
            if len(self.memory.priority) >  50000:
                self.cond = True
            
            pipe = self.connect.pipeline()
            pipe.lrange("experience", 0, -1)
            pipe.ltrim("experience", -1, 0)
            data += pipe.execute()[0]
            data: list
            self.connect.delete("experience")
            if len(data) > 0:
                check_time = time.time()
                self.memory.push(data)
                # print("Push Time:{:.3f}".format(time.time() - check_time))
                self.total_frame += len(data)
                data.clear()
                assert len(self.memory.memory) == len(self.memory.priority)
                if len(self.memory) > 50000:
                    with self._lock:
                        if len(self.deque) < 5:
                            zx = time.time()

                            self.deque.append(self.buffer())
                            self.deque.append(self.buffer())
                            mm = time.time() - zx
                            t += 1
                if self.lock:
                    self.deque.clear()
                    self.memory.remove_to_fit()
                    self.lock = False
                if (t+1) % 500 == 0:
                    profile = cProfile.Profile()
                    profile.runctx('self.buffer()', globals(), locals())
                    profile.print_stats()
                
            gc.collect()
        
    def sample(self):
        if len(self.deque) > 0:
            return self.deque.pop(0)
        else:
            return False