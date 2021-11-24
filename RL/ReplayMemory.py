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
        self.connect = redis.StrictRedis()
        self._lock = threading.Lock()
        self.deque = []
        self.device = torch.device(LEARNER_DEVICE)
    
    def buffer(self, file=None):
        """
            experience:
                [s, a, r, ----, s, a, r, d]
            
            state[List]:
                keys:
                    1 minute
                    5 minute
                    15 minute
                    60 minute

                    1m cell state
                    5m cell state
                    15m cell state
                    60m cell state
            
            action[int]:
                0 or 1
            
            reward[float]:
                float
            
            done[Bool]:
                Bool
        """
        if file is None:
            transition_bin = self.memory.sample(32)
            transition = [pickle.loads(bin) for bin in transition_bin]
        else:
            x = open(file, 'rb')
            data = x.read()
            x.close()
            transition = pickle.loads(data)
        BATCH = len(transition)
        
        np_transition = np.array(transition)
        state = np_transition[:, 0]
        policy = [s[-1] for s in state]
        done = list(np_transition[:, -1].astype(np.bool8))
        policy = np.stack(policy, axis=0)
        policy = torch.tensor(policy).to(self.device)
        d_state = [s[0] for s in state]
        d_state = torch.tensor(np.stack(d_state, axis=0)).to(self.device)
        # BATCH, SEQ<, 4, 3
        m1_state = d_state[:, :, 0].permute(1, 0, 2).contiguous()
        # BATCH, SEQ, 4, 3
        kxk = UNROLL_STEP + 1

        m5_idx = [i*5 for i in range(int(kxk/5))]
        m5_state = d_state[:, m5_idx, 1].permute(1, 0, 2).contiguous()

        m15_idx = [i*15 for i in range(int(kxk/15))]
        m15_state = d_state[:, m15_idx, 2].permute(1, 0, 2).contiguous()

        m60_idx = [i*60 for i in range(int(kxk/60))]
        m60_state = d_state[:, m60_idx, 3].permute(1, 0, 2).contiguous()


        m1_cell_state =torch.cat([s[1][0] for s in state], 1).to(self.device)
        m5_cell_state =torch.cat([s[1][1] for s in state], 1).to(self.device)
        m15_cell_state =torch.cat([s[1][2] for s in state], 1).to(self.device)
        m60_cell_state =torch.cat([s[1][3] for s in state], 1).to(self.device)
        cell_state = (m1_cell_state, m5_cell_state, m15_cell_state, m60_cell_state)
        state = (m1_state, m5_state, m15_state, m60_state)
        action = np_transition[:, 1]
        if USE_FLOAT64:
            dtype = np.float64
        else:
            dtype = np.float32
        action = np.stack([a.astype(dtype) for a in action], 0)
        reward = np_transition[:, 2]
        reward = np.stack([r.astype(dtype) for r in reward], 0)
        # self.deque.append(((state, cell_state), action, reward))
        return (state, cell_state, policy), action, reward, done

    def run(self):
        t = 0
        while True:
            if len(self.memory) > REPLAY_MEMORY_LEN * 0.8:
                if len(self.deque) < 5:
                    with self._lock:
                        self.deque.append(self.buffer())
            t += 1
            pipe = self.connect.pipeline()
            pipe.lrange("trajectory", 0, -1)
            pipe.ltrim("trajectory", -1, 0)
            data = pipe.execute()[0]
            self.connect.delete("trajectory")
            if data is not None:
                with self._lock:
                    for d in data:
                        self.memory.push(d)
            time.sleep(0.01)
            gc.collect()
        
    def sample(self):
        if len(self.memory) > REPLAY_MEMORY_LEN * 0.8:
            if len(self.deque) > 0:
                d = self.deque.pop(0)
                return d
            else:
                # print(len(self.memory))
                return False
        else:
            # print(len(self.memory))
            return False