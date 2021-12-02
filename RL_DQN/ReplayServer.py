from configuration import *

from baseline.PER import PER

import _pickle as pickle
import numpy as np

import torch
import redis
import gc


class ReplayServer:
    def __init__(self):
        self.memory = PER(
            maxlen=REPLAY_MEMORY_LEN,
            max_value=1.0,
            beta=BETA
        )

        self.FLAG_BATCH = False
        self.FLAG_REMOVE = False
        self.FLAG_ENOUGH = False

        self.connect = redis.StrictRedis(host=REDIS_SERVER, port=6379)

        self.device = torch.device("cpu")
        self.total_transition = 0

        self.connect.set("FLAG_BATCH", pickle.dumps(False))
    
    def update(self):
        pipe = self.connect.pipeline()
        pipe.lrange("update", 0, -1)
        pipe.ltrim("update", -1, 0)
        data = pipe.execute()[0]
        self.connect.delete("update")
        if len(data) == 0:
            return
        else:
            idx_list, vals_list = [], []
            for d in data:
                idx, vals = pickle.loads(d)
                idx: list
                vals: np.ndarray

                idx_list += idx
                vals_list.append(vals)
            vals_np = np.concatenate(vals_list, 0)

            try:
                self.memory.update(idx_list, vals_np)
            except:
                print("Update fails, if it happens")

    def buffer(self):
        m = 16
        experiences, prob, idx = self.memory.sample(
            BATCHSIZE * m
        )
        n = len(self.memory.memory)
        weight = (1 / (n * prob)) ** BETA
        weight /= self.memory.max_weight

        experiences = np.array([pickle.loads(bin) for bin in experiences])
        state = np.stack(experiences[:, 0], 0)
        action = experiences[:, 1]
        reward = experiences[:, 2]
        next_state = np.stack(experiences[:, 3], 0)
        done = experiences[:, 4]
        states = np.vsplit(state, m)

        actions = np.split(action, m)

        rewards = np.split(reward, m)

        next_states = np.vsplit(next_state, m)

        dones = np.split(done, m)

        weights = weight.split(BATCHSIZE)
        idices = idx.split(BATCHSIZE)

        for s, a, r, n_s, d, w, i in zip(
            states, actions, rewards, next_states, dones, weights, idices
        ):
            self.connect.rpush(
                "BATCH",pickle.dumps(
                    [s, a, r, n_s, d, w, i]
                )
            )

    def run(self):
        data = []
        k = 50000
        while 1:
            if len(self.memory.priority.prior_torch) > k:
                self.FLAG_BATCH = True
                self.connect.set("FLAG_BATCH", pickle.dumps(True))
            
            pipe = self.connect.pipeline()
            pipe.lrange("experience", 0, -1)
            pipe.ltrim("experience", -1, 0)
            data += pipe.execute()[0]
            data: list
            self.connect.delete("experience")

            if len(data) > 0:
                self.memory.push(data)
                self.total_transition += len(data)
                data.clear()
                if len(self.memory) > 1000:
                    cond = self.connect.get(
                        "FLAG_ENOUGH"
                    )
                    if cond is not None:
                        self.FLAG_ENOUGH = pickle.loads(cond)
                        # Learner에서 정해준다.

                    if not self.FLAG_ENOUGH:
                        self.buffer()
            
            self.update()
            if len(self.memory) > REPLAY_MEMORY_LEN:
                cond = self.connect.get(
                        "FLAG_REMOVE"
                    )
                if cond is not None:
                    self.FLAG_REMOVE = pickle.loads(cond)
                    # Learner에서 요청하면
                
                if self.FLAG_REMOVE:
                    self.memory.remove_to_fit()
                    self.connect.set(
                        "FLAG_REMOVE", pickle.dumps(False)
                    )
                    self.FLAG_REMOVE = False
                    # 요청을 수행하고 다시
            gc.collect()