# BITCOIN SIMULATOR FOR BACKTEST
from datetime import datetime
from configuration import DURATION, RENDER_TIME, UNIT_MINUTE
from gym import Env
from typing import Union, List

from Simulator.pipeline import DataPipeLine
from Simulator.renderer import Renderer
import pandas as pd
import numpy as np

import random
import os


class Simulator(Env):

    def __init__(
        self,
        to: Union[List[str], str],
        duration:Union[List[int], int],
        init_volume: float=1
    ):
        self.to = to
        self.duration = duration
        self.init_volume = init_volume
        self.var_mode = False
        self.renderer = Renderer()
        self.num_len = self.renderer.screen_size
        # self.init_pipeline()
        self.prev_notional_price = None
        # self.reset()
        self.coin = False       

    def init_pipeline(self):
        self.pipeline = DataPipeLine(self.to, self.duration)

    def init_random(self):
        path = './data/process'
        list_path = os.listdir(path)

        k = random.choice(list_path)
        k = k[:19]
        path = os.path.join(path, k)
        self.to = k
        self.duration = DURATION
        self.init_pipeline()
        self.coin = False

    def reset(self):
        def normalize(x, i):
            if i == 0:
                x[2] /= 100000
                x[3] /= 100
            if i == 1:
                x[2] /= 100000
                
                x[3] /= 500
            if i == 2:
                x[2] /= 100000
                x[3] /= 1500
            if i == 3:
                x[2] /= 500000
                x[3] /= 6000
            return x
        render_time = UNIT_MINUTE[RENDER_TIME]
        self.midpoint = self.pipeline.data[render_time]['midpoint'].to_numpy()
        self.datetime = pd.to_datetime(self.pipeline.data[render_time]['time']).to_numpy()
        self.count = 0
        self.renderer.init_data(
            self.datetime[:self.num_len],
            self.midpoint[:self.num_len]
        )
        self.prev_notional_price = self.midpoint[0]
        state = self.pipeline.get(0, unit=1)
        self.coin = False
        state = [np.append(normalize(i, j), np.array(float(self.coin))) for j, i in enumerate(state)]
        self.init_account = self.init_volume * state[0][1]
        return state

    def render(self, state):
        s = state[RENDER_TIME]
        tt, midpoint = s[:2]
        if self.count < self.num_len:
            pass
        else:
            self.renderer.render(
                midpoint=midpoint,
                tt=tt
            )

    def step(self, action=0, unit=1):
        # action -> 0, 1, 2
        def normalize(x, i):
            if i == 0:
                x[2] /= 100000
                x[3] /= 100
            if i == 1:
                x[2] /= 100000
                
                x[3] /= 500
            if i == 2:
                x[2] /= 100000
                x[3] /= 1500
            if i == 3:
                x[2] /= 500000
                x[3] /= 6000
            return x

        state = self.pipeline.get(self.count, unit=unit)
        if state is None:
            done = True
            reward = 0
        else:
            done = False
            self.count += 1
            idx = UNIT_MINUTE.index(unit)
            delta = (state[idx][1] - self.prev_notional_price) / self.init_account
            fee = 0.0000
            if self.coin:
                if action == 0:
                    action = "HOLD"
                    reward = delta
                    
                    # reward = 0.01
                elif action == 1:
                    action = "ASK"
                    self.coin = False
                    reward = -fee
                
            else:
                if action == 0:
                    action = "HOLD"
                    # reward = -delta
                    reward = 0
                else:
                    action = "BID"
                    self.coin = True
                    reward = -fee
            # self.init_account += reward
            self.prev_notional_price = state[idx][1]
            state = [np.append(normalize(i, j), np.array(float(self.coin))) for j, i in enumerate(state)]
        return state, reward, done, None

    