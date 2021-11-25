# BITCOIN SIMULATOR FOR BACKTEST
from datetime import datetime
from configuration import DURATION, RENDER_TIME, UNIT_MINUTE, FEE
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
        self.renderer = []
        self.offset = 48 * 60
        for unit in UNIT_MINUTE:
            self.renderer.append(Renderer(unit=unit))
        self.num_len = self.renderer[0].screen_size
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

    def reset(self, mode='x'):
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
        i = 0
        self.midpoint, self.datetime, self.volume = [], [], []
        self.prev_notional_price = []
        k = [60, 12, 4, 1]
        for unit, m in zip(UNIT_MINUTE, k):
            render_time = unit
            self.midpoint.append(self.pipeline.data[render_time]['midpoint'].to_numpy())
            self.datetime.append(pd.to_datetime(self.pipeline.data[render_time]['time']).to_numpy())
            self.volume.append(self.pipeline.data[render_time]['acc_volume'].to_numpy())
            self.count = 0
            if mode is 'human':
                self.renderer[i].init_data(
                    self.datetime[i][self.num_len * (m-1):self.num_len * m],
                    self.midpoint[i][self.num_len * (m-1):self.num_len * m]
                )
            else:
                self.renderer[i].init_data(
                    self.datetime[i][self.num_len * (m-1):self.num_len * m],
                    self.midpoint[i][self.num_len * (m-1):self.num_len * m],
                    self.volume[i][self.num_len * (m-1):self.num_len * m]
                )
            i += 1
        for r in self.renderer:
            r.render()
        self.prev_notional_price = self.midpoint[0][self.offset]

        
        state = self.pipeline.get(self.offset, unit=1)
        self.coin = False
        state = [np.append(normalize(i, j), np.array(float(self.coin))) for j, i in enumerate(state)]
        self.krw_account = self.init_volume * state[0][1]
        self.coin_account = 0
        self.bid_price = state[0][1]
        self.total_account = self.krw_account
        self.init_krw_account =  self.init_volume * state[0][1]
        return state

    def render(self, state):
        obs_list = []
        m1_obs = self.renderer[0].render(state[0])
        obs_list.append(m1_obs)
        if self.count % 5 == 0:
            m5_obs = self.renderer[1].render(state[1])
        else:
            m5_obs = self.renderer[1].get_current_fig()
        obs_list.append(m5_obs)
        if self.count % 15 == 0:
            m15_obs = self.renderer[2].render(state[2])
        else:
            m15_obs = self.renderer[2].get_current_fig()
        obs_list.append(m15_obs)
        if self.count % 60 == 0:
            m60_obs = self.renderer[3].render(state[3])
        else:
            m60_obs = self.renderer[3].get_current_fig()
        obs_list.append(m60_obs)

        obs = np.stack(obs_list, axis=-1)
        return obs
        

    def step(self, action=0):
        # action -> 0, 1, 2
        def normalize(x, i):
            x[2] = x[2] / 100000
            # x[3] = x[3] / 100
            zz = 100
            if i == 0:
                # x[2] = x[2] / 100000
                x[3] = x[3] / zz
            if i == 1:
                # x[2] /= 100000
                
                x[3] = x[3] / (5*zz)
            if i == 2:
                # x[2] /= 100000
                x[3] = x[3] / (15 * zz)
            if i == 3:
                # x[2] /= 500000
                x[3] = x[3] / (60 * zz)
            return x
        unit=1
        self.count += 1
        state = self.pipeline.get(self.offset + self.count, unit=1)
        if state is None:
            done = True
            reward = 0
        else:
            done = False
            # self.count += 1
            idx = UNIT_MINUTE.index(unit)
            current_price = state[idx][1]
            delta = (state[idx][1] - self.prev_notional_price)
            fee = FEE
            if self.coin:
                if action == 0:
                    action = "HOLD"
                    reward = 0
                    # reward = 0.01
                elif action == 1:
                    action = "ASK"
                    self.coin = False
                    reward = (current_price - self.bid_price) / self.bid_price
                    reward -= fee
                    self.krw_account = current_price * self.coin_account * (1 - fee)
                    self.coin_account = 0
            else:
                if action == 0:
                    action = "HOLD"
                    # reward = -delta
                    reward = 0
                else:
                    action = "BID"
                    self.coin = True
                    reward = -fee
                    self.bid_price = current_price
                    self.coin_account = self.krw_account * (1-fee) / current_price
                    self.krw_account = 0
                    
            self.prev_notional_price = state[idx][1]
            state = [np.append(i, np.array(float(self.coin))) for j, i in enumerate(state)]
        return state, reward, done, None

    