# BITCOIN SIMULATOR FOR BACKTEST
from datetime import datetime
from configuration import DURATION, RENDER_TIME, RUN_UNIT_TIME, UNIT_MINUTE, FEE
from gym import Env
from typing import Union, List

from Simulator.pipeline import DataPipeLine
from Simulator.renderer import Renderer
import pandas as pd
import numpy as np

import random
import math
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

    def reset(self, mode='x'):
        self.init_random()
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
        self.coin = True
        # state = [np.append(i, np.array(float(self.coin))) for j, i in enumerate(state)]
        state.append(float(self.coin))
        return state

    def render(self, state):
        mode = state[-1]
        obs_list = []
        m1_obs = self.renderer[0].render(state[0],mode)
        obs_list.append(m1_obs)
        if self.count % 5 == 0:
            m5_obs = self.renderer[1].render(state[1],mode)
        else:
            m5_obs = self.renderer[1].get_current_fig()
        obs_list.append(m5_obs)
        if self.count % 15 == 0:
            m15_obs = self.renderer[2].render(state[2],mode)
        else:
            m15_obs = self.renderer[2].get_current_fig()
        obs_list.append(m15_obs)
        if self.count % 60 == 0:
            m60_obs = self.renderer[3].render(state[3],mode)
        else:
            m60_obs = self.renderer[3].get_current_fig()
        obs_list.append(m60_obs)

        obs = np.stack(obs_list, axis=-1)
        return obs
        
    def step(self, action=0):
        unit = UNIT_MINUTE[RUN_UNIT_TIME]
        self.count += 1
        state = self.pipeline.get(self.offset + self.count, unit=unit)
        if state is None:
            done = True
            reward = 0
        else:
            done = False
            # self.count += 1
            idx = UNIT_MINUTE.index(unit)
            current_price = state[idx][1]

            temp = 100 * math.log(current_price / self.prev_notional_price)
            fee = 100 * math.log(1-FEE)

            if self.coin:
                # Hold or Sell
                if action == 0:
                    # Hold
                    reward = temp
                    self.coin = True
                else:
                    # Sell
                    reward = fee
                    self.coin = False
            else:
                if action == 0:
                    reward = 0
                    self.coin = False
                else:
                    reward = fee
                    self.coin = True
                    
            self.prev_notional_price = current_price
        state.append(float(self.coin))
        return state, reward, done, None

    