from datetime import datetime

from configuration import FEE

from Simulator.renderer_v1 import Renderer
from Simulator.pipeline_v1 import *

from typing import List

import numpy as np
import pandas as pd

import random
import math
import os


LEVERAGE = 100

def generate_random_start(day):
    data_list = os.listdir('./data/v0')
    data_list.sort()
    total_len = len(data_list)

    start_id = random.randint(1, total_len-10)

    data_list = [data_list[i] for i in range(start_id, start_id+day)]
    pipelines = []
    for d in data_list:
        k_list = d.split(',')
        kk = k_list[0]
        kk = kk[:10]
        to = datetime.fromisoformat(kk)
        pipelines.append(
            DataPipeLine(to, duration=timedelta(days=1))
        )
    return pipelines

def generate_test_start(idx=0):
    data_list = os.listdir('./data/test')
    data_list.sort()
    total_len = len(data_list)

    start_id = idx

    data_list = [data_list[i] for i in range(start_id, start_id+5)]
    pipelines = []
    for d in data_list:
        k_list = d.split(',')
        kk = k_list[0]
        kk = kk[:10]
        to = datetime.fromisoformat(kk)
        pipelines.append(
            DataPipeLine(to, duration=timedelta(days=1), test=True)
        )
    return pipelines


class PortFolio:
    """
        Args:
            KRW Account

                balance

                is_krw


            ETC Account

                balance

                is_coin

                average_price
                    if there is no coin, -1
            
            current_value
        
        Methods
            update

            rebalance

    """

    __slots__=["KRW_Balance", "Coin_Balance", "Ticker", "Is_Coin", "Average_Price", "Current_Value", "Is_KRW", "Info"]

    def __init__(
        self,
        krw_balance=0,
        coin_balance=1,
        ticker="ETH",
        is_coin=True,
        Average_price=5000000,
    ):

        self.KRW_Balance = 0
        self.Coin_Balance = 1
        self.Ticker = ticker
        self.Is_Coin = is_coin
        self.Is_KRW = False
        self.Average_Price = Average_price
        self.Current_Value = None
        self.update()

        self.Coin_Balance = random.random()
        self.KRW_Balance = self.Current_Value * (1 - self.Coin_Balance)
        self.update()

        self.Info = {}
    
    def update(self):
        self.Current_Value = self.KRW_Balance + self.Average_Price * self.Coin_Balance
        if self.Coin_Balance < 0.001:
            self.Is_Coin = False
        else:
            self.Is_Coin = True
        
        if self.KRW_Balance < 1000:
            self.Is_KRW = False
        else:
            self.Is_KRW = True
        if self.Coin_Balance < 0:
            raise RuntimeError("Coin Balance must be positive.")
    
    def print(self):
        print(
            """
            ----------------
            KRW_BALANCE: {:.3f}

            TICKER: {}

            COIN_BALANCE: {:.5f}

            COIN_PRICE: {:.3f}

            TOTAL_VALUE: {:.3f}

            IS_KRW:{}

            IS_COIN:{}
            
            """.format(
                self.KRW_Balance,
                self.Ticker,
                self.Coin_Balance,
                self.Average_Price,
                self.Current_Value,
                self.Is_KRW,
                self.Is_Coin
            )
        )

    def get_info(self):
        ss = ["KRW_Balance", "Coin_Balance", "Ticker", "Is_Coin", "Average_Price", "Current_Value", "Is_KRW", "Info"]
        for s in ss:
            self.Info[s] = getattr(self, s)
        
        return deepcopy(self.Info)


class Simulator:

    def __init__(
        self,
        pipelines: List[DataPipeLine] = None,
        unit_step=15,
        test=False,
        size=24,
        day=3
    ):
        if pipelines is None:
            if test:
                pipelines = generate_test_start(1)
                self.idx = 1
            else:
                mz = int(size / 24)
                pipelines = generate_random_start(day + mz)
        self.size = size
        setpipeline = SetDataPipeLine(pipelines)
        offset = 1440 * int(self.size / 24)
        self.pipe = DataPipeLine_Sim(setpipeline, offset=offset, size=size)
        self.renderer = Renderer(self.pipe)
        self.unit_step = unit_step
        self.day = day
    
    def reset_pipeline(self, test=False):
        if test:
            try:
                self.idx += 3
            except:
                self.idx = 1
            pipelines = generate_test_start(self.idx)
            print(self.idx)
            
        else:
            mz = int(self.size / 24)
            pipelines = generate_random_start(self.day + mz)
        # setpipeline = SetDataPipeLine(pipelines)
        # self.pipe = DataPipeLine_Sim(setpipeline)
        # self.renderer = Renderer(self.pipe)
        self.pipe.update(pipelines)
        
    def reset(self, test=False, port=None):
        self.reset_pipeline(test)
        obs = self.renderer.reset()
        current_price = float(self.renderer.y_vec[1][-1])
        if port is None:
            self.portfolio = PortFolio(
                0, 1, Average_price=current_price, is_coin=True
            )
        else:
            self.portfolio = port
        info = self.portfolio.get_info()
        return (obs, info)
    
    def rebalance_KRW_2_COIN(self, amount=0.25, current_price=0):
        previous_value = deepcopy(self.portfolio.Current_Value)
        reward = 0
        idle = False
        if self.portfolio.Is_KRW:
            
            KRW = self.portfolio.KRW_Balance * amount 
            KRW_2_COIN = KRW / current_price * (1 - FEE)

            self.portfolio.KRW_Balance -= KRW
            self.portfolio.Coin_Balance += KRW_2_COIN
            self.portfolio.Average_Price = current_price

            self.portfolio.update()
            current_value = deepcopy(self.portfolio.Current_Value)

            reward =  LEVERAGE * math.log(current_value / previous_value)

        else:
            self.portfolio.Average_Price = current_price
            self.portfolio.update()
            current_value = deepcopy(self.portfolio.Current_Value) * (1 - FEE)

            reward = LEVERAGE * math.log(current_value / previous_value)
            idle = True
        
        return reward, idle
    
    def rebalance_COIN_2_KRW(self, amount=0.25, current_price=0):
        previous_value = deepcopy(self.portfolio.Current_Value)
        reward = 0
        idle = False
        if self.portfolio.Is_Coin:
            
            COIN = self.portfolio.Coin_Balance * amount 
            COIN_2_KRW =COIN * (current_price) * (1 - FEE)

            self.portfolio.Coin_Balance -= COIN
            self.portfolio.KRW_Balance += COIN_2_KRW 
            self.portfolio.Average_Price = current_price
            self.portfolio.update()

            current_value = deepcopy(self.portfolio.Current_Value)

            reward = LEVERAGE * math.log(current_value / previous_value)

        else:
            self.portfolio.Average_Price = current_price
            self.portfolio.update()
            current_value = deepcopy(self.portfolio.Current_Value) * (1 - FEE)

            reward = LEVERAGE * math.log(current_value / previous_value)
            idle = True
        
        return reward, idle

    def step(self, action=0):
        """
            Action Space

                0 -> do nothing

                1 -> KRW -> COIN (25%)
                2 -> KRW -> COIN (50%)
                3 -> KRW -> COIN (75%)
                4 -> KRW -> COIN (100%)

                5 -> COIN -> KRW (25%)
                6 -> COIN -> KRW (25%)
                7 -> COIN -> KRW (25%)
                8 -> COIN -> KRW (25%)
        """
        obs, done = self.renderer.step(duration=self.unit_step)
        _, candle_info = obs
        y1, y2 = candle_info

        current_price = y1[1][-1]

        amount_list = [.05, .10, .25, 0.5]

        # rebalance
        # if action == 0:
        #     reward = 0
        #     prev_value = deepcopy(self.portfolio.Current_Value)
        #     self.portfolio.Average_Price = current_price
        #     self.portfolio.update()
        #     value = deepcopy(self.portfolio.Current_Value)
        #     reward = 100 * math.log(value / prev_value)
        
        # elif action == 1:
        #     reward = self.rebalance_KRW_2_COIN(1, current_price)
        # else:
        #     reward = self.rebalance_COIN_2_KRW(1, current_price)

        if action == 0:
            prev_value = deepcopy(self.portfolio.Current_Value)
            self.portfolio.Average_Price = current_price
            self.portfolio.update()
            value = deepcopy(self.portfolio.Current_Value)
            reward_ = LEVERAGE * math.log(value / prev_value)
            idle = False
        elif action < 5:
            idx = int(action - 1)
            amount = amount_list[idx]
            reward_, idle = self.rebalance_KRW_2_COIN(amount, current_price)
        elif action < 9:
            idx = int(action - 5)
            amount = amount_list[idx]
            reward_, idle = self.rebalance_COIN_2_KRW(amount, current_price)
        else:
            raise RuntimeWarning("Action Space !!")
        
        if idle:
            reward = reward_
        else:
            reward = reward_
        
        # reward_ -> raw !!
        # reward -> reward_ + idle !!
        
        info = self.portfolio.get_info()

        return (obs, info), reward, done, idle

    def print(self):
        self.portfolio.print()