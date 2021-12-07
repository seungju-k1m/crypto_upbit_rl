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


def generate_random_start():
    data_list = os.listdir('./data/v0')
    data_list.sort()
    total_len = len(data_list)

    start_id = random.randint(1, total_len-9)

    data_list = [data_list[i] for i in range(start_id, start_id+3)]
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

        self.KRW_Balance = krw_balance
        self.Coin_Balance = coin_balance
        self.Ticker = ticker
        self.Is_Coin = is_coin
        self.Is_KRW = False
        self.Average_Price = Average_price
        self.Current_Value = None
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
            
            """.format(
                self.KRW_Balance,
                self.Ticker,
                self.Coin_Balance,
                self.Average_Price,
                self.Current_Value
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
        unit_step=15
    ):
        if pipelines is None:
            pipelines = generate_random_start()
        setpipeline = SetDataPipeLine(pipelines)
        self.pipe = DataPipeLine_Sim(setpipeline)
        self.renderer = Renderer(self.pipe)
        self.unit_step = unit_step
    
    def reset_pipeline(self):
        pipelines = generate_random_start()
        # setpipeline = SetDataPipeLine(pipelines)
        # self.pipe = DataPipeLine_Sim(setpipeline)
        # self.renderer = Renderer(self.pipe)
        self.pipe.update(pipelines)
        
    def reset(self):
        self.reset_pipeline()
        obs = self.renderer.reset()
        current_price = float(self.renderer.y_vec[1][-1])
        self.portfolio = PortFolio(
            0, 1, Average_price=current_price, is_coin=True
        )
        info = self.portfolio.get_info()
        return (obs, info)
    
    def rebalance_KRW_2_COIN(self, amount=0.25, current_price=0):
        previous_value = deepcopy(self.portfolio.Current_Value)
        reward = 0
        if self.portfolio.Is_KRW:
            
            KRW = self.portfolio.KRW_Balance * amount * (1 - FEE)
            KRW_2_COIN = KRW / (current_price+1e-3)

            self.portfolio.KRW_Balance *= (1 - amount)
            self.portfolio.Coin_Balance += KRW_2_COIN
            self.portfolio.Average_Price = current_price

            self.portfolio.update()
            current_value = deepcopy(self.portfolio.Current_Value)

            reward =  100 * math.log(current_value / previous_value)

        else:
            self.portfolio.Average_Price = current_price
            self.portfolio.update()
            current_value = deepcopy(self.portfolio.Current_Value)

            reward = 100 * math.log(current_value / previous_value)
        
        return reward
    
    def rebalance_COIN_2_KRW(self, amount=0.25, current_price=0):
        previous_value = deepcopy(self.portfolio.Current_Value)
        reward = 0
        if self.portfolio.Is_Coin:
            
            COIN = self.portfolio.Coin_Balance * amount
            COIN_2_KRW =COIN * (current_price)

            self.portfolio.Coin_Balance *= (1 - amount)
            self.portfolio.KRW_Balance += COIN_2_KRW * (1 - FEE)
            self.portfolio.Average_Price = current_price
            self.portfolio.update()

            current_value = deepcopy(self.portfolio.Current_Value)

            reward = 100 *  math.log(current_value / previous_value)

        else:
            self.portfolio.Average_Price = current_price
            self.portfolio.update()
            current_value = deepcopy(self.portfolio.Current_Value)

            reward = 100 * math.log(current_value / previous_value)
        
        return reward

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
        reward = deepcopy(self.portfolio.Current_Value)
        _, candle_info = obs
        y1, y2 = candle_info

        current_price = y1[1][-1]

        amount_list = [.25, .5, .75, 1.0]

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
            reward = 0
            prev_value = deepcopy(self.portfolio.Current_Value)
            self.portfolio.Average_Price = current_price
            self.portfolio.update()
            value = deepcopy(self.portfolio.Current_Value)
            reward = 100 * math.log(value / prev_value)

        elif action < 5:
            idx = int(action - 1)
            amount = amount_list[idx]
            reward = self.rebalance_KRW_2_COIN(amount, current_price)
        elif action < 9:
            idx = int(action - 5)
            amount = amount_list[idx]
            reward = self.rebalance_COIN_2_KRW(amount, current_price)
        else:
            raise RuntimeWarning("Action Space !!")
        
        info = self.portfolio.get_info()

        return (obs, info), reward, done, {}

    def print(self):
        self.portfolio.print()