"""

DataPipeLine
    data -> pd.DataFrame
        columns: time, midpoint, acc_volume at every 1 minute

    methods

        get(self, start, end, unit):
            get data from start idx to end in the data of unit.
        

SetDataPipeLine

    pipeline의 list를 입력으로 받는 instance
    각 pipeline을 통합한다.

    get

DataPipeline_Sim

    Simulator에서 쓰이는 실제 DataPipeline module

    시뮬레이터는 7일간 이루어진다.

    시뮬레이터에 필요한 데이터는 8일치다.

    토요일 -> 다음주 일요일까지 데이터를 모아야 한 번의 시뮤레이터 완성.

    Args:
        offset -> 하루치, 즉 1440min

        count -> step !! 만약 데이터를 다 소모하게 된다면, done을 return한다.



    


DataPipeLine_Sim

"""
from copy import deepcopy
from typing import List
from utils.utils import get_candle_minute_info, get_candle_minute_info_remake
from datetime import datetime, timedelta
from configuration import UNIT_MINUTE
from typing import List, Dict
import _pickle as pickle

import pandas as pd
import numpy as np

import random
import os


def download_minute_data(to:datetime, duration:timedelta, unit=1) -> List:
    number_of_candle = duration / timedelta(minutes=unit)
    iteration = int(number_of_candle / 200)
    remainder = number_of_candle - iteration * 200
    
    to_str = to.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    data = []
    for _ in range(iteration):
        tmp = get_candle_minute_info_remake(
            unit=unit, count=200, to=to_str
        )
        to -= timedelta(minutes=200 * unit)
        
        to_str = to.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        data += tmp
    
    if remainder > 0:
        data += get_candle_minute_info_remake(
            unit=unit, count=int(remainder), to=to_str
        )
    return data[::-1]

def generate_sample():
    pass

template = {'time':[], 'midpoint':[], 'acc_volume':[]}

class DataPipeLine:
    
    def __init__(
        self,
        to:datetime,
        duration:timedelta,
        test=False
    ):
        if test:

            self.data = self.preprocess_raw_data(to, duration, 'test')
        else:
            self.data = self.preprocess_raw_data(to, duration, 'v0')
        self.length_info = deepcopy(template)

        for i in [1, 5, 15, 60]:
            self.length_info[i] = len(self.data[i])

    @staticmethod
    def preprocess_raw_data(to:datetime, duration:timedelta, kk=None):
        to_str = to.strftime("%Y-%m-%d")
        if kk is None:
            kk = 'v0'
        save_path = './data/{}/'.format(kk) + to_str + '_'+ str(duration) + '.bin'

        if os.path.isfile(save_path):
            file = open(save_path, 'rb')
            raw_data = file.read()
            output = pickle.loads(raw_data)
        else:
            base_unit = [1, 5, 15, 60]
            output = {}
            for unit in base_unit:
                data = download_minute_data(to, duration,unit)
            
                data_df = pd.DataFrame(template)
                for d in data:
                    tmp = {
                        'time':d['candle_date_time_utc'],
                        'midpoint':(d['trade_price'] + d['opening_price'])/2,
                        'acc_volume':d['candle_acc_trade_volume']
                        }
                    data_df = data_df.append(
                        tmp, ignore_index=True
                    )
                data_df = data_df.set_index(data_df['time'])
                data_df.sort_index()
                output[unit] = data_df
            
            file = open(save_path, 'wb')
            file.write(pickle.dumps(output))
            file.close()
        return output

    def get(self, start, end, unit):
        if unit not in [1, 5, 15, 60]:
            raise RuntimeError("UNIT  in [1, 5, 15, 60")
        data = self.data[unit].iloc[start:end]
        return data


class SetDataPipeLine:

    def __init__(
        self,
        pipelines: List[DataPipeLine]
    ):
        # self.pipelines = pipelines

        self.data = template

        self.length_info = {1:0, 5:0, 15:0, 60:0}

        for i in [1, 5, 15, 60]:
            len_info = 0
            df_info = []
            for p in pipelines:
                len_info += len(p.data[i])
                df_info.append(p.data[i])
            df = pd.concat(df_info)
            df.sort_index()
            self.data[i] = df
            self.length_info[i] = len_info
    
    
    def get(self, start, end, unit):
        if unit not in [1, 5, 15, 60]:
            raise RuntimeError("UNIT  in [1, 5, 15, 60")
        return self.data[unit].iloc[start:end]
    
    def update(self, pipelines):
        for i in [1, 5, 15, 60]:
            len_info = 0
            df_info = []
            for p in pipelines:
                len_info += len(p.data[i])
                df_info.append(p.data[i])
            df = pd.concat(df_info)
            df.sort_index()
            self.data[i] = df
            self.length_info[i] = len_info


class DataPipeLine_Sim:

    def __init__(
        self,
        data:SetDataPipeLine or DataPipeLine,
        offset:int = 1440,
        size=24
    ):

        self.offset = offset
        self.offset = int(offset * (1 + 0.1 * random.random()))
        self.count = 0
        self.data = data
        self.end_step = len(data.data[1]) - self.count
        self.size = size
    
    def update(self, pipelines):
        self.data.update(pipelines)

    @staticmethod
    def process(data:pd.DataFrame):
        # output = {'time':None, 'midpoint':None, 'acc_volume':None}
        output = [data['time'].to_numpy(), data['midpoint'].to_numpy(), data['acc_volume'].to_numpy()]
        return output

    def reset(self) -> dict:
        self.count = 0
        output = {}
        for unit in [1, 5, 15, 60]:
            output[unit] = self.process(self.data.get(int(self.offset/unit)-self.size, int(self.offset/unit), unit))
        return output
    
    def step(self, duration=1) -> Dict:
        "duration만큼, stepping !!"
        
        count = self.count + self.offset
        output = {}
        if count < self.end_step:
            for unit in [1, 5, 15, 60]:
                output[unit] = self.process(self.data.get(int(count/unit), int((count+duration)/unit), unit))
            self.count += duration
            return output, False
        else:
            self.count += duration
            return output, True