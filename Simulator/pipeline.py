from utils.utils import get_candle_minute_info_remake
from configuration import STARTDAY
from Simulator.renderer import Renderer
import numpy as np
import pandas as pd

import _pickle as pickle
import os


from datetime import datetime, timedelta

"""
분봉을 만들고, 그것을 기점으로, 시간, day, week에 해당하는 분봉

Indictator를 계산 !!
"""

class DataPipeLine:
    def __init__(
        self,
        to=None,
        duration=1):

        if to is None:
            to = STARTDAY
        self.to = to
        unit_minute=[1, 5, 15, 60]
        self.unit_minute = unit_minute
        self._data = [[] for i in range(len(unit_minute))]
        path = './data/{}_duration_{}'.format(to, duration)
        if not os.path.isfile(path):
            for i, u_m in enumerate(unit_minute):
                self._data[i] += self.init_minute_data(
                    to, duration, u_m
                )
            bin_data = pickle.dumps(self._data)
            file = open(path, 'wb')
            file.write(bin_data)
            file.close()
        else:
            file = open(path, 'rb')
            self._data = pickle.loads(
                file.read()
            )
            file.close()
        print("Load Data")

        self.current_time = datetime.fromisoformat(
            self._data[0][0]['candle_date_time_utc']
        )

        self.renderer = Renderer()
    
    def render(self):
        pass

    @staticmethod
    def init_minute_data(to, duration, unit_minute):
        to_datetime = datetime.fromisoformat(to)
        
        duration = duration
        miniute_duration = duration * 1440
        
        per_data = unit_minute / miniute_duration

        number_of_iteration = int(1 / per_data)

        iteration = int(number_of_iteration / 200)
        remainder = (number_of_iteration - iteration * 200) !=0
        data = []
        to_tmp = to
        for _ in range(iteration):
            data += get_candle_minute_info_remake(
                unit=unit_minute, count=200, to=to_tmp
            )
            to_datetime -= timedelta(minutes=unit_minute * 200)
            to_tmp = to_datetime.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        
        if remainder:
            remainder_count = number_of_iteration - iteration * 200
            data += get_candle_minute_info_remake(
                unit=unit_minute, count=remainder_count, to=to_tmp
            )
        return data[::-1]
    
    @staticmethod
    def preprocess_data(data):
        unit_minute=[1, 5, 15, 60]
        to = data[0][0]['candle_date_time_utc']

        # convert raw_data to pd.DataFrame

        # 