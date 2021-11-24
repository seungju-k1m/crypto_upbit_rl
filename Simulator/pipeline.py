from pandas.core.frame import DataFrame
from utils.utils import get_candle_minute_info_remake
from configuration import STARTDAY
from Simulator.renderer import Renderer
import numpy as np
import pandas as pd

import _pickle as pickle
import os


from datetime import datetime, timedelta
from configuration import UNIT_MINUTE

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
        self.UNIT_MINUTE = UNIT_MINUTE
        self._data = [[] for i in range(len(UNIT_MINUTE))]
        path = './data/raw/{}_duration_{}'.format(to, duration)
        self.path = './data/process/{}_duration_{}'.format(to, duration)
        if not os.path.isfile(path):
            for i, u_m in enumerate(UNIT_MINUTE):
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
        self.preprocess_data()
        self.data_len = [len(self.data[i]) for i in UNIT_MINUTE]
        # midpoint

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
            tmp = get_candle_minute_info_remake(
                unit=unit_minute, count=200, to=to_tmp
            )
            data += tmp
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
    
    def preprocess_data(self):
        to = self.path + ' process.bin'
        if not os.path.isfile(to):
            output = {}
            # UNIT_MINUTE=[1, 5, 15, 60]
            for d, u_m in zip(self._data, UNIT_MINUTE):
                output[u_m] = None
                for i, dict_d in enumerate(d):
                    t = dict_d['candle_date_time_utc']
                    _midpoint = (dict_d['trade_price'] + dict_d['opening_price']) / 2
                    _delta = dict_d['trade_price'] - dict_d['opening_price']
                    acc_volume = dict_d['candle_acc_trade_volume']
                    tmp_output = [[t, _midpoint, _delta, acc_volume]]
                    tmp = pd.DataFrame(tmp_output, columns=['time', 'midpoint', "delta", 'acc_volume'], index=[i])
                    # tmp = pd.DataFrame.from_dict(tmp_output, orient='columns', index=)
                    if output[u_m] is None:
                        output[u_m] = tmp
                    else:
                        output[u_m] = pd.concat([output[u_m], tmp])
                output[u_m].set_index(output[u_m]['time'])
                output[u_m].drop(output[u_m].columns[0], axis=1)
            bin_output = pickle.dumps(output)
            file = open(to, 'wb')
            file.write(bin_output)
            file.close()
        else:
            file = open(to, 'rb')
            output = file.read()
            output = pickle.loads(output)
            file.close()
        self.data = output

    def get(self, ind, unit=1):
        idx = UNIT_MINUTE.index(unit)
        if ind > (self.data_len[idx] - 1):
            return None
        
        output = []
        for i in UNIT_MINUTE:
            ii = int(ind / i * unit)
            tmp = self.data[i].iloc[ii].to_numpy()
            tmp[0] = np.datetime64(tmp[0])
            output.append(
                tmp
            )
        return output
        