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