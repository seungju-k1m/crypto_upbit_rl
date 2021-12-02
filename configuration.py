from baseline.utils import jsonParser, writeTrainInfo
from baseline.utils import setup_logger
from datetime import datetime, timedelta

import logging
import os
import torch


_path_ = './cfg/ddqn.json'


_parser_ = jsonParser(_path_)
_data_ = _parser_.loadParser()
_key_ = list(_data_.keys())
DATA = _data_

USE_REDIS_SERVER = True
USE_PER = True
USE_RANDOM_START = False
ALPHA = 0.6
BETA = 0.4
GAMMA = .99
TARGET_FREQUENCY = 2500
N = 16
BATCHSIZE = 512
TOTAL_TRAINING_STEP = int(5e4)

FEE = _data_['FEE']
ACCESS_KEY = _data_['ACCESS_KEY']
SECRETE_KEY = _data_['SECRETE_KEY']
MARKET = _data_['MARKET']
LOG_MODE = _data_['LOG_MODE']

DEVICE = _data_["device"]
LEARNER_DEVICE = _data_["learner_device"]

UNROLL_STEP = _data_["unroll_step"]

OPTIM_INFO = _data_["optim"]


REPLAY_MEMORY_LEN = _data_["REPLAY_MEMORY_LEN"]

if "use_gym" in _key_:
    USE_GYM = _data_["use_gym"]
else:
    USE_GYM = False

try:
    REDIS_SERVER = _data_["redis_server"]
except:
    REDIS_SERVER = "localhost"

STARTDAY = '2021-09-13 00:00:00'
DURATION = 15

TIMEINTERVAL = 2


URL = "https://api.upbit.com"


_current_time_ = datetime.now()
_logger_ = logging.getLogger("Crypto_RL")
_logger_.setLevel(logging.ERROR)


_str_time_ = _current_time_.strftime("%m_%d_%Y_%H_%M_%S")
_log_path_ = os.path.join(
    './log', _str_time_
)

LOG_PATH = _log_path_

_indicator_log_path_ = os.path.join(
    _log_path_, 'indicator.log'
)

TRAIN_LOG_MODE = False
if TRAIN_LOG_MODE:
    if not os.path.isdir(_log_path_):
        os.mkdir(_log_path_)
    _train_log_path_ = os.path.join(
        _log_path_, "train.log"
    )
    TRAIN_LOGGER = setup_logger("train", _train_log_path_)
    _train_header_ = "step,norm,mean_value,entropy"
    TRAIN_LOGGER.info(_train_header_)

    _actor_log_path_ = os.path.join(
        _log_path_, "actor.log"
    )
    ACTOR_LOGGER = setup_logger("actor", _actor_log_path_)
    _actor_header_ = "day,income"
    ACTOR_LOGGER.info(_actor_header_)

# Logging !!
if LOG_MODE:
    if not os.path.isdir(_log_path_):
        os.mkdir(_log_path_)
    INDICT_LOGGER = setup_logger('indictator', _indicator_log_path_)
    _ask_log_path_ = os.path.join(
        _log_path_, 'ask.log'
    )
    _bid_log_path_ = os.path.join(
        _log_path_, 'bid.log'
    )
    _info_log_path_ = os.path.join(
        _log_path_, 'info.log'
    )

    ASK_LOGGER = setup_logger("ask", _ask_log_path_)
    BID_LOGGER = setup_logger("bid", _bid_log_path_)
    INFO_LOGGER = setup_logger("info", _info_log_path_)

    _ask_header_ = 'time,uuid,volume,market'
    _bid_header_ = 'time,uuid,price,market'
    _info_ = writeTrainInfo(_data_)

    INFO_LOGGER.info(_info_)
    ASK_LOGGER.info(_ask_header_)
    BID_LOGGER.info(_bid_header_)

# ------------------------------------------------

RENDER_TIME = 0
RUN_UNIT_TIME = 0
# 0:1, 1:5, 2:15, 3:60
# UNIT_MINUTE = [1, 5, 15, 60]
UNIT_MINUTE = [5]
RENDER_MODE = True