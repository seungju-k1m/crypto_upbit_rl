from baseline.utils import writeTrainInfo

from utils.utils import *
import time
from datetime import datetime
from Simulator.pipeline import DataPipeLine
from Simulator.simulator import Simulator
from RL.Player import Player
from RL.ReplayMemory import Replay
from RL.Learner import Learner
from argparse import ArgumentParser

import ray

parser = ArgumentParser()

parser.add_argument(
    "--num-worker",
    type=int,
    default=1
)


if __name__ == "__main__":


    # unit = 15
    # count = 3
    # data = get_candle_minute_info(unit, count, MARKET)
    # data = get_candle_day_info(count, MARKET)
    # data = get_candle_week_info(count, MARKET)
    # data = get_candle_month_info(count, MARKET)
    # for d in data:
    #     print(
    #         writeTrainInfo(d)
    #     )
    
    # account_info = get_account_info()
    # info_str = writeTrainInfo(account_info)
    # balance = get_current_balance()

    # market_info = get_market_info()
    # info = writeTrainInfo(market_info)

    # data = get_candle_minute_info_remake(10, 10)
    # data = data[::-1]
    # for d in data:
    #     dd = writeTrainInfo(d)
    #     print(dd)
    # print(info)
    # ---------------bid & ask --------------
    # get_bid(portion=1.0)
    # get_ask(portion=1.0)
    # -----------------------------------------

    # -------------DataPipeLine---------------

    # pipeline = DataPipeLine(
    #     to = '2021-10-13 00:00:00',
    #     duration=2
    # )
    # data = get_order_book()
    # k = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data['timestamp']/1000))

    # info = writeTrainInfo(data)
    # info_orderbook = data['orderbook_units'][0]
    # info_orderbook = writeTrainInfo(info_orderbook)
    # print(info)
    # -----------------------------------------


    # --------------Simulator----------------

    # sim = Simulator(to='2021-09-13 00:00:00', duration=15)
    # sim.init_random()
    # sim.reset()
    # # sim.renderer.set_ylim(
    # #     55000000, 80000000
    # # )
    # while 1:
    #     s, reward, done, info = sim.step(unit=UNIT_MINUTE[RENDER_TIME])
    #     # s, reward, done, info = sim.step(unit=1)
    #     if done:
    #         break
    #     sim.render(s, mode="2")
    # sim.reset()

    # ---------------------------------------
    # p = Player()
    # p.get_sample()
    # # p.run()
    # # ------------ Replay Memory ------------
    # m = Replay()
    # t = m.buffer('./sample.bin')
    # # ---------------------------------------

    # # ------------ Replay Memory ------------
    # l = Learner()
    # while 1:
    #     l.train(t)
    # -------------- Player ----------------
    args = parser.parse_args()
    num_worker = args.num_worker
    num_worker = 2
    
    ray.init(num_cpus=num_worker)
    Player = ray.remote(
            num_cpus=1)(Player)
    players = []
    for i in range(num_worker):
        players.append(
            Player.remote()
        )

    ray.get([p.run.remote() for p in players])
    
    # ---------------------------------------

    