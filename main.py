from baseline.utils import writeTrainInfo

from utils.utils import *

from Simulator.pipeline import DataPipeLine

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


    # -------------DataPipeLine---------------

    pipeline = DataPipeLine(
        to = '2021-10-13 00:00:00',
        duration=7
    )
