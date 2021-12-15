from os import pipe
from Simulator.pipeline_v2 import DataPipeLine, SetDataPipeLine, DataPipeLine_Sim
from Simulator.simulator_v2 import Simulator
from Simulator.renderer_v2 import Renderer, Animator

from datetime import datetime, timedelta

from R2D2_Crypto.Player import Player

import numpy as np
import time


if __name__ == "__main__":
    # p = Player()
    # p.run()


    # plot_unit = 30
    # sim = Simulator(size=24, day=1, unit=15)
    # sim.reset()
    # for i in range(100):
    #     sim.step(0)

    # a = Animator(sim.pipe, plot_unit=plot_unit)
    # a.plot()
    
    # cumulative_reward = 0

    # for j in range(10):
    #     obs = sim.reset()
    #     sim.portfolio.print()
    #     m = time.time()
    #     for t in range(100000):
            
    #         action = np.random.randint(0, 8)
    #         o, r, done, _ = sim.step(action)
    #         cumulative_reward += r
    #         if (t+1) % 500 == 0:
    #             sim.portfolio.print()
    #         if done:
    #             break
    #     sim.portfolio.print()
    #     print("Simulation Is Done")
    #     print(cumulative_reward)
    #     print(time.time() - m)

    to = datetime(2020, 1, 2, 0, 0)
    to_list = [to - timedelta(days=1)]
    pipe_list = []
    for i in range(365):
        to_list.append(
            to + timedelta(days=i)
        )
    duration = timedelta(days=1)
    for t in to_list:
        pipe_list.append(
            DataPipeLine(t, duration, False)
        )


    # set_pipeline = SetDataPipeLine(pipe_list)
    # sim_pipeline = DataPipeLine_Sim(set_pipeline)

    # # output = sim_pipeline.step(15)
    # # print(output)

    # renderer = Renderer(
    #     sim_pipeline
    # )
    # obs = renderer.reset()
    # # next_obs, done = renderer.step(15)
    # # next_obs = renderer.render()
    # print("chgeck")
    # i = 1
    # while 1:
    #     renderer.step(15)
    #     print(i)
    #     i+=1
    #     renderer.render()
