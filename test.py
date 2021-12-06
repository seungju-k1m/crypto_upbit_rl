from os import pipe
from Simulator.pipeline_v1 import DataPipeLine, SetDataPipeLine, DataPipeLine_Sim
from Simulator.simulator_v1 import Simulator
from Simulator.renderer_v1 import Renderer

from datetime import datetime, timedelta

from RL_Crypto.Player import Player

import numpy as np
import time


if __name__ == "__main__":
    p = Player()
    p.run()


    sim = Simulator()
    
    cumulative_reward = 0

    for j in range(10):
        obs = sim.reset()
        sim.portfolio.print()

        for t in range(10000):
            m = time.time()
            action = np.random.randint(0, 8)
            o, r, done, _ = sim.step(action)
            cumulative_reward += r
            if (t+1) % 100 == 0:
                sim.portfolio.print()
            if done:
                break
        sim.portfolio.print()
        print("Simulation Is Done")
        print(cumulative_reward)

    # to = datetime(2020, 1, 2, 0, 0)
    # to_list = [to - timedelta(days=1)]
    # pipe_list = []
    # for i in range(7):
    #     to_list.append(
    #         to + timedelta(days=i)
    #     )
    # duration = timedelta(days=1)
    # for t in to_list:
    #     pipe_list.append(
    #         DataPipeLine(t, duration)
    #     )


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
