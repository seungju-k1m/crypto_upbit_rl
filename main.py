from RL.Agent import Agent
from RL.Configuration import Cfg


if __name__ == "__main__":
    path = './cfg/demo.json'
    cfg = Cfg(path)
    agent = Agent(cfg)