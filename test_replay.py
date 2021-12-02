from RL_DQN.ReplayMemory import Replay_Server


if __name__ == "__main__":
    s = Replay_Server()
    s.start()

    while 1:
        m = 1