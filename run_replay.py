from RL_DQN.ReplayServer import ReplayServer
import ray


if __name__ == "__main__":
    server = ReplayServer()
    ray.init(num_cpus=2)
    server.run()