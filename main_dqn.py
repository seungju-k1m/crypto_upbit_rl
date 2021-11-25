
from RL_DQN.Player import Player
from RL_DQN.ReplayMemory import Replay
from RL_DQN.Learner import Learner
from argparse import ArgumentParser

import ray

parser = ArgumentParser()

parser.add_argument(
    "--num-worker",
    type=int,
    default=1
)


if __name__ == "__main__":

    # -------------- Player ----------------
    args = parser.parse_args()
    num_worker = args.num_worker
    num_worker = 1
    p = Player()
    p.run()
    
    ray.init(num_cpus=num_worker)
    Player = ray.remote(
            num_cpus=1)(Player)
    players = []
    for i in range(num_worker):
        players.append(
            Player.remote()
        )

    ray.get([p.gym_run.remote() for p in players])
    
    # ---------------------------------------

    