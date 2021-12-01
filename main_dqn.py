
from RL_DQN.Player import Player
from RL_DQN.ReplayMemory import Replay
from RL_DQN.Learner import Learner
from argparse import ArgumentParser

import ray

parser = ArgumentParser()

parser.add_argument(
    "--num-worker",
    type=int,
    default=2
)

parser.add_argument(
    "--start-idx",
    type=int,
    default=0
)


if __name__ == "__main__":

    # -------------- Player ----------------
    args = parser.parse_args()
    num_worker = args.num_worker
    start_idx = args.start_idx
    # num_worker = 2
    p = Player()
    # p.test_gym()
    p.gym_run()
    
    ray.init(num_cpus=num_worker)
    Player = ray.remote(
            num_cpus=1)(Player)
    players = []
    for i in range(num_worker):
        players.append(
            Player.remote(idx=start_idx+i)
        )

    ray.get([p.gym_run.remote() for p in players])
    
    # ---------------------------------------

    