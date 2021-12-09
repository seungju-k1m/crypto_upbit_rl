from RL_R2D2.Player import Player
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
    args = parser.parse_args()
    num_worker = args.num_worker
    start_idx = args.start_idx

    ray.init(num_cpus=num_worker)
    player = ray.remote(num_cpus=1)(Player)
    players = []
    for i in range(num_worker):
        players.append(
            player.remote(idx=i + start_idx)
        )
    
    ray.get([p.run.remote() for p in players])