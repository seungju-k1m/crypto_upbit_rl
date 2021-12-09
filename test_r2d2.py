from RL_R2D2.Player import Player

import ray

if __name__ == "__main__":
    # p = Player()
    # p.run()

    ray.init(num_cpus=2)
    player = ray.remote(num_cpus=1)(Player)
    players = []
    for i in range(2):
        players.append(
            player.remote(idx=i * 8)
        )
    
    ray.get([p.run.remote() for p in players])