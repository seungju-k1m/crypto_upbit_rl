import gym
import time


if __name__ == "__main__":
    env = gym.make('BreakoutNoFrameskip-v4')
    env.seed(123)
    env.reset()

    done = False

    while done is False:
        _, __, done, ___ = env.step(env.action_space.sample())
        env.render()
        time.sleep(0.1)