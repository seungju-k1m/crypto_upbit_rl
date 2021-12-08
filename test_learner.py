from configuration import CRYPTO_MODE
# from RL_DQN.Learner import Learner
# from RL_Crypto.Learner import Learner as Learner_cpt
from RL_R2D2.Learner import Learner

if __name__ == "__main__":
    l = Learner()
    l.run()