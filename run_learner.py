from configuration import CRYPTO_MODE
from RL_DQN.Learner import Learner
from RL_Crypto.Learner import Learner as Learner_cpt

if __name__ == "__main__":

    if CRYPTO_MODE:
        l = Learner_cpt()
    else:
        l = Learner()
    l.run()