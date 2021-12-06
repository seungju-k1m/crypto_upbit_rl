from configuration import CRYPTO_MODE, REDIS_SERVER_PUSH
from RL_DQN.ReplayServer import ReplayServer
from RL_Crypto.ReplayServer import ReplayServer as ReplayServer_cpt


if __name__ == "__main__":
    if CRYPTO_MODE:
        server = ReplayServer_cpt()
    else:
        server = ReplayServer()
    server.run()

    # redis_server = [redis.StrictRedis(host=REDIS_SERVER_PUSH, port=6379) for i in range(4)]
    # mm = []
    # while 1:
    #     k = server.sample()
    #     if k is False:
    #         continue

    #     mm.append(k)
    #     if len(mm) == 4:
    #         for r, m in zip(redis_server, mm):
    #             rr = dill.dumps(r)
    #             call.remote(rr, m)
    #         mm.clear()
