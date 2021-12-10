from configuration import ALG

if ALG == "APE_X":
    from APE_X.ReplayServer import ReplayServer

elif ALG == "CPT":
    from APE_X_Crypto.ReplayServer import ReplayServer

else:
    raise RuntimeError("!!")


if __name__ == "__main__":

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
