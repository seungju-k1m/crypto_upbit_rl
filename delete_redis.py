from configuration import REDIS_SERVER
import redis


c = redis.StrictRedis(host=REDIS_SERVER, port=6379)

name = c.scan()

if len(name[-1]) > 0:
    c.delete(*name[-1])
    print("DELETE")