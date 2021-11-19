from utils.utils import *
from RL.Configuration import Cfg

class Agent:

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self._current_info :dict = None
        self.get_account_info()

    def get_account_info(self):
        payload = generate_payload(self.cfg.access_key)
        header = generate_author_header(payload, self.cfg.secrete_key)
        res = generate_request(header)
        account_info_json = res.json()[0]
        info = ["balance"]
        self._current_info = {}
        for i in info:
            self._current_info[i] = account_info_json[i]
        print("Current Account Information !!")
        print("------------------------")
        print(self._current_info)
        print("------------------------")
