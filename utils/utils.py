import os
import jwt
import uuid
import requests


URL = "https://api.upbit.com"


def generate_payload(access_key: str):
    tmp = {
        "access_key":access_key,
        "nonce":str(uuid.uuid4())
    }
    return tmp

def generate_author_header(payload:dict, secrete_key: str):
    token = jwt.encode(payload, secrete_key)
    author_token = "Bearer {}".format(token)
    header = {"Authorization":author_token}
    return header

def generate_request(header):
    path = os.path.join(URL, 'v1', 'accounts')
    res = requests.get(
        path,
        headers=header
    )
    return res