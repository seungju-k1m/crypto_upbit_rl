import os
import jwt
import uuid
import requests

import hashlib
from urllib.parse import urlencode
from baseline.utils import writeTrainInfo
from configuration import *


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

def generate_request(header, path='v1/accounts', query=None):
    path = os.path.join(URL, path)
    res = requests.get(
        path,
        headers=header,
        params=query
    )
    return res

def generate_all_procedure(payload, secrete_key, query, path='v1/accounts'):
    header = generate_author_header(payload, secrete_key)
    res = generate_request(header, path, query)
    return res

def preprocess_query_2_payload(query, access_key):
    query_string = urlencode(query).encode()
    m = hashlib.sha512()
    m.update(query_string)
    query_hash = m.hexdigest()
    payload = {
        'access_key':access_key,
        'nonce':str(uuid.uuid4()),
        'query_hash':query_hash,
        'query_hash_alg':"SHA512"
    }
    return payload

# --------------- CANDLE INFO ------------------------

def get_candle_minute_info(unit=1, count=1):
    t = "v1/candles/minutes/{}?market={}&count={}".format(unit, MARKET, count)

    url = os.path.join(
        URL, t
    )
    header = {"Accept": "application/json"}
    response = requests.request("GET", url, headers=header)
    data = response.json()
    return data

def get_candle_minute_info_remake(unit=1, count=1, to=None):
    if to is None:
        to = STARTDAY
    query = {
        'market':MARKET,
        'to':to,
        'count':count
    }
    query_string = urlencode(query).encode()
    m = hashlib.sha512()
    m.update(query_string)
    query_hash = m.hexdigest()
    payload = {
        'access_key':ACCESS_KEY,
        'nonce':str(uuid.uuid4()),
        'query_hash':query_hash,
        'query_hash_alg':"SHA512"
    }
    res = generate_all_procedure(payload, SECRETE_KEY, query, 'v1/candles/minutes/{}'.format(unit))
    return res.json()

def get_candle_day_info(count=1, market=None):
    if market is None:
        market = 'KRW-BTC'
    t = "v1/candles/days/?market={}&count={}".format(market, count)

    url = os.path.join(
        URL, t
    )
    header = {"Accept": "application/json"}
    response = requests.request("GET", url, headers=header)
    data = response.json()
    return data

def get_candle_week_info(count=1, market=None):
    if market is None:
        market = 'KRW-BTC'
    t = "v1/candles/weeks/?market={}&count={}".format(market, count)

    url = os.path.join(
        URL, t
    )
    header = {"Accept": "application/json"}
    response = requests.request("GET", url, headers=header)
    data = response.json()
    return data

def get_candle_month_info(count=1, market=None):
    if market is None:
        market = 'KRW-BTC'
    t = "v1/candles/months/?market={}&count={}".format(market, count)

    url = os.path.join(
        URL, t
    )
    header = {"Accept": "application/json"}
    response = requests.request("GET", url, headers=header)
    data = response.json()
    return data

# ----------------------------------------------------

# -------------------ACCUOUNT, BID, ASK ------------------

def get_account_info():
    payload = generate_payload(ACCESS_KEY)
    header = generate_author_header(payload, SECRETE_KEY)
    res = generate_request(header)
    account_info_json = res.json()[0]
    return account_info_json

def get_current_balance():
    info_json = get_account_info()
    balance = float(info_json['balance'])
    return balance

def get_market_info():
    query = {
        'market':MARKET
    }
    query_string = urlencode(query).encode()
    m = hashlib.sha512()
    m.update(query_string)
    query_hash = m.hexdigest()
    payload = {
        'access_key':ACCESS_KEY,
        'nonce':str(uuid.uuid4()),
        'query_hash':query_hash,
        'query_hash_alg':"SHA512"
    }
    res = generate_all_procedure(payload, SECRETE_KEY, query, 'v1/orders/chance')
    data = res.json()
    return data

def get_bid(portion=0.1):
    balance = get_current_balance()
    min_total = 5000.0

    portion = max(portion, min_total / balance)
    if balance < min_total:
        RuntimeWarning("Balance is less than min total")
        return None
    price = int(int(portion*balance) / int(min_total)) * min_total
    if price < min_total:
        price = min_total
    query = {
            'market':MARKET,
            'side':'bid',
            'price':str(price),
            'ord_type':'price'
        }
    payload = preprocess_query_2_payload(query, ACCESS_KEY)
    header = generate_author_header(
        payload, SECRETE_KEY
    )
    res = requests.post(os.path.join(
        URL, 'v1/orders'
    ), params=query, headers=header)
    data = res.json()
    keys = ['created_at', 'uuid', 'price', 'market']
    if 'error' in list(data.keys()):
        RuntimeWarning("BID Fails~~")
        return data
    if LOG_MODE:
        BID_LOGGER.info(
            '{},{},{},{}'.format(
                data[keys[0]], data[keys[1]], data[keys[2]], data[keys[3]]
            )
        )
    
    return data

def get_ask(portion=0.1):

    market_info = get_market_info()
    ask_account = market_info['ask_account']
    balance = ask_account['balance']
    volume = str(portion * float(balance))
    query = {
        'market':MARKET,
        'side':'ask',
        'volume':volume,
        'ord_type':'market'
    }
    payload = preprocess_query_2_payload(query, ACCESS_KEY)
    header = generate_author_header(
        payload, SECRETE_KEY
    )
    res = requests.post(os.path.join(
        URL, 'v1/orders'
    ), params=query, headers=header)
    data = res.json()
    if 'error' in list(data.keys()):
        RuntimeWarning("ASK Fails~~")
    keys = ['created_at', 'uuid', 'volume', 'market']
    if LOG_MODE:
        ASK_LOGGER.info(
            '{},{},{},{}'.format(
                data[keys[0]], data[keys[1]], data[keys[2]], data[keys[3]]
            )
        )
    return data


# ---------orderbook

def get_order_book(to):
    if to is None:
        to = STARTDAY
    query = {
        'market':MARKET,
        'to':to,
        'count':count
    }
    query_string = urlencode(query).encode()
    m = hashlib.sha512()
    m.update(query_string)
    query_hash = m.hexdigest()
    payload = {
        'access_key':ACCESS_KEY,
        'nonce':str(uuid.uuid4()),
        'query_hash':query_hash,
        'query_hash_alg':"SHA512"
    }
    res = generate_all_procedure(payload, SECRETE_KEY, query, 'v1/candles/minutes/{}'.format(unit))
    return res.json()