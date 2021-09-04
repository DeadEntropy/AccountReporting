from json import JSONDecodeError

import yfinance as yf
import pandas as pd
from cachetools import cached, LRUCache
from bkanalysis.portfolio import cache
import requests
import re

mem_cache_history = LRUCache(maxsize=1024)
mem_cache_currency = LRUCache(maxsize=1024)

__query_yahoo_url = "https://query2.finance.yahoo.com/v1/finance/search"
__isin_to_symbol_mapping_path = r'isin_cache.json'
__isin_cache = cache.CacheDict(__isin_to_symbol_mapping_path)

regex_ticker = re.compile(r'^[a-zA-Z][a-zA-Z][0-9]+$')

__currencies = ['EUR', 'USD', 'GBP', 'CHF', 'JPY', 'CAD', 'AUD', 'KRW', 'CNH']
__crypto = ['BTC', 'ETH']


def get_spot_price(instr, currency):
    if instr is None:
        return None
    if instr == currency:
        return 1.0
    elif (len(instr) == 3) and (instr in __currencies or instr in __crypto):  # its a currency/crypto
        if instr in __crypto:
            symbol = f'{instr}-{currency}'
        else:
            symbol = f'{instr}{currency}=X'
        try:
            return __get_history(symbol, '1y').iloc[0].Close
        except JSONDecodeError as e:
            raise JSONDecodeError(f'failed to get spot for {symbol}:', e.doc, e.pos)
        except:
            raise Exception(f'failed to get spot for {symbol}.')
    elif regex_ticker.search(instr):  # its an isin
        symbol = get_with_isin_map(instr)
        if symbol is None:
            print(f'Could not associate symbol to {instr}')
            return None
        if regex_ticker.search(symbol):  # would lead to an infinite loop
            print(f'symbol associated to {instr} is still an isin.')
            return None
        return get_spot_price(symbol, currency)
    else:
        try:
            spot_native = __get_last_close(instr, '1y')
            native_ccy = __get_currency(instr)
            if native_ccy is None:
                print(f'Could not identify native_ccy for {instr}')
                return None
            fx = get_spot_price(native_ccy, currency)

            return spot_native * fx
        except:
            print(f'Could not find spot for {instr}')
            return None


def get_spot_prices(instr_list, currency):
    fx_spots = {}
    for instr in instr_list:
        fx_spots[instr] = get_spot_price(instr, currency)

    return fx_spots


def get_symbol_from_isin(isin):
    return __isin_cache.get(isin, __get_symbol_from_isin)


def __get_symbol_from_isin(isin):
    params = {'q': isin, 'quotesCount': 1, 'newsCount': 0}
    r = requests.get(__query_yahoo_url, params=params)
    try:
        data = r.json()
    except:
        # raise Exception(f'failed to parse answer for {isin}: {r}')
        print(f'failed to parse answer for {isin}: {r}')
        return None

    try:
        return data['quotes'][0]['symbol']
    except IndexError:
        return None


def __get_time_series_in_currency(symbol, currency, period):
    close_native_ccy = __get_history(symbol, period).Close
    native_ccy = __get_currency(symbol)
    if native_ccy is None:
        return None

    if native_ccy == currency:
        return close_native_ccy

    if len(close_native_ccy) == 0:
        return None

    fx_ts = __get_history(f"{native_ccy}{currency}=X", period).Close

    frame = {'eq': close_native_ccy, 'fx': fx_ts}
    result = pd.DataFrame(frame)
    result = result.dropna(1, 'all')
    result = result.dropna(0, 'any')
    result['eq_fx'] = result['eq'] * result['fx']

    return result['eq_fx']


__isin_map = {'FR0014000RC4': 'FR0000120321'}


def get_with_isin_map(isin):
    if isin == 'CASH': return None
    return get_symbol_from_isin(__isin_map[isin]) if isin in __isin_map else get_symbol_from_isin(isin)


def __get_ticker(symbol):
    return yf.Ticker(symbol)


def __get_last_close(symbol, period):
    hist = __get_history(symbol, period)
    if hist is None:
        return None
    if len(hist.Close) == 0:
        return None
    return hist.Close[-1]


@cached(mem_cache_history)
def __get_history(symbol, period):
    return __get_ticker(symbol).history(period=period)


__currency_map = {
    'FHQP.F': 'EUR',
    'PBF7.F': 'EUR'
}


@cached(mem_cache_currency)
def __get_currency(symbol):
    if symbol in __currency_map:
        return __currency_map[symbol].upper()

    ticker = __get_ticker(symbol)
    if ticker is None:
        return None
    try:
        info = ticker.info
    except IndexError:
        return None
    except ValueError:
        return None
    except KeyError:
        return None

    if info is None:
        return None
    if 'currency' in info:
        return info['currency'].upper()
    else:
        return None