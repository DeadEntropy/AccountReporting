import yfinance as yf
import pandas as pd
from cachetools import cached, LRUCache
from bkanalysis.portfolio import cache
import requests


__query_yahoo_url = "https://query2.finance.yahoo.com/v1/finance/search"
__isin_to_symbol_mapping_path = r'isin_cache.json'
__isin_cache = cache.CacheDict(__isin_to_symbol_mapping_path)


def get_spot_prices(instr_list, currency):
    fx_spots = {}
    for ccy in instr_list:
        if ccy == currency:
            fx_spots[ccy] = 1.0
        elif len(ccy) == 3:  # its a currency
            if ccy in ['BTC', 'ETH']:
                ticker = f'{ccy}-{currency}'
            else:
                ticker = f'{ccy}{currency}=X'
            try:
                fx_spots[ccy] = yf.Ticker(ticker).history(period='1d').iloc[0].Close
            except:
                raise Exception(f'failed to get fx spot for {ticker}.')
        else:  # its a ticker
            fx_spots[ccy] = get_close_from_isin(ccy, currency)

    return fx_spots


def get_symbol_from_isin(isin):
    return __isin_cache.get(isin, __get_symbol_from_isin)


def __get_symbol_from_isin(isin):
    params = {'q': isin, 'quotesCount': 1, 'newsCount': 0}
    r = requests.get(__query_yahoo_url, params=params)
    try:
        data = r.json()
    except:
        raise Exception(f'failed to parse answer for {isin}: {r}')

    try:
        return data['quotes'][0]['symbol']
    except IndexError:
        return None


mem_cache = LRUCache(maxsize=1024)
@cached(mem_cache)
def __get_history(ticker, period):
    return ticker.history(period=period)


def __get_time_series_in_currency(index, currency, period):
    ticker = yf.Ticker(index)
    close_native_ccy = __get_history(ticker, period).Close
    native_ccy = __get_currency(ticker)
    if native_ccy is None:
        return None

    if native_ccy == currency:
        return close_native_ccy

    fx_ts = __get_history(yf.Ticker(f"{native_ccy}{currency}=X"), period).Close

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


mem_cache_symbol = LRUCache(maxsize=1024)
@cached(mem_cache_symbol)
def __get_ticker(symbol):
    return yf.Ticker(symbol)


mem_cache_close = LRUCache(maxsize=1024)
@cached(mem_cache_close)
def __get_close(ticker, currency='GBP'):
    hist = __get_history(ticker, '1d')
    if len(hist) == 0:
        return None
    native_ccy = __get_currency(ticker)
    if native_ccy is None:
        return None

    fx = __get_history(yf.Ticker(f"{native_ccy}{currency}=X"), '1d').iloc[0].Close

    return hist.iloc[0].Close * fx


mem_cache_currency = LRUCache(maxsize=1024)
@cached(mem_cache_currency)
def __get_currency(ticker):
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
        return info['currency']
    else:
        return None


def get_close_from_isin(isin, currency):
    symbol = get_with_isin_map(isin)
    if symbol is None:
        return 0.0
    ticker = __get_ticker(symbol)
    if ticker is None:
        return 0.0
    close = __get_close(ticker, currency)
    if close is None:
        return 0.0
    return close