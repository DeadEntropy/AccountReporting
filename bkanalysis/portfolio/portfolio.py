import requests
import yfinance as yf
import pandas as pd
import numpy as np
from bkanalysis.portfolio import cache
from cachetools import cached, LRUCache

__query_yahoo_url = "https://query2.finance.yahoo.com/v1/finance/search"
__isin_to_symbol_mapping_path = r'isin_cache.json'
__isin_cache = cache.CacheDict(__isin_to_symbol_mapping_path)


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


def __get_time_series_in_currency(ticker, currency, period):
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


def get_benchmarks(start_point, indices=['^FCHI', '^GSPC', '^FTSE', 'URTH'], period='1y', currency='GBP'):
    tss = [__get_time_series_in_currency(yf.Ticker(index), currency, period) for index in indices]
    return [ts * start_point / ts[0] for ts in tss]


__isin_map = {'FR0014000RC4': 'FR0000120321'}


def get_with_isin_map(isin):
    if isin == 'CASH': return None
    return get_symbol_from_isin(__isin_map[isin]) if isin in __isin_map else get_symbol_from_isin(isin)


mem_cache_symbol = LRUCache(maxsize=1024)
@cached(mem_cache_symbol)
def __get_ticker(symbol):
    return yf.Ticker(symbol)


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


def process_stock(stocks, key='isin', period='1y', fallback_key=None, currency='GBP'):
    if fallback_key is None:
        stocks['symbol'] = [get_with_isin_map(v) for v in stocks[key]]
    else:
        stocks['symbol'] = [get_with_isin_map(v) if get_with_isin_map(v) is not None else get_with_isin_map(v_fallback)
                            for (v, v_fallback) in zip(stocks[key], stocks[fallback_key])]

    stocks['yf_ticker'] = [__get_ticker(symbol) if symbol is not None else None for symbol in stocks['symbol']]
    stocks['close'] = [__get_close(ticker, currency) if ticker is not None else None for ticker in stocks['yf_ticker']]
    stocks['native_currency'] = [__get_currency(ticker) if ticker is not None else None for ticker in stocks['yf_ticker']]
    stocks['currency'] = currency
    stocks[f'times_series_{period}_{currency}'] = [__get_time_series_in_currency(ticker, currency, period) if ticker is not None else None for ticker
                                        in stocks.yf_ticker]


def total_return(ts):
    return (ts[len(ts) - 1] - ts[0]) / ts[0]


def clean_time_series(stocks, period, currency='GBP'):
    frame = {}
    for (index, ts) in zip(stocks.index, stocks[f'times_series_{period}_{currency}']):
        frame[str(index)] = ts

    result = pd.DataFrame(frame)
    result = result.dropna(1, 'all')

    for col in result.columns:
        if result[col].isna().sum() > len(result[col]) * 0.5:
            result = result.drop(col, axis=1)

    for col in result.columns:
        previous_value = result[col].iloc[0]
        for k in reversed(result[col].keys()):
            if np.isnan(result[col][k]) or result[col][k] == 0 or result[col][k] < 0.2 * previous_value:
                result[col][k] = previous_value
            previous_value = result[col][k]

        for k in result[col].keys():
            if np.isnan(result[col][k]) or result[col][k] == 0 or result[col][k] < 0.2 * previous_value:
                result[col][k] = previous_value
            previous_value = result[col][k]

    for idx in stocks.index:
        if str(idx) in result.columns:
            stocks.at[idx, f'times_series_{period}_{currency}'] = result[str(idx)]
        else:
            stocks.at[idx, f'times_series_{period}_{currency}'] = None


def get_portfolio_ts(stocks, start_point, period, currency='GBP'):
    _sum_ts = None
    for idx in stocks.index:
        if np.isnan(stocks['count'].iloc[idx]):
            continue
        if stocks[f'times_series_{period}_{currency}'].iloc[idx] is None:
            continue
        ts = stocks[f'times_series_{period}_{currency}'].iloc[idx] * stocks['count'].iloc[idx]
        if _sum_ts is None:
            _sum_ts = ts
        else:
            _sum_ts = _sum_ts + ts

    if start_point is None:
        return _sum_ts

    return _sum_ts * start_point / _sum_ts[0]
