import requests
import yfinance as yf
import pandas as pd
import numpy as np

__query_yahoo_url = "https://query2.finance.yahoo.com/v1/finance/search"


def get_symbol_from_isin(isin):
    params = {'q': isin, 'quotesCount': 1, 'newsCount': 0}
    r = requests.get(__query_yahoo_url, params=params)
    data = r.json()
    try:
        return data['quotes'][0]['symbol']
    except IndexError:
        return None


def get_benchmarks(start_point, indices=['^FCHI', '^GSPC', 'URTH'], period='1y'):
    tss = [yf.Ticker(index).history(period=period).Close for index in indices]
    return [ts * start_point / ts[0] for ts in tss]


__isin_map = {'FR0014000RC4': 'FR0000120321'}


def get_with_isin_map(isin):
    if isin == 'CASH': return None
    return get_symbol_from_isin(__isin_map[isin]) if isin in __isin_map else get_symbol_from_isin(isin)


def __get_close(ticker):
    hist = ticker.history(period='1d')
    if len(hist) == 0:
        return None
    return hist.iloc[0].Close


def process_stock(stocks, key='isin', period='1y', fallback_key=None):
    if fallback_key is None:
        stocks['symbol'] = [get_with_isin_map(v) for v in stocks[key]]
    else:
        stocks['symbol'] = [get_with_isin_map(v) if get_with_isin_map(v) is not None else get_with_isin_map(v_fallback)
                            for (v, v_fallback) in zip(stocks[key], stocks[fallback_key])]

    stocks['yf_ticker'] = [yf.Ticker(symbol) if symbol is not None else None for symbol in stocks['symbol']]
    stocks['close'] = [__get_close(ticker) if ticker is not None else None for ticker in stocks['yf_ticker']]
    stocks[f'times_series_{period}'] = [ticker.history(period=period).Close if ticker is not None else None for ticker
                                        in stocks.yf_ticker]


def annual_return(ts):
    return (ts[len(ts) - 1] - ts[0]) / ts[0]


def clean_time_series(stocks, period):
    frame = {}
    for (index, ts) in zip(stocks.index, stocks[f'times_series_{period}']):
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
            stocks.at[idx, f'times_series_{period}'] = result[str(idx)]
        else:
            stocks.at[idx, f'times_series_{period}'] = None


def get_portfolio_ts(stocks, start_point, period):
    _sum_ts = None
    for idx in stocks.index:
        if np.isnan(stocks['count'].iloc[idx]):
            continue
        if stocks[f'times_series_{period}'].iloc[idx] is None:
            continue
        ts = stocks[f'times_series_{period}'].iloc[idx] * stocks['count'].iloc[idx]
        if _sum_ts is None:
            _sum_ts = ts
        else:
            _sum_ts = _sum_ts + ts

    return _sum_ts * start_point / _sum_ts[0]