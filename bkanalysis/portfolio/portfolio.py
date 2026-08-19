from bkanalysis.market import market_prices as mp
import pandas as pd
import numpy as np


def get_benchmarks(start_point, indices=["^FCHI", "^GSPC", "^FTSE", "URTH"], period="1y", currency="GBP"):
    tss = [mp.__get_time_series_in_currency(index, currency, period) for index in indices]
    return [ts * start_point / ts[0] for ts in tss]


def process_stock(stocks, key="isin", period="1y", fallback_key=None, currency="GBP"):
    if fallback_key is None:
        stocks["symbol"] = [mp.get_with_isin_map(v) for v in stocks[key]]
    else:
        stocks["symbol"] = [
            mp.get_with_isin_map(v) if mp.get_with_isin_map(v) is not None else mp.get_with_isin_map(v_fallback)
            for (v, v_fallback) in zip(stocks[key], stocks[fallback_key])
        ]

    stocks["yf_ticker"] = [mp.__get_ticker(symbol) if symbol is not None else None for symbol in stocks["symbol"]]
    stocks["close_native"] = [mp.__get_last_close(symbol, period) if symbol is not None else None for symbol in stocks["symbol"]]
    stocks["native_currency"] = [mp.get_currency(symbol).upper() if symbol is not None else None for symbol in stocks["symbol"]]
    stocks["close"] = [mp.get_spot_price(symbol, currency) for symbol in stocks["symbol"]]
    stocks["currency"] = currency
    stocks[f"times_series_{period}_{currency}"] = [
        mp.__get_time_series_in_currency(symbol, currency, period) if symbol is not None else None for symbol in stocks["symbol"]
    ]


def total_return(ts):
    return (ts[len(ts) - 1] - ts[0]) / ts[0]


def clean_time_series(stocks, period, currency="GBP"):
    frame = {}
    for index, ts in zip(stocks.index, stocks[f"times_series_{period}_{currency}"]):
        frame[str(index)] = ts

    result = pd.DataFrame(frame)
    result = result.dropna(1, "all")

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
            stocks.at[idx, f"times_series_{period}_{currency}"] = result[str(idx)]
        else:
            stocks.at[idx, f"times_series_{period}_{currency}"] = None


def get_portfolio_ts(stocks, start_point, period, currency="GBP"):
    _sum_ts = None
    for idx in stocks.index:
        if np.isnan(stocks["count"].iloc[idx]):
            continue
        if stocks[f"times_series_{period}_{currency}"].iloc[idx] is None:
            continue
        ts = stocks[f"times_series_{period}_{currency}"].iloc[idx] * stocks["count"].iloc[idx]
        if _sum_ts is None:
            _sum_ts = ts
        else:
            _sum_ts = _sum_ts + ts

    if start_point is None:
        return _sum_ts

    return _sum_ts * start_point / _sum_ts[0]
