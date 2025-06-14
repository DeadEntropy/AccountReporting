from json import JSONDecodeError

import yfinance as yf

# from yahooquery import Ticker
import pandas as pd
from cachetools import cached, LRUCache
import requests
import re

mem_cache_history = LRUCache(maxsize=1024)
mem_cache_currency = LRUCache(maxsize=1024)
mem_cache_symbol = LRUCache(maxsize=1024)

__query_yahoo_url_search = "https://query2.finance.yahoo.com/v1/finance/search"

regex_ticker = re.compile(r"^[a-zA-Z][a-zA-Z][0-9]+")

__currencies = ["EUR", "USD", "GBP", "CHF", "JPY", "CAD", "AUD", "KRW", "CNH"]
__crypto = ["BTC", "ETH", "ETH2"]
__crypto_staking_map = {"ETH2": "ETH"}


@cached(mem_cache_symbol)
def get_symbol(instr, currency):
    if instr is None:
        return None
    elif (len(instr) == 3 or len(instr) == 4) and (instr in __currencies or instr in __crypto):  # its a currency/crypto
        if instr in __crypto:
            if instr in __crypto_staking_map:
                return f"{__crypto_staking_map[instr]}-{currency}"
            return f"{instr}-{currency}"
        else:
            return f"{instr}{currency}=X"
    elif regex_ticker.search(instr) and len(instr) == 12:  # its an isin
        return get_with_isin_map(instr)

    return instr


def get_spot_price(instr, currency):
    if instr is None:
        return None
    if instr == currency:
        return 1.0
    elif (len(instr) == 3) and (instr in __currencies or instr in __crypto):  # its a currency/crypto
        if instr in __crypto:
            symbol = f"{instr}-{currency}"
        else:
            symbol = f"{instr}{currency}=X"
        try:
            return get_history(symbol, "1y").sort_values("Date").iloc[-1].Close
        except JSONDecodeError as e:
            raise JSONDecodeError(f"failed to get spot for {symbol}:", e.doc, e.pos)
        except:
            raise Exception(f"failed to get spot for {symbol}.")
    elif regex_ticker.search(instr) and len(instr) == 12:  # its an isin
        symbol = get_with_isin_map(instr)
        if symbol is None:
            print(f"Could not associate symbol to {instr}")
            return None
        if regex_ticker.search(symbol):  # would lead to an infinite loop
            print(f"symbol associated to {instr} is still an isin.")
            return None
        return get_spot_price(symbol, currency)
    else:
        try:
            spot_native = __get_last_close(instr, "1y")
            if spot_native is None:
                print(f"Could not find last close for {spot_native}.")
                return None
            native_ccy = get_currency(instr)
            if native_ccy is None:
                print(f"Could not identify native_ccy for {instr}.")
                return None
            fx = get_spot_price(native_ccy, currency)
            if fx is None:
                print(f"Could not find spot for {native_ccy} in {currency}.")
                return None

            return spot_native * fx
        except Exception as e:
            print(f"Could not find spot for {instr} in {currency}: {e}")
            return None


def get_spot_prices(instr_list, currency):
    fx_spots = {}
    for instr in instr_list:
        fx_spots[instr] = get_spot_price(instr, currency)

    return fx_spots


def get_symbol_from_isin(isin):
    return __get_ticker(isin).info["symbol"]


def __get_time_series_in_currency(symbol, currency, period):
    close_native_ccy = get_history(symbol, period).Close
    native_ccy = get_currency(symbol)
    if native_ccy is None:
        return None

    if native_ccy == currency:
        return close_native_ccy

    if len(close_native_ccy) == 0:
        return None

    fx_ts = get_history(f"{native_ccy}{currency}=X", period).Close

    frame = {"eq": close_native_ccy, "fx": fx_ts}
    result = pd.DataFrame(frame)
    result = result.dropna(1, "all")
    result = result.dropna(0, "any")
    result["eq_fx"] = result["eq"] * result["fx"]

    return result["eq_fx"]


__isin_map = {"FR0014000RC4": "FR0000120321"}


def get_with_isin_map(isin):
    if isin == "CASH":
        return None
    return get_symbol_from_isin(__isin_map[isin]) if isin in __isin_map else get_symbol_from_isin(isin)


def __get_ticker(symbol):
    return yf.Ticker(symbol)


def __get_last_close(symbol, period):
    hist = get_history(symbol, period)
    if hist is None:
        print(f"hist for {symbol} for {period} is None.")
        return None
    if len(hist.Close) == 0:
        print(f"hist for {symbol} for {period} contains no values.")
        return None

    return hist.Close[-1]


@cached(mem_cache_history)
def get_history(symbol, period):
    return __get_ticker(symbol).history(period=period)


__currency_map = {"FHQP.F": "EUR", "PBF7.F": "EUR", "34153P7M4": "USD"}


@cached(mem_cache_currency)
def get_currency(symbol):
    try:
        if symbol in __currency_map:
            return __currency_map[symbol]
        return yf.Ticker(symbol).info["currency"].upper()
    except Exception as e:
        raise Exception(f"Failed to query currency for {symbol}", e)
