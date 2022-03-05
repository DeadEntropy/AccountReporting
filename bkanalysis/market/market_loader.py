from enum import Enum
import configparser
import ast
import pandas as pd
import logging
from datetime import datetime
from bkanalysis.config import config_helper as ch
from bkanalysis.market import market_prices as mp
from bkanalysis.market.price import Price


class SOURCE(Enum):
    YAHOO = 1
    FILE = 2
    HARDCODED = 3


class MarketLoader:

    def __init__(self, config=None):
        if config is None:
            self.config = configparser.ConfigParser()
            if len(self.config.read(ch.source)) != 1:
                raise OSError(f'no config found in {ch.source}')
        else:
            self.config = config

        if 'Market' not in self.config:
            raise Exception('Expected a Market entry in the config.')
        self.source_map = {}
        for (k,v) in ast.literal_eval(self.config['Market']['source_map']).items():
            self.source_map[k] = [SOURCE[v[0].upper()]] + list(v[1:])
        self.source_default = SOURCE.YAHOO

    def load(self, instruments, ref_currency: str, period: str):
        # Get the symbols from the list of instruments
        logging.info(f'Getting Symbols ({len(instruments)})')
        symbols = [self.get_symbol(instr, ref_currency) if instr not in self.source_map.keys() else instr for instr in instruments]

        # Get the additional currencies
        logging.info(f'Getting Additional Currencies ({len(symbols)})')
        currencies_from_yahoo = list(set([self.get_currency(symbol) for symbol in symbols if symbol not in self.source_map.keys()]))
        symbols = list(set([f'{ccy}{ref_currency}=X' for ccy in currencies_from_yahoo if ccy != ref_currency and ccy is not None] + symbols))

        # Get the time series
        logging.info('Getting Times Series of Mkt Data')
        values = {symbol: self.get_history(symbol, period) for symbol in symbols if symbol is not None}

        logging.info('Done Loading all Mkt Data')
        return values

    # return a Dictionary of date:Price
    def get_history(self, symbol: str, period: str):
        if symbol in self.source_map:
            if self.source_map[symbol][0] == SOURCE.YAHOO:
                logging.info(f'Getting {period} of data for {symbol} from {SOURCE.YAHOO}')
                return self.get_history_from_yahoo(symbol, period)
            elif self.source_map[symbol][0] == SOURCE.FILE:
                logging.info(f'Getting {period} of data for {symbol} from {SOURCE.FILE}')
                if len(self.source_map[symbol]) != 2:
                    raise Exception('source is FILE but not path was passed in.')
                return self.get_history_from_file(self.source_map[symbol][1])
            elif self.source_map[symbol][0] == SOURCE.HARDCODED:
                logging.info(f'Getting {period} of data for {symbol} from {SOURCE.HARDCODED}')
                if len(self.source_map[symbol]) != 3:
                    raise Exception('source is HARDCODED but not VALUE was passed in.')
                return self.get_history_from_hardcoded(self.source_map[symbol][1], self.source_map[symbol][2])
            else:
                raise Exception(f'{self.source_map[symbol][0]} source is not supported.')

        if self.source_default == SOURCE.YAHOO:
            logging.info(f'Getting {period} of data for {symbol} from {SOURCE.YAHOO}')
            return self.get_history_from_yahoo(symbol, period)

        raise Exception(f'default source can not be {self.source_default}.')

    _FILE_DATE_FORMAT = '%d/%m/%Y'

    @staticmethod
    def get_history_from_hardcoded(values: dict, currency: str):
        logging.debug(f'Getting History from HARDCODED.')
        return {datetime.strptime(date, MarketLoader._FILE_DATE_FORMAT): Price(close, currency)\
                for (date, close) in values.items()}

    @staticmethod
    def get_history_from_file(path: str):
        logging.debug(f'Getting History from FILE from {path}')
        df = pd.read_csv(path)
        currency = df['Currency code'][0]
        df['Unit Price Date'] = pd.to_datetime(df['Unit Price Date'], format=MarketLoader._FILE_DATE_FORMAT)
        return {date: Price(close, currency) for (date, close) in df.set_index('Unit Price Date')['Unit Price'].items()}

    @staticmethod
    def get_history_from_yahoo(symbol: str, period: str):
        logging.debug(f'Getting History from YAHOO for {symbol} {period}')
        currency = MarketLoader.get_currency(symbol)
        return {date: Price(close, currency) for (date, close) in mp.get_history(symbol, period).Close.iteritems()}

    @staticmethod
    def get_symbol(instr: str, ref_currency: str):
        return mp.get_symbol(instr, ref_currency)

    @staticmethod
    def get_currency(symbol: str):
        return mp.get_currency(symbol)
