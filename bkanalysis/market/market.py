import datetime as dt
import pandas as pd
from bkanalysis.market.price import Price
from typing import Union


class Market:

    _EXTRAPOLATE_LEFT = True

    def __init__(self, dict_of_values):
        self._dict = dict_of_values
        self._dict_sorted_dates = {}

    def supported_instr(self):
        return list(self._dict.keys())

    def __get_previous_date(self, instr: str, date: Union[dt.datetime, pd.Timestamp]):
        if instr not in self._dict.keys():
            raise Exception(f'{instr} is not in the Market.')

        if len(self._dict[instr].keys()) == 0:
            raise Exception(f'No Data available for {instr} in the Market.')

        if instr not in self._dict_sorted_dates:
            sorted_list = sorted(list(self._dict[instr].keys()), reverse=True)
            self._dict_sorted_dates[instr] = sorted_list
        else:
            sorted_list = self._dict_sorted_dates[instr]

        min_date = sorted_list[-1]

        date = pd.Timestamp(date)
        min_date = pd.Timestamp(min_date)

        if date.tz_localize(None) < min_date.tz_localize(None):
            if self._EXTRAPOLATE_LEFT:
                return min_date
            raise Exception(f'{date} is before the first available date for {instr} in the Market.')

        return next(d for d in sorted_list if pd.Timestamp(d).tz_localize(None) < date.tz_localize(None))

    def get_price(self, instr: str, date: dt.datetime):
        if instr not in self._dict.keys():
            raise Exception(f'{instr} is not in the Market.')

        if date not in self._dict[instr].keys():
            return self._dict[instr][self.__get_previous_date(instr, date)]

        return self._dict[instr][date]

    def get_price_in_currency(self, instr: str, date: dt.datetime, currency: str):
        if instr not in self._dict.keys():
            raise Exception(f'{instr} is not in the Market.')

        if date not in self._dict[instr].keys():
            date = self.__get_previous_date(instr, date)

        if self._dict[instr][date].currency == currency:
            return self._dict[instr][date].value

        fx = self.get_price(f'{self._dict[instr][date].currency}{currency}=X', date)
        return self._dict[instr][date].value * fx.value
