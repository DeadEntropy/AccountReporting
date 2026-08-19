import datetime as dt
import pandas as pd
from bkanalysis.market.price import Price
from typing import Union


class Market:

    _EXTRAPOLATE_LEFT = True

    def __init__(self, dict_of_values, linear_interpolation: bool = False):
        self._dict = {
            asset: {
                pd.Timestamp(k).tz_convert(None) if pd.Timestamp(k).tzinfo is not None else pd.Timestamp(k): v
                for k, v in dict_of_values[asset].items()
                if v.value > 0.0
            }
            for asset in dict_of_values
        }
        self._dict_sorted_dates = {}
        self.linear_interpolation = linear_interpolation

    def supported_instr(self):
        return list(self._dict.keys())

    def __get_last_date(self, instr: str):
        if instr not in self._dict.keys():
            raise Exception(f"{instr} is not in the Market.")

        if len(self._dict[instr].keys()) == 0:
            raise Exception(f"No Data available for {instr} in the Market.")

        if instr not in self._dict_sorted_dates:
            sorted_list = sorted(list(self._dict[instr].keys()), reverse=True)
            self._dict_sorted_dates[instr] = sorted_list
        else:
            sorted_list = self._dict_sorted_dates[instr]

        return sorted_list[0]

    def __get_closest_date(self, instr: str, date: Union[dt.datetime, pd.Timestamp], previous: bool = True):
        if instr not in self._dict.keys():
            raise Exception(f"{instr} is not in the Market.")

        if len(self._dict[instr].keys()) == 0:
            raise Exception(f"No Data available for {instr} in the Market.")

        if instr not in self._dict_sorted_dates:
            sorted_list = sorted(list(self._dict[instr].keys()), reverse=True)
            self._dict_sorted_dates[instr] = sorted_list
        else:
            sorted_list = self._dict_sorted_dates[instr]

        min_date = sorted_list[-1]

        date = pd.Timestamp(date).tz_localize(None)

        if date < min_date:
            if self._EXTRAPOLATE_LEFT:
                return min_date
            raise Exception(f"{date} is before the first available date for {instr} in the Market.")

        if previous:
            return next(d for d in sorted_list if d < date)
        else:
            return next(d for d in sorted_list if d >= date)

    def get_price(self, instr: str, date: dt.datetime):
        if instr not in self._dict.keys():
            raise Exception(f"{instr} is not in the Market.")

        if date not in self._dict[instr].keys():
            return self._dict[instr][self.__get_closest_date(instr, date)]

        return self._dict[instr][date]

    def get_price_in_currency(self, instr: str, date: dt.datetime, currency: str):
        if instr not in self._dict.keys():
            raise Exception(f"{instr} is not in the Market.")

        if date not in self._dict[instr].keys():
            date_prev = self.__get_closest_date(instr, date, previous=True)
            if (not self.linear_interpolation) | (date.date() > self.__get_last_date(instr).date()):
                v = self._dict[instr][date_prev].value
            else:  # linear interpolation
                date_next = self.__get_closest_date(instr, date, previous=False)
                v1 = self._dict[instr][date_prev].value
                v2 = self._dict[instr][date_next].value

                w1 = (date_next.date() - date.date()).days
                w2 = (date.date() - date_prev.date()).days

                if w1 + w2 == 0:
                    raise Exception()
                v = (w2 * v2 + w1 * v1) / (w1 + w2)
            c = self._dict[instr][date_prev].currency
        else:
            v = self._dict[instr][date].value
            c = self._dict[instr][date].currency

        if c == currency:
            return v

        fx = self.get_price(f"{c}{currency}=X", date)
        return v * fx.value
