import datetime as dt


class Market:
    class Price:
        def __init__(self, value: float, currency: str):
            self.value = value
            self.currency = currency

    _EXTRAPOLATE_LEFT = True

    def __init__(self, values):
        self._dict = {value[0]: {date: self.Price(close, value[2]) for (date, close) in value[1].Close.iteritems()} for value in list(values)}

    def __get_previous_date(self, instr: str, date: dt.datetime):
        if instr not in self._dict.keys():
            raise Exception(f'{instr} is not in the Market.')

        if len(self._dict[instr].keys()) == 0:
            raise Exception(f'No Data available for {instr} in the Market.')

        if date < min(list(self._dict[instr].keys())):
            if self._EXTRAPOLATE_LEFT:
                return min(list(self._dict[instr].keys()))
            raise Exception(f'{date} is before the first available date for {instr} in the Market.')

        return next(d for d in sorted(list(self._dict[instr].keys()), reverse=True) if d < date)

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
