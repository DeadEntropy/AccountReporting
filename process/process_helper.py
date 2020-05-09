import datetime


def get_adjusted_month(dt):
    if dt.day > 20:
        if dt.month < 12:
            return dt.month + 1
        return 1
    return dt.month


def get_adjusted_year(dt):
    if dt.day > 20 and dt.month == 12:
        return dt.year + 1
    return dt.year


def get_fiscal_year(dt):
    if dt < datetime.datetime(dt.year, 4, 5):
        return dt.year - 1
    return dt.year


def get_year_to_date(dt):
    return int((datetime.date.today() - dt.date()).days / 365.25)