from datetime import datetime

import pandas as pd


def normalize_date_column(date_column):
    """Normalizes the date column to a consistent date format."""

    def convert_date(value):
        if isinstance(value, pd.Timestamp) or isinstance(value, datetime):
            return value.date()
        try:
            return pd.to_datetime(value).date()
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid date format: {value}") from exc

    return pd.to_datetime(date_column.map(convert_date))


def is_ccy(asset):
    """check is an asset in a currency in the yahoo nomenclature i.e. 'GBPUSD=X'"""
    if len(asset) != 8:
        return False
    if asset.upper().endswith("=X"):
        return True
    return False
