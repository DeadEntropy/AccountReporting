import configparser
from re import sub

import numpy as np
import pandas as pd
import glob
import os
from bkanalysis.config.config_helper import parse_list
from bkanalysis.transforms.account_transforms import static_data as sd
from bkanalysis.config import config_helper as ch

MONEY_IN = "Money in (GBP)"
MONEY_OUT = "Money out (GBP)"


def can_handle(path_in, config, *args):
    if not path_in.endswith("csv"):
        return False
    df = pd.read_csv(path_in, nrows=1)
    expected_columns = parse_list(config["expected_columns"])
    return set(df.columns) == set(expected_columns)


def as_string(v):
    if v == 0:
        return "0"
    return v


def get_product_name(s):
    names = []
    for e in list(s.unique()):
        if e is not np.nan:
            names.append(e)

    if len(names) != 1:
        raise Exception(f"Found more than one Produce Names: {names}")

    return names[0]


def get_year(s):
    # if int(datetime.datetime.now().year) != 2020:
    #     raise Exception('This function only works for 2020!')

    return s


def _get_date_from_description(description):
    """extracts the date embedded in the description, or returns None when there is none"""
    if not isinstance(description, str) or "\n" not in description:
        return None
    parts = description.split("\n")[1].split(" ")
    if len(parts) != 3:
        return None
    try:
        return pd.to_datetime(parts[2], format="%Y/%m/%d")
    except (ValueError, TypeError):
        return None


def get_dates_from_description(df, fallback_year):
    results = []
    year = None
    for index, row in df.iterrows():
        current_date = _get_date_from_description(row["Description"])
        if current_date is not None:
            results.append(current_date)
            year = current_date.year
        else:
            date = row["Completed Date"]
            results.append(pd.to_datetime(f"{date}, {fallback_year if (year is None) else year}", format="%b %d, %Y"))

    return results


def load(path_in, config, *args):
    df = pd.read_csv(path_in, parse_dates=["Completed Date"])
    expected_columns = parse_list(config["expected_columns"])
    assert set(df.columns) == set(expected_columns), (
        f'Was expecting [{", ".join(expected_columns)}] but file columns ' f'are [{", ".join(df.columns)}]. (Vault)'
    )

    df[MONEY_IN] = df[MONEY_IN].fillna(0)
    df[MONEY_OUT] = df[MONEY_OUT].fillna(0)
    df["Interest rate (AER)"] = df["Interest rate (AER)"].fillna("")

    df[MONEY_IN] = [float(sub(r"[^\d\-.]", "", as_string(x))) for x in df[MONEY_IN]]
    df[MONEY_OUT] = [float(sub(r"[^\d\-.]", "", as_string(x))) for x in df[MONEY_OUT]]

    df_out = pd.DataFrame(columns=sd.target_columns)

    df_out.Date = df["Completed Date"]
    df_out.Account = get_product_name(df["Product name"])
    df_out.Currency = config["currency"]
    df_out.Amount = df[MONEY_IN] + df[MONEY_OUT]
    df_out.Subcategory = df["Description"].str.split("\n").str[0].str.strip()
    df_out.Memo = (df["Description"].str.split("\n").str[0].str.strip() + " " + df["Interest rate (AER)"].astype(str)).str.strip()
    df_out["AccountType"] = config["account_type"]

    return df_out
