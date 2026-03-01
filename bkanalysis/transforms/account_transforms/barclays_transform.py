# coding=utf8
import configparser
import ast

import pandas as pd
import glob
import os
import re
from bkanalysis.config.config_helper import parse_list
from bkanalysis.transforms.account_transforms import static_data as sd
from bkanalysis.config import config_helper as ch


def can_handle(path_in, config, *args):
    if not path_in.endswith("csv"):
        return False
    df = pd.read_csv(path_in, nrows=1)
    expected_columns = parse_list(config["expected_columns"])
    return set(df.columns) == set(expected_columns)


def load(path_in, config, *args):
    df = pd.read_csv(path_in)
    expected_columns = parse_list(config["expected_columns"])
    assert set(df.columns) == set(expected_columns), (
        f'Was expecting [{", ".join(expected_columns)}] but file columns ' f'are [{", ".join(df.columns)}]. (Barclays)'
    )

    df_out = df.drop("Number", axis=1)

    df_out.Date = pd.to_datetime(df_out.Date, format="%d/%m/%Y")
    account_currencies = ast.literal_eval(config["account_currencies"])
    df_out["Currency"] = [account_currencies[acc] for acc in df_out.Account]
    df_out["Memo"] = [re.sub(" +", " ", memo) for memo in df_out.Memo]
    df_out["AccountType"] = config["account_type"]

    return df_out
