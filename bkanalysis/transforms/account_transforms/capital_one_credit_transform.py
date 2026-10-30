# coding=utf8
import pandas as pd
from bkanalysis.config.config_helper import parse_list
from bkanalysis.transforms.account_transforms import static_data as sd


def can_handle(path_in, config, *args):
    if not path_in.lower().endswith("csv"):
        return False
    df = pd.read_csv(path_in, nrows=1)
    expected_columns = parse_list(config["expected_columns"])
    return set(df.columns) == set(expected_columns)


def load(path_in, config, *args):
    df = pd.read_csv(path_in)
    expected_columns = parse_list(config["expected_columns"])
    assert set(df.columns) == set(expected_columns), (
        f'Was expecting [{", ".join(expected_columns)}] but file columns ' f'are [{", ".join(df.columns)}]. (Capital One Credit)'
    )

    df_out = pd.DataFrame(columns=sd.target_columns)

    df_out.Date = pd.to_datetime(df["Transaction Date"], format="%Y-%m-%d")
    df_out.Account = config["account_name"]
    df_out.Currency = config["currency"]
    df_out.Memo = df["Description"]
    df_out.Subcategory = df["Category"]
    df_out["AccountType"] = config["account_type"]
    # purchases are listed as positive Debits, payments and refunds as positive Credits
    df_out.Amount = df["Credit"].fillna(0.0) - df["Debit"].fillna(0.0)

    return df_out
