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
        f'Was expecting [{", ".join(expected_columns)}] but file columns ' f'are [{", ".join(df.columns)}]. (Wells Fargo Credit)'
    )

    df_out = pd.DataFrame(columns=sd.target_columns)

    df_out.Date = pd.to_datetime(df["DATE"], format="%m/%d/%Y")
    df_out.Account = config["account_name"]
    df_out.Currency = config["currency"]
    df_out.Memo = df["DESCRIPTION"]
    # the extract has no category column
    df_out.Subcategory = "Empty Category"
    df_out["AccountType"] = config["account_type"]
    # amounts are already signed: purchases negative, payments and refunds positive
    df_out.Amount = pd.to_numeric(df["AMOUNT"])

    return df_out
