# coding=utf8
import pandas as pd
from bkanalysis.transforms.account_transforms import static_data as sd
from bkanalysis.transforms.account_transforms.transformation_helper import can_handle_csv, read_csv_with_expected_columns


def can_handle(path_in, config, *args):
    return can_handle_csv(path_in, config)


def load(path_in, config, *args):
    df = read_csv_with_expected_columns(path_in, config, "Wells Fargo Credit")

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
