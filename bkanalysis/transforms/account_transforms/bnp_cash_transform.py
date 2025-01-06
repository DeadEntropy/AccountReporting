import configparser

import pandas as pd
import glob
import os
from bkanalysis.config.config_helper import parse_list
from bkanalysis.transforms.account_transforms import static_data as sd
from bkanalysis.config import config_helper as ch


def can_handle(path_in, config, *args):
    if not path_in.endswith("csv"):
        return False

    try:
        df = pd.read_csv(path_in)
    except Exception:
        return False

    expected_columns = parse_list(config["expected_columns"], False)

    return set(df.columns) == set(expected_columns)


def load(path_in, config, *args):
    df = pd.read_csv(path_in)
    expected_columns = parse_list(config["expected_columns"], False)

    assert set(df.columns) == set(
        expected_columns
    ), f'Was expecting [{", ".join(expected_columns)}] but file columns are [{", ".join(df.columns)}]. (BNP Cash)'

    df_out = pd.DataFrame(columns=sd.target_columns)
    df_out.Date = pd.to_datetime(df["Date operation"], dayfirst=True)
    df_out.Account = config["account_name"]
    df_out.Currency = config["currency"]
    df_out.Amount = df["Montant operation"]
    df_out.Subcategory = df["Sous Categorie operation"]
    df_out.Memo = df["Libelle operation"]
    df_out["AccountType"] = config["account_type"]

    return df_out


def load_save(config):
    files = glob.glob(os.path.join(config["folder_in"], "*.csv"))
    print(f"found {len(files)} CSV files in {config['folder_in']}.")
    if len(files) == 0:
        return

    df_list = [load(f, config) for f in files]
    for df_temp in df_list:
        df_temp["count"] = df_temp.groupby(sd.target_columns).cumcount()
    df = pd.concat(df_list)
    df.drop_duplicates().drop(["count"], axis=1).sort_values("Date", ascending=False).to_csv(config["path_out"], index=False)


def load_save_default():
    config = configparser.ConfigParser()
    config.read(ch.source)

    load_save(config["Coinbase"])
