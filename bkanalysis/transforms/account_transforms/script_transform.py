# coding=utf8
import configparser

import pandas as pd
import glob
import os
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta

from bkanalysis.transforms.account_transforms import static_data as sd
from bkanalysis.config import config_helper as ch


def can_handle(path_in, config, *args):
    if not path_in.lower().endswith("json"):
        return False
    return True


def load(path_in: str, config, *args):
    if os.path.exists(path_in):
        try:
            with open(path_in, "r") as json_file:
                json_obj = json.load(json_file)
        except TypeError:
            raise TypeError(f"Failed to deserialise jSon file '{path_in}' {json_file}")
    else:
        raise TypeError(f"Failed to local jSon file '{path_in}' {json_file}")

    dfs = []
    for _, data in json_obj.items():
        freq = data["freq"].upper()
        if freq == "MONTHLY":
            start_dt = datetime.strptime(data["start_date"], "%d-%b-%Y")
            end_dt = datetime.strptime(data["end_date"], "%d-%b-%Y")

            date_range = []
            delta = relativedelta(months=1)
            while start_dt < end_dt and start_dt < datetime.now():
                date_range.append(start_dt)
                start_dt += delta
        elif freq == "SINGLE":
            start_dt = datetime.strptime(data["start_date"], "%d-%b-%Y")
            date_range = [start_dt]
        else:
            raise Exception("Only Supports Monthly freq & Single payments.")

        df_temp = pd.DataFrame(columns=sd.target_columns)
        df_temp.Date = date_range
        df_temp.Account = data["account_name"]
        df_temp.Currency = data["currency"]
        df_temp.Amount = data["amount"]
        df_temp.Subcategory = data["subcategory"]
        df_temp.Memo = data["memo"]
        df_temp["AccountType"] = data["account_type"]
        dfs.append(df_temp)

    return pd.concat(dfs).sort_values("Date")


def load_save(config):
    files = glob.glob(os.path.join(config["folder_in"], "*.json"))
    print(f"found {len(files)} JSON files in {config['folder_in']}.")
    if len(files) == 0:
        return

    df_list = [load(f, config) for f in files]
    for df_temp in df_list:
        df_temp["count"] = df_temp.groupby(sd.target_columns).cumcount()
    df = pd.concat(df_list)
    df.drop_duplicates().drop(["count"], axis=1).sort_values("Date", ascending=False).to_csv(config["path_out"], index=False)


def load_save_default():
    config = configparser.ConfigParser()
    if len(config.read(ch.source)) != 1:
        raise OSError(f"no config found in {ch.source}")

    load_save(config["Script"])
