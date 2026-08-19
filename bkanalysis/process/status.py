# coding=utf8
import pandas as pd
import configparser
from bkanalysis.config import config_helper as ch


class LastUpdate:

    def __init__(self, config=None):
        if config is None:
            self.config = configparser.ConfigParser()
        if len(self.config.read(ch.source)) != 1:
            raise OSError(f"no config found in {ch.source}")

    def last_update(self, df_input):
        dic_last_update = {}
        df_input["Date"] = pd.to_datetime(df_input["Date"])
        for bank_acc in df_input.Account.unique():
            dic_last_update[bank_acc] = [df_input[df_input.Account == bank_acc].Date.max().strftime("%Y-%m-%d")]

        return pd.DataFrame.from_dict(dic_last_update, orient="index", columns=["LastUpdate"])

    def last_update_save(self):
        df_input = pd.read_csv(self.config["IO"]["path_aggregated"])
        self.last_update(df_input).to_csv(self.config["IO"]["path_last_updated"])
