import configparser

import pandas as pd

from bkanalysis.config import config_helper
from bkanalysis.process import process
from bkanalysis.transforms import master_transform

from .manager_helper import normalize_date_column


class DataManager:
    """Main Data Handling Class."""

    COLUMNS = [
        "Date",
        "Account",
        "Amount",
        "Subcategory",
        "Memo",
        "Currency",
        "MemoSimple",
        "MemoMapped",
        "Type",
        "FullType",
        "SubType",
        "FullSubType",
        "MasterType",
        "FullMasterType",
        "AccountType",
        "FacingAccount",
        "SourceFile",
    ]

    def __init__(self, config=None):
        self._load_config(config)
        self.include_xls = True
        self.include_json = True
        self.ignore_overrides = True
        self.remove_offsetting = True

        self.transactions = None

    @property
    def accounts(self):
        """returns the list of all accounts"""
        return self.transactions.Account.unique()

    @property
    def assets(self) -> dict:
        """returns a series of asset and their first transaction date"""
        return self.transactions.groupby("Asset").Date.min().to_dict()

    def to_disk(self, path: str) -> None:
        """saves the transactions to disk"""
        self.transactions.to_csv(path, index=False)

    def load_pregenerated_data(self, path) -> None:
        """loads the pregenerated data from disk"""
        self.transactions = pd.read_csv(path, parse_dates=["Date"])

    def _load_config(self, config=None):
        if config is None:
            self.config = configparser.ConfigParser()
            if len(self.config.read(config_helper.source)) != 1:
                raise OSError(f"no config found in {config_helper.source}")
        else:
            self.config = config

    def load_data_from_disk(self) -> None:
        """loads the account data from disk. the location needs to be specified in the config file"""
        df_raw = self._load_raw_data_from_disk()
        self.transactions: pd.DataFrame = self._enrich_raw_data(df_raw)[DataManager.COLUMNS].rename(
            {"Currency": "Asset", "Amount": "Quantity"}, axis=1
        )
        self.transactions["Date"] = normalize_date_column(self.transactions["Date"])

    def _load_raw_data_from_disk(self) -> pd.DataFrame:
        """returns the transaction data from file"""
        mt = master_transform.Loader(self.config)
        return mt.load_all(self.include_xls, self.include_json)

    def _enrich_raw_data(self, df_raw) -> pd.DataFrame:
        """return the transaction data enriched with the mapping"""
        pr = process.Process(self.config)
        return pr.process(
            df_raw,
            ignore_overrides=self.ignore_overrides,
            remove_offsetting=self.remove_offsetting,
        )

    @property
    def map_account_type(self):
        """returns a mapping between Account and AccountType"""
        return (
            self.transactions[["Account", "AccountType"]]
            .drop_duplicates()
            .sort_values("Account")
            .set_index("Account")
            .AccountType.fillna(" ")
        )

    @property
    def map_type(self):
        """returns a mapping between Type and FullType"""
        return {
            **self.transactions[["Type", "FullType"]].drop_duplicates().sort_values("Type").set_index("Type").FullType.fillna(" "),
            **{" ": " "},
        }

    @property
    def map_subtype(self):
        """returns a mapping between SubType and FullSubType"""
        return {
            **self.transactions[["SubType", "FullSubType"]]
            .drop_duplicates()
            .sort_values("SubType")
            .set_index("SubType")
            .FullSubType.fillna(" "),
            **{" ": " "},
        }

    @property
    def map_mastertype(self):
        """returns a mapping between MasterType and FullMasterType"""
        return (
            self.transactions[["MasterType", "FullMasterType"]]
            .drop_duplicates()
            .sort_values("MasterType")
            .set_index("MasterType")
            .FullMasterType.fillna(" ")
        )
