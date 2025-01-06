import configparser
from datetime import datetime

import pandas as pd

from bkanalysis.config import config_helper
from bkanalysis.market import market_loader
from bkanalysis.process import process
from bkanalysis.transforms import master_transform


def normalize_date_column(date_column):
    """Normalizes the date column to a consistent date format."""

    def convert_date(value):
        if isinstance(value, pd.Timestamp) or isinstance(value, datetime):
            return value.date()
        try:
            return pd.to_datetime(value).date()
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid date format: {value}") from exc

    return date_column.map(convert_date)


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
        self.transactions = self._enrich_raw_data(df_raw)[DataManager.COLUMNS].rename(
            {"Currency": "Asset", "Amount": "Quantity"}, axis=1
        )
        self.transactions["Date"] = normalize_date_column(self.transactions["Date"])

    def _load_raw_data_from_disk(self) -> pd.DataFrame:
        """returns the transaction data from file"""
        mt = master_transform.Loader(self.config, False)
        return mt.load_all(self.include_xls, self.include_json)

    def _enrich_raw_data(self, df_raw) -> pd.DataFrame:
        """return the transaction data enriched with the mapping"""
        pr = process.Process(self.config)
        return pr.process(
            df_raw,
            ignore_overrides=self.ignore_overrides,
            remove_offsetting=self.remove_offsetting,
        )

    def get_assets(self) -> pd.DataFrame:
        """returns a series of asset and their first transaction date"""
        return self.transactions.groupby("Asset").Date.min().to_dict()


class DataMarket:
    """Class to handle the interaction of the data with the market prices"""

    def __init__(self, data_manager: DataManager, ref_currency: str, config=None):
        self.data_manager = data_manager
        self.config = config
        self.ref_currency = ref_currency
        self.period = "10y"
        self.prices = None

    def get_asset_map(self) -> dict:
        """returns a map of the assets to their market symbol"""
        loader = market_loader.MarketLoader()
        return {
            instr: (
                loader.get_symbol(instr, "USD")
                if instr not in loader.source_map.keys()
                else instr
            )
            for instr in self.data_manager.transactions.Asset.dropna().unique()
        }

    def load_prices(self) -> None:
        """loads market prices from sources specified in the config file"""

        loader = market_loader.MarketLoader(self.config)
        market_prices = loader.load(
            self.data_manager.get_assets(), self.ref_currency, self.period
        )

        df_prices = DataMarket.__dataframe_from_nested_dict(market_prices)
        df_prices["Date"] = normalize_date_column(df_prices["Date"])

        df_fx = pd.concat(
            [
                df_prices[df_prices.Asset.str.startswith(ccy)]
                for ccy in df_prices.Currency.unique()
                if ccy != self.ref_currency
            ]
        )
        df_fx["Asset"] = df_fx["Asset"].str.replace(
            f"{self.ref_currency}=X", ""
        )  # TODO properly parse the asset to extract the ccy
        df_fx.rename(
            {"Currency": "RefCurrency", "Asset": "Currency", "Price": "Fx"},
            axis=1,
            inplace=True,
        )

        df = pd.merge(df_prices, df_fx, on=["Date", "Currency"], how="left")
        df.Fx = df.Fx.fillna(1)
        df["PriceInRefCurrency"] = df.Price * df.Fx

        self.prices = df[["Asset", "Date", "PriceInRefCurrency"]]

    def get_values(self):
        """returns the data_manager.transactions with the market prices"""
        # Initialize an empty list to store results
        results = {}

        # Process each asset individually
        unique_assets = self.data_manager.transactions["AssetMapped"].unique()
        for asset in unique_assets:
            if type(asset) != str:
                continue
            # Filter data for the current asset
            df_trans_asset = self.data_manager.transactions[
                self.data_manager.transactions["AssetMapped"] == asset
            ]
            df_prices_asset = self.prices[self.prices["AssetMapped"] == asset]

            date_range = sorted(
                set(
                    list(df_trans_asset.Date.unique())
                    + list(df_prices_asset.Date.unique())
                )
            )

            df_prices_asset = (
                df_prices_asset.set_index("Date")
                .reindex(date_range)
                .ffill()
                .bfill()
                .reset_index()
            )
            df_prices_asset = df_prices_asset[
                df_prices_asset["Date"].isin(df_trans_asset["Date"])
            ]
            merged = pd.merge(
                df_trans_asset, df_prices_asset, on=["AssetMapped", "Date"], how="left"
            )
            results[asset] = merged

        # Combine results for all assets
        return pd.concat(results.values(), ignore_index=True)

    @staticmethod
    def __dataframe_from_nested_dict(data):
        rows = []
        for asset, date_dict in data.items():
            for date, price_obj in date_dict.items():
                row = {
                    "Asset": asset,
                    "Date": date,
                    "Currency": price_obj.currency,
                    "Price": price_obj.value,
                }
                rows.append(row)
        return pd.DataFrame(rows)


class DataServer:
    """Class to prepare the data for the UI"""

    def __init__(self, data_manager: DataManager, config=None):
        self.data_manager = data_manager
        self.config = config

    def get_values_timeseries(self, filters) -> pd.DataFrame:
        """returns a timeseries of the values"""
        pass

    def get_values_by_types(self, filters) -> pd.DataFrame:
        """returns a breakdown by types"""
        pass

    def get_values_by_month(self, filters):
        """returns a breakdown by month"""
        pass
